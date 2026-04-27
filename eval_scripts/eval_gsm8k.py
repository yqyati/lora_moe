"""GSM8K 评估脚本（生成式 + 答案抽取）。

用法（单卡）:
    python eval_scripts/eval_gsm8k.py \
        --base_model allenai/OLMoE-1B-7B-0924 \
        --adapter_path ./output/v1_olmoe \
        --batch_size 4

用法（多卡，需要 torchrun）:
    torchrun --nproc_per_node=8 eval_scripts/eval_gsm8k.py \
        --base_model allenai/OLMoE-1B-7B-0924 \
        --adapter_path ./output/v1_olmoe \
        --batch_size 32
"""

import json
import os
import torch  
import torch.distributed as dist
from datasets import load_dataset
from tqdm import tqdm

from _common import common_arg_parser, load, generate_batch, extract_gsm8k_answer


PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "End your answer with '#### <number>'.\n\nQuestion: {question}\nAnswer:"
)


def main():
    args = common_arg_parser("Evaluate moe_lora checkpoint on GSM8K.").parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_dist = world_size > 1
    is_main = local_rank == 0

    if is_dist:                                                                                                                            
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")  

    tokenizer, model = load(args)

    ds = load_dataset("gsm8k", "main", split="test")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    prompts = [PROMPT_TEMPLATE.format(question=s["question"]) for s in ds]
    golds = [extract_gsm8k_answer(s["answer"]) for s in ds]

    # 每个 rank 处理 [rank, rank+world_size, rank+2*world_size, ...] 这一切片，
    # 这样把数据均匀分到所有 GPU 上，最后再 gather 起来。
    my_indices = list(range(local_rank, len(prompts), world_size)) if is_dist else list(range(len(prompts)))

    my_records = []
    batch_starts = list(range(0, len(my_indices), args.batch_size))
    iterator = tqdm(batch_starts, desc=f"GSM8K rank{local_rank}") if is_main else batch_starts

    for i in iterator:
        batch_idx = my_indices[i : i + args.batch_size]
        batch_prompts = [prompts[idx] for idx in batch_idx]
        completions = generate_batch(model, tokenizer, batch_prompts, args.max_new_tokens)
        for j, completion in enumerate(completions):
            idx = batch_idx[j]
            pred = extract_gsm8k_answer(completion)
            ok = pred is not None and pred == golds[idx]
            my_records.append({
                "idx": idx,
                "question": ds[idx]["question"],
                "gold": golds[idx],
                "pred": pred,
                "completion": completion,
                "correct": ok,
            })

    # 收集所有 rank 的 records 到 rank 0
    if is_dist:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, my_records)
        records = [r for sub in gathered for r in sub]
    else:
        records = my_records

    # 切片打乱了顺序，按 idx 排序还原
    records.sort(key=lambda r: r["idx"])

    if is_main:
        correct = sum(int(r["correct"]) for r in records)
        acc = correct / len(records)
        print(f"\nGSM8K accuracy: {acc:.4f} ({correct}/{len(records)})")

        if args.save_path:
            with open(args.save_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"Saved per-sample predictions to {args.save_path}")

    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""GSM8K 评估脚本（生成式 + 答案抽取）。

用法:
    python eval_scripts/eval_gsm8k.py \
        --base_model allenai/OLMoE-1B-7B-0924 \
        --adapter_path ./output/v1_olmoe \
        --batch_size 4
"""

import json
from datasets import load_dataset
from tqdm import tqdm

from _common import common_arg_parser, load, generate_batch, extract_gsm8k_answer


PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "End your answer with '#### <number>'.\n\nQuestion: {question}\nAnswer:"
)


def main():
    args = common_arg_parser("Evaluate moe_lora checkpoint on GSM8K.").parse_args()
    tokenizer, model = load(args)

    ds = load_dataset("gsm8k", "main", split="test")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    prompts = [PROMPT_TEMPLATE.format(question=s["question"]) for s in ds]
    golds = [extract_gsm8k_answer(s["answer"]) for s in ds]

    correct = 0
    records = []
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="GSM8K"):
        batch = prompts[i : i + args.batch_size]
        completions = generate_batch(model, tokenizer, batch, args.max_new_tokens)
        for j, completion in enumerate(completions):
            idx = i + j
            pred = extract_gsm8k_answer(completion)
            ok = pred is not None and pred == golds[idx]
            correct += int(ok)
            records.append({
                "idx": idx,
                "question": ds[idx]["question"],
                "gold": golds[idx],
                "pred": pred,
                "completion": completion,
                "correct": ok,
            })

    acc = correct / len(prompts)
    print(f"\nGSM8K accuracy: {acc:.4f} ({correct}/{len(prompts)})")

    if args.save_path:
        with open(args.save_path, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved per-sample predictions to {args.save_path}")


if __name__ == "__main__":
    main()

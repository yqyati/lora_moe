"""HumanEval 评估脚本（生成式 + 执行 unit test）。

依赖: pip install human-eval

用法（单卡）:
    python eval_scripts/eval_humaneval.py \
        --base_model allenai/OLMoE-1B-7B-0924 \
        --adapter_path ./output/v1_olmoe \
        --batch_size 4

用法（多卡，需要 torchrun）:
    torchrun --nproc_per_node=8 eval_scripts/eval_humaneval.py \
        --base_model allenai/OLMoE-1B-7B-0924 \
        --adapter_path ./output/v1_olmoe \
        --batch_size 16
"""

import json
import os
import re

import torch
import torch.distributed as dist
from tqdm import tqdm

from _common import common_arg_parser, load, generate_batch


def extract_code(completion: str, prompt: str) -> str:
    """从 completion 抽出可拼接在 prompt 后面的函数体代码。

    HumanEval 需要 prompt + completion 能拼成完整可执行的函数，
    所以 completion 应该是缩进的函数体，不含函数签名重复。
    """
    # 1) 如果有 markdown fence，先提取里面的内容
    m = re.search(r"```(?:python)?\n(.*?)```", completion, re.DOTALL)
    if m:
        completion = m.group(1)

    # 2) 如果模型重复了函数签名(def xxx)，只取签名之后的部分
    lines = completion.split("\n")
    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            # 找到函数体开始（跳过这个 def 行）
            completion = "\n".join(lines[i + 1:])
            break

    # 3) 截断：遇到新的顶层定义（非缩进 def/class/if __name__）时停止
    out_lines = []
    for line in completion.split("\n"):
        stripped = line.strip()
        if stripped and not line.startswith((" ", "\t")) and out_lines:
            if stripped.startswith(("def ", "class ", "if __name__", "print(", "assert ")):
                break
        out_lines.append(line)

    return "\n".join(out_lines)


def main():
    args = common_arg_parser("Evaluate moe_lora checkpoint on HumanEval.").parse_args()

    try:
        from human_eval.data import read_problems
        from human_eval.execution import check_correctness
    except ImportError:
        raise SystemExit("Please install human-eval: pip install human-eval")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_dist = world_size > 1
    is_main = local_rank == 0

    if is_dist:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    tokenizer, model = load(args)

    problems = read_problems()
    items = list(problems.items())
    if args.limit:
        items = items[: args.limit]

    prompts = [v["prompt"] for _, v in items]

    my_indices = list(range(local_rank, len(prompts), world_size)) if is_dist else list(range(len(prompts)))

    my_records = []
    batch_starts = list(range(0, len(my_indices), args.batch_size))
    iterator = tqdm(batch_starts, desc=f"HumanEval rank{local_rank}") if is_main else batch_starts

    for i in iterator:
        batch_idx = my_indices[i : i + args.batch_size]
        batch_prompts = [prompts[idx] for idx in batch_idx]
        completions = generate_batch(model, tokenizer, batch_prompts, args.max_new_tokens)
        for j, raw in enumerate(completions):
            idx = batch_idx[j]
            task_id, problem = items[idx]
            code = extract_code(raw, problem["prompt"])
            try:
                result = check_correctness(problem, code, timeout=10.0)
                ok = result["passed"]
            except Exception as e:
                ok = False
                result = {"passed": False, "result": f"exception: {e}"}
            my_records.append({
                "idx": idx,
                "task_id": task_id,
                "completion": code,
                "passed": ok,
                "exec_result": result.get("result", ""),
            })

    if is_dist:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, my_records)
        records = [r for sub in gathered for r in sub]
    else:
        records = my_records

    records.sort(key=lambda r: r["idx"])

    if is_main:
        correct = sum(int(r["passed"]) for r in records)
        acc = correct / len(records)
        print(f"\nHumanEval pass@1: {acc:.4f} ({correct}/{len(records)})")

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

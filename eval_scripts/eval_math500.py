"""MATH-500 评估脚本（OpenAI PRM800K 论文用的 MATH 子集）。

用法（单卡）:
    python eval_scripts/eval_math500.py \
        --base_model allenai/OLMoE-1B-7B-0924 \
        --adapter_path ./output/v1_olmoe \
        --batch_size 4

用法（多卡，需要 torchrun）:
    torchrun --nproc_per_node=8 eval_scripts/eval_math500.py \
        --base_model allenai/OLMoE-1B-7B-0924 \
        --adapter_path ./output/v1_olmoe \
        --batch_size 32
"""

import json
import os
import re

import torch
import torch.distributed as dist
from datasets import load_dataset
from tqdm import tqdm

from _common import common_arg_parser, load, generate_batch


PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "Put your final answer in \\boxed{{...}}.\n\n"
    "Problem: {problem}\nSolution:"
)


def extract_boxed(text: str):
    """抽出 \\boxed{...} 里的内容。处理嵌套花括号。"""
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    i = idx + len("\\boxed")
    while i < len(text) and text[i] != "{":
        i += 1
    if i >= len(text):
        return None
    depth = 0
    start = i
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : j].strip()
    return None


def normalize(s: str) -> str:
    """MATH-500 答案归一化(强化版,处理常见 latex 等价形式)。"""
    if s is None:
        return ""
    # 1. 去空格 + 各种空白 latex 命令
    s = s.replace(" ", "").replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\:", "")
    # 2. 去 \left / \right 修饰(集合 / 坐标对常见)
    s = s.replace("\\left", "").replace("\\right", "")
    # 3. 统一分数:dfrac/tfrac → frac,a/b → \frac{a}{b}
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    # 简单分数 (-?数字)/(数字) → \frac{a}{b}(必须是整体分数,不能伤及 \frac{1}{2} 等)
    s = re.sub(r"(?<!\w)(-?\d+)/(\d+)(?!\w)", r"\\frac{\1}{\2}", s)
    # 4. 统一 \sqrt: \sqrt2 → \sqrt{2}
    s = re.sub(r"\\sqrt(\d+)", r"\\sqrt{\1}", s)
    # 5. 浮点小数末尾零 → 整数(5.0 → 5)
    s = re.sub(r"(?<![\d.])(-?\d+)\.0+(?![\d.])", r"\1", s)
    # 6. 指数大括号:x^2 ↔ x^{2}(统一去掉单字符大括号)
    s = re.sub(r"\^\{(\w)\}", r"^\1", s)
    # 7. \pi vs pi 等希腊字母
    s = s.replace("\\pi", "pi").replace("\\theta", "theta")
    # 8. \cdot / \times → *
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    # 9. 末尾标点
    s = s.rstrip(".")
    return s.lower()


def main():
    args = common_arg_parser("Evaluate moe_lora checkpoint on MATH-500.").parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_dist = world_size > 1
    is_main = local_rank == 0

    if is_dist:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    tokenizer, model = load(args)

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    prompts = [PROMPT_TEMPLATE.format(problem=s["problem"]) for s in ds]

    my_indices = list(range(local_rank, len(prompts), world_size)) if is_dist else list(range(len(prompts)))

    my_records = []
    batch_starts = list(range(0, len(my_indices), args.batch_size))
    iterator = tqdm(batch_starts, desc=f"MATH-500 rank{local_rank}") if is_main else batch_starts

    for i in iterator:
        batch_idx = my_indices[i : i + args.batch_size]
        batch_prompts = [prompts[idx] for idx in batch_idx]
        completions = generate_batch(model, tokenizer, batch_prompts, args.max_new_tokens)
        for j, completion in enumerate(completions):
            idx = batch_idx[j]
            pred = extract_boxed(completion)
            gold = ds[idx]["answer"]
            ok = normalize(pred) == normalize(gold)
            my_records.append({
                "idx": idx,
                "problem": ds[idx]["problem"],
                "gold": gold,
                "pred": pred,
                "completion": completion,
                "correct": ok,
            })

    if is_dist:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, my_records)
        records = [r for sub in gathered for r in sub]
    else:
        records = my_records

    records.sort(key=lambda r: r["idx"])

    if is_main:
        correct = sum(int(r["correct"]) for r in records)
        acc = correct / len(records)
        print(f"\nMATH-500 accuracy: {acc:.4f} ({correct}/{len(records)})")
        print("Note: 用宽松字符串匹配；严格做法请用 sympy 求等价。")

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

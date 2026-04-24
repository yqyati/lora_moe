"""MATH-500 评估脚本（OpenAI PRM800K 论文用的 MATH 子集）。

用法:
    python eval_scripts/eval_math500.py \
        --base_model allenai/OLMoE-1B-7B-0924 \
        --adapter_path ./output/v1_olmoe \
        --batch_size 4
"""

import json
import re
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
    # 找开始的 {
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
    """简单归一化：去空格、去 \\ 前缀、统一分数 / 等价表达。
    完整版应该用 sympy 求等价，这里用宽松字符串匹配。"""
    if s is None:
        return ""
    s = s.replace(" ", "").replace("\\!", "").replace("\\,", "").replace("\\;", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.rstrip(".")
    return s.lower()


def main():
    args = common_arg_parser("Evaluate moe_lora checkpoint on MATH-500.").parse_args()
    tokenizer, model = load(args)

    # MATH-500 在 HuggingFaceH4/MATH-500 上
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    prompts = [PROMPT_TEMPLATE.format(problem=s["problem"]) for s in ds]

    correct = 0
    records = []
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="MATH-500"):
        batch = prompts[i : i + args.batch_size]
        completions = generate_batch(model, tokenizer, batch, args.max_new_tokens)
        for j, completion in enumerate(completions):
            idx = i + j
            pred = extract_boxed(completion)
            gold = ds[idx]["answer"]
            ok = normalize(pred) == normalize(gold)
            correct += int(ok)
            records.append({
                "idx": idx,
                "problem": ds[idx]["problem"],
                "gold": gold,
                "pred": pred,
                "completion": completion,
                "correct": ok,
            })

    acc = correct / len(prompts)
    print(f"\nMATH-500 accuracy: {acc:.4f} ({correct}/{len(prompts)})")
    print("Note: 用宽松字符串匹配；严格做法请用 sympy 求等价。")

    if args.save_path:
        with open(args.save_path, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved per-sample predictions to {args.save_path}")


if __name__ == "__main__":
    main()

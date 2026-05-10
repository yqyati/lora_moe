"""MATH-500 调试脚本:对比 raw prompt vs chat_mode prompt 的输出。

仅用于诊断 — 不计算 accuracy,只把每个 sample 的完整输入输出打印出来,
让你眼睛判断:
  1. 模型实际产出长什么样(R1-style? \\boxed{}? 还是乱写?)
  2. raw prompt 和 chat_mode 哪个让模型更倾向于按训练分布生成
  3. 当前 extract_boxed + normalize 是否真的把对的判错了

用法:
    python eval_scripts/eval_math500_debug.py \\
        --base_model allenai/OLMoE-1B-7B-0924 \\
        --adapter_path saves/olmoe/moe_lora/v2_global_pool128_best \\
        --limit 5
"""

import os
import sys
from datasets import load_dataset

from _common import common_arg_parser, load, generate_batch, wrap_default_chat

# 同 eval_math500.py 的原 prompt(raw 模式)
RAW_PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "Put your final answer in \\boxed{{...}}.\n\n"
    "Problem: {problem}\nSolution:"
)

# chat_mode 用户消息(只包"题目要求",外层 wrap_default_chat 加 Human:/Assistant:)
CHAT_USER_MSG_TEMPLATE = (
    "Solve the following math problem step by step. "
    "Put your final answer in \\boxed{{...}}.\n\n"
    "Problem: {problem}"
)


def extract_boxed(text: str):
    """同 eval_math500.py — 抽最后一个 \\boxed{...}。"""
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
    if s is None:
        return ""
    s = s.replace(" ", "").replace("\\!", "").replace("\\,", "").replace("\\;", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.rstrip(".")
    return s.lower()


def main():
    parser = common_arg_parser("Debug MATH-500 eval: raw vs chat_mode side-by-side.")
    parser.add_argument("--max_print", type=int, default=5,
                        help="最多打印几个 sample(默认 5,够看出规律)")
    args = parser.parse_args()

    # 强制单卡 + 限制样本(只是 debug 不需要分布式)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        print("⚠️  此脚本仅支持单卡 debug,请直接 python(不用 torchrun)")
        sys.exit(1)

    if args.limit is None:
        args.limit = args.max_print

    tokenizer, model = load(args)

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    ds = ds.select(range(min(args.max_print, len(ds))))

    print("\n" + "=" * 80)
    print(f"DEBUG: MATH-500 raw vs chat_mode (n={len(ds)})")
    print("=" * 80)

    for i, sample in enumerate(ds):
        problem = sample["problem"]
        gold = sample["answer"]

        # 两种 prompt
        raw_prompt = RAW_PROMPT_TEMPLATE.format(problem=problem)
        chat_user_msg = CHAT_USER_MSG_TEMPLATE.format(problem=problem)
        chat_prompt = wrap_default_chat(tokenizer, chat_user_msg)

        # 两次生成(各自 max_new_tokens)
        raw_out = generate_batch(model, tokenizer, [raw_prompt], args.max_new_tokens)[0]
        chat_out = generate_batch(model, tokenizer, [chat_prompt], args.max_new_tokens)[0]

        raw_pred = extract_boxed(raw_out)
        chat_pred = extract_boxed(chat_out)

        raw_correct = normalize(raw_pred) == normalize(gold)
        chat_correct = normalize(chat_pred) == normalize(gold)

        print(f"\n{'#' * 80}")
        print(f"# Sample {i}  |  Gold: '{gold}'")
        print(f"{'#' * 80}")
        print(f"\nProblem: {problem[:300]}{'...' if len(problem) > 300 else ''}")

        print(f"\n--- [RAW prompt] (current eval_math500.py) ---")
        print(f"Prompt: {raw_prompt[:200]}...")
        print(f"\nCompletion (前 1500 字符):\n{raw_out[:1500]}")
        print(f"\n  Extracted boxed: '{raw_pred}'")
        print(f"  normalize match: {raw_correct}")

        print(f"\n--- [CHAT prompt] (Human:/Assistant: 包装) ---")
        print(f"Prompt: {chat_prompt[:300]}...")
        print(f"\nCompletion (前 1500 字符):\n{chat_out[:1500]}")
        print(f"\n  Extracted boxed: '{chat_pred}'")
        print(f"  normalize match: {chat_correct}")

        print(f"\n--- [对比] ---")
        print(f"  raw correct:  {raw_correct}")
        print(f"  chat correct: {chat_correct}")

    print(f"\n{'=' * 80}")
    print("看完上面输出后,你应该能判断:")
    print("  1. 模型是否在 raw 模式下产出 R1-style 但没 \\boxed{}")
    print("  2. chat_mode 是否让模型回到训练分布的输出格式")
    print("  3. extract_boxed / normalize 有没有把对的判错")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

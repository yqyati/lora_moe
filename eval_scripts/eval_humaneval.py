"""HumanEval 评估脚本（生成式 + 执行 unit test）。

依赖: pip install human-eval

用法:
    python eval_scripts/eval_humaneval.py \
        --base_model allenai/OLMoE-1B-7B-0924 \
        --adapter_path ./output/v1_olmoe \
        --batch_size 4
"""

import json
import re
from tqdm import tqdm

from _common import common_arg_parser, load, generate_batch


def extract_code(completion: str) -> str:
    """从 completion 抽出 Python 代码块。优先 ```python ... ```，否则原文。"""
    m = re.search(r"```(?:python)?\n(.*?)```", completion, re.DOTALL)
    if m:
        return m.group(1)
    # 没 markdown，截到第一个非缩进行（认为是函数体外）
    lines = completion.split("\n")
    out_lines = []
    for line in lines:
        if line.strip() and not line.startswith((" ", "\t")) and out_lines:
            # 函数体已经写完
            break
        out_lines.append(line)
    return "\n".join(out_lines)


def main():
    args = common_arg_parser("Evaluate moe_lora checkpoint on HumanEval.").parse_args()
    tokenizer, model = load(args)

    try:
        from human_eval.data import read_problems
        from human_eval.execution import check_correctness
    except ImportError:
        raise SystemExit("Please install human-eval: pip install human-eval")

    problems = read_problems()
    items = list(problems.items())
    if args.limit:
        items = items[: args.limit]

    prompts = [v["prompt"] for _, v in items]

    correct = 0
    records = []
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="HumanEval"):
        batch = prompts[i : i + args.batch_size]
        completions = generate_batch(model, tokenizer, batch, args.max_new_tokens)
        for j, raw in enumerate(completions):
            task_id, problem = items[i + j]
            code = extract_code(raw)
            # check_correctness 需要 prompt + completion 拼接
            full = problem["prompt"] + code
            sample = {"task_id": task_id, "completion": code}
            try:
                result = check_correctness(problem, sample, timeout=10.0)
                ok = result["passed"]
            except Exception as e:
                ok = False
                result = {"passed": False, "result": f"exception: {e}"}
            correct += int(ok)
            records.append({
                "task_id": task_id,
                "completion": code,
                "passed": ok,
                "exec_result": result.get("result", ""),
            })

    acc = correct / len(prompts)
    print(f"\nHumanEval pass@1: {acc:.4f} ({correct}/{len(prompts)})")

    if args.save_path:
        with open(args.save_path, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved per-sample predictions to {args.save_path}")


if __name__ == "__main__":
    main()

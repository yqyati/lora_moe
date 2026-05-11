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

from _common import common_arg_parser, load, generate_batch, wrap_default_chat


# 用户消息模板,模仿训练数据(magicoder)的形态:
# 自然语言任务描述 + 提供函数签名(带 # Your code here 占位) + 明确要求补全
_USER_MSG_TEMPLATE = (
    "You are tasked with implementing a Python function. "
    "You are provided with the following function signature and docstring as a starting point:\n\n"
    "```python\n{prompt}    # Your code here\n```\n\n"
    "Your task is to complete the function body so that it satisfies the docstring's requirements. "
    "Return the complete implemented function (including the signature) inside a ```python``` code block."
)


def extract_python_block(completion: str) -> str:
    """从 assistant 回复里抽 ```python ... ``` 块作为完整函数。"""
    # 去掉 think 块
    completion = re.sub(r"<think>.*?</think>", "", completion, flags=re.DOTALL)
    # 优先抽 ```python 块,其次任意 ``` 块
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", completion, re.DOTALL)
    if m:
        return m.group(1).rstrip()
    # fence 没闭合(模型生成被截断):抽 ```python 之后到末尾
    m = re.search(r"```(?:python|py)?\s*\n(.*)$", completion, re.DOTALL)
    if m:
        return m.group(1).rstrip()
    # 无 fence:直接返回原文兜底
    return completion.rstrip()


def extract_code_raw(completion: str, prompt: str) -> str:
    """raw 模式(原始 base 模型续写,无 chat wrapper):
    返回作为 prompt 后缀拼接的函数体。

    base 模型续写时常乱编新函数(`def helper`、`def variant_xxx`),
    所以**不能**像之前那样"取最后一个 def 之后的 body" —— 那会把废话
    函数的 docstring 当目标函数体。改成直接交给 step 3:第一遇到顶层
    非缩进的 def/class/裸语句就截断,只保留缩进的函数体部分。
    """
    completion = re.sub(r"<think>.*?</think>", "", completion, flags=re.DOTALL)
    completion = completion.rstrip()
    completion = completion.lstrip("\n")

    m = re.search(r"```(?:python)?\s*\n(.*?)```", completion, re.DOTALL)
    if m:
        completion = m.group(1)

    out_lines = []
    for line in completion.split("\n"):
        stripped = line.strip()
        # 顶层非缩进的非空非注释行 → 函数体已结束,截断
        if stripped and not line.startswith((" ", "\t")) and out_lines:
            if stripped.startswith(("def ", "class ", "if __name__", "print(", "assert ",
                                    "# Test", "# Example", "# Check")):
                break
            if not stripped.startswith("#"):
                break
        out_lines.append(line)

    while out_lines and not out_lines[-1].strip():
        out_lines.pop()

    result = "\n".join(out_lines)
    if not result.strip():
        result = completion
    return result


def pass_at_k(n, c, k):
    """计算 pass@k（组合公式）：n 次采样中 c 次通过时，随机选 k 个至少有 1 个通过的概率。

    pass@k = 1 - C(n-c, k) / C(n, k)
    """
    if n - c < k:
        return 1.0
    if c == 0:
        return 0.0
    # 用连乘避免大数阶乘溢出
    result = 1.0
    for i in range(k):
        result *= (n - c - i) / (n - i)
    return 1.0 - result


def main():
    parser = common_arg_parser("Evaluate moe_lora checkpoint on HumanEval.")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="每题生成几次（pass@k 的 n）")
    parser.add_argument("--k", type=int, default=1,
                        help="pass@k 的 k 值")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="采样温度，0=greedy，>0 时启用 sampling")
    parser.add_argument("--chat_mode", action="store_true",
                        help="用 LlamaFactory `default` chat template 包装 prompt;"
                             "评测 SFT/chat 微调过的 checkpoint 时必开。")
    args = parser.parse_args()

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

    raw_prompts = [v["prompt"] for _, v in items]
    if args.chat_mode:
        # 自动检测 Qwen3:用 Qwen3 chat template + 关闭 thinking
        is_qwen3 = (
            tokenizer.chat_template is not None
            and "qwen" in str(args.base_model).lower()
        )
        if is_qwen3:
            def _wrap_qwen3(p):
                msgs = [{"role": "user", "content": _USER_MSG_TEMPLATE.format(prompt=p)}]
                try:
                    return tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    return tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True,
                    )
            gen_prompts = [_wrap_qwen3(p) for p in raw_prompts]
        else:
            # OLMoE / 其他默认: 用 LlamaFactory default template
            gen_prompts = [
                wrap_default_chat(tokenizer, _USER_MSG_TEMPLATE.format(prompt=p))
                for p in raw_prompts
            ]
    else:
        gen_prompts = raw_prompts

    my_indices = list(range(local_rank, len(gen_prompts), world_size)) if is_dist else list(range(len(gen_prompts)))

    my_records = []
    batch_starts = list(range(0, len(my_indices), args.batch_size))
    num_samples = args.num_samples

    for sample_round in range(num_samples):
        if is_main and num_samples > 1:
            print(f"\n--- Sample round {sample_round + 1}/{num_samples} ---")
        iterator = tqdm(batch_starts, desc=f"HumanEval rank{local_rank} round{sample_round+1}") if is_main else batch_starts

        for i in iterator:
            batch_idx = my_indices[i : i + args.batch_size]
            batch_prompts = [gen_prompts[idx] for idx in batch_idx]
            completions = generate_batch(model, tokenizer, batch_prompts, args.max_new_tokens,
                                         temperature=args.temperature)
            for j, raw in enumerate(completions):
                idx = batch_idx[j]
                task_id, problem = items[idx]
                if args.chat_mode:
                    # chat 模式:抽出 ```python``` 里的完整函数,用空 prompt 方式交给 check_correctness
                    full_code = extract_python_block(raw)
                    fake_problem = {**problem, "prompt": ""}
                    try:
                        result = check_correctness(fake_problem, full_code, timeout=10.0)
                        ok = result["passed"]
                    except Exception as e:
                        ok = False
                        result = {"passed": False, "result": f"exception: {e}"}
                    saved_code = full_code
                else:
                    # raw 模式:抽函数体,拼在 prompt 后面
                    code = extract_code_raw(raw, problem["prompt"])
                    try:
                        result = check_correctness(problem, code, timeout=10.0)
                        ok = result["passed"]
                    except Exception as e:
                        ok = False
                        result = {"passed": False, "result": f"exception: {e}"}
                    saved_code = code
                my_records.append({
                    "idx": idx,
                    "task_id": task_id,
                    "sample_round": sample_round,
                    "completion": saved_code,
                    "raw": raw,
                    "passed": ok,
                    "exec_result": result.get("result", ""),
                })

    if is_dist:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, my_records)
        records = [r for sub in gathered for r in sub]
    else:
        records = my_records

    records.sort(key=lambda r: (r["idx"], r["sample_round"]))

    if is_main:
        k = args.k
        # 按 task 聚合
        from collections import defaultdict
        task_results = defaultdict(list)
        for r in records:
            task_results[r["task_id"]].append(r["passed"])

        # 计算 pass@k
        total_pass_at_k = 0.0
        for task_id, results in task_results.items():
            n = len(results)
            c = sum(results)
            total_pass_at_k += pass_at_k(n, c, k)
        score = total_pass_at_k / len(task_results)
        total_tasks = len(task_results)
        print(f"\nHumanEval pass@{k}: {score:.4f} (n={num_samples}, {total_tasks} problems)")

        if args.save_path:
            os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
            with open(args.save_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"Saved per-sample predictions to {args.save_path}")

    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

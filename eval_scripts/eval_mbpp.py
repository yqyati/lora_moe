"""MBPP 评估脚本（生成式 + 执行 assert 测试）。

用法（单卡）:
    python eval_scripts/eval_mbpp.py \
        --base_model allenai/OLMoE-1B-7B-0924 \
        --adapter_path ./output/v1_olmoe \
        --batch_size 4

用法（多卡，需要 torchrun）:
    torchrun --nproc_per_node=8 eval_scripts/eval_mbpp.py \
        --base_model allenai/OLMoE-1B-7B-0924 \
        --adapter_path ./output/v1_olmoe \
        --batch_size 32
"""

import json
import os
import re
import signal
import tempfile
import traceback
from contextlib import contextmanager

import torch
import torch.distributed as dist
from datasets import load_dataset
from tqdm import tqdm

from _common import common_arg_parser, load, generate_batch


PROMPT_TEMPLATE = (
    "You are an expert Python programmer. Write a Python function to solve the following task.\n\n"
    "Task: {text}\n"
    "Your code should satisfy the following test cases:\n{tests}\n\n"
    "```python\n"
)


def build_parser():
    p = common_arg_parser("Evaluate moe_lora checkpoint on MBPP.")
    p.add_argument("--subset", default="sanitized", choices=["sanitized", "full"],
                   help="MBPP subset: 'sanitized' (427 题, 推荐) 或 'full' (974 题)")
    p.add_argument("--timeout", type=float, default=10.0, help="每道题执行超时（秒）")
    return p


def extract_code(completion: str) -> str:
    """从 completion 抽出 Python 代码。

    注意:prompt 末尾是 "```python\\n",模型从 fence 内部开始写,所以 completion
    通常形如 "<code>\\n```\\n<explanation>",只有闭合 ``` 没有开头 ```。
    """
    # 1) 有完整 ```python ... ``` 块(模型自己写了开头 fence)→ 抽里面
    m = re.search(r"```(?:python)?\n(.*?)```", completion, re.DOTALL)
    if m:
        return m.group(1)

    # 2) 没完整块 → 截到第一个 ``` (闭合 fence)之前
    fence_idx = completion.find("```")
    if fence_idx >= 0:
        return completion[:fence_idx]

    # 3) 完全没 fence → 截到第一个顶层裸语句(避免把解释文本当代码)
    out_lines = []
    for line in completion.split("\n"):
        stripped = line.strip()
        if stripped and not line.startswith((" ", "\t")) and out_lines:
            if stripped.startswith(("def ", "class ", "import ", "from ", "@")) or stripped.startswith("#"):
                out_lines.append(line)
                continue
            break
        out_lines.append(line)
    return "\n".join(out_lines)


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError(f"Timed out after {seconds} seconds")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)


def run_tests(code: str, tests: list, timeout: float) -> dict:
    """执行生成的代码 + assert 测试用例，返回 {passed: bool, error: str|None}。"""
    full_code = code + "\n" + "\n".join(tests)
    try:
        with time_limit(timeout):
            exec(full_code, {"__builtins__": __builtins__}, {})
        return {"passed": True, "error": None}
    except TimeoutError as e:
        return {"passed": False, "error": str(e)}
    except Exception as e:
        return {"passed": False, "error": f"{type(e).__name__}: {e}"}


def main():
    args = build_parser().parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_dist = world_size > 1
    is_main = local_rank == 0

    if is_dist:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    tokenizer, model = load(args)

    # 加载 MBPP 数据集
    if args.subset == "sanitized":
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    else:
        # full 版本: 测试集是 11-510 (task_id 11..510)
        ds = load_dataset("google-research-datasets/mbpp", "full", split="test")

    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    # 构造 prompt
    # 检测模型类型: Qwen3 / OLMoE / 其他
    # tokenizer.chat_template 存在 → 用 apply_chat_template; 否则裸 prompt
    use_chat_template = tokenizer.chat_template is not None and "qwen" in str(args.base_model).lower()

    prompts = []
    for sample in ds:
        task_text = sample.get("text") or sample.get("prompt")
        test_str = "\n".join(sample["test_list"][:3])
        raw_prompt = PROMPT_TEMPLATE.format(text=task_text, tests=test_str)
        if use_chat_template:
            # Qwen3: 把 raw_prompt 当 user message,套 chat template,end with assistant 起始
            messages = [{"role": "user", "content": raw_prompt}]
            try:
                # Qwen3 默认 thinking mode 会输出 <think>...</think>,占满 max_new_tokens
                # → 强制关闭 thinking,模型直接输出代码
                wrapped = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                # 老版 tokenizer 不支持 enable_thinking 参数
                wrapped = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            # 在 assistant 起始后立刻接 ```python\n,引导模型直接写代码
            if not wrapped.endswith("```python\n"):
                wrapped = wrapped.rstrip() + "\n```python\n"
            prompts.append(wrapped)
        else:
            prompts.append(raw_prompt)

    my_indices = list(range(local_rank, len(prompts), world_size)) if is_dist else list(range(len(prompts)))

    my_records = []
    batch_starts = list(range(0, len(my_indices), args.batch_size))
    iterator = tqdm(batch_starts, desc=f"MBPP rank{local_rank}") if is_main else batch_starts

    for i in iterator:
        batch_idx = my_indices[i : i + args.batch_size]
        batch_prompts = [prompts[idx] for idx in batch_idx]
        completions = generate_batch(model, tokenizer, batch_prompts, args.max_new_tokens)
        for j, completion in enumerate(completions):
            idx = batch_idx[j]
            code = extract_code(completion)
            # 用全部测试用例来判定
            all_tests = ds[idx]["test_list"]
            result = run_tests(code, all_tests, args.timeout)
            my_records.append({
                "idx": idx,
                "task_id": ds[idx].get("task_id", idx),
                "text": ds[idx].get("text") or ds[idx].get("prompt"),
                "code": code,
                "passed": result["passed"],
                "error": result["error"],
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
        print(f"\nMBPP ({args.subset}) pass@1: {acc:.4f} ({correct}/{len(records)})")

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

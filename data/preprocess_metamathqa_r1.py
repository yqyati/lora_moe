"""下载 oumi-ai/MetaMathQA-R1，删掉 <think>...</think>，保存为 LlamaFactory sharegpt 格式。

R1 数据 schema:
    - prompt: str            用户问题
    - response: str          含 <think>...</think>...答案 的完整 R1 输出
    - messages: list[dict]   [{role: user, content: prompt}, {role: assistant, content: response}]
    - metadata: dict         含 'finish_reason' 等

我们删掉 <think>...</think> 段（包括标签本身），只保留最终答案。

用法:
    python data/preprocess_metamathqa_r1.py
    python data/preprocess_metamathqa_r1.py --limit 50000 --output_path data/metamathqa_r1_clean_50k.json
    python data/preprocess_metamathqa_r1.py --keep_think         # 调试: 不删 think
"""

import argparse
import json
import os
import re
import sys
from typing import Dict


THINK_PATTERN = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)


def strip_think(text: str) -> str:
    """删掉 <think>...</think> 段，包括标签和后面的换行。"""
    if not text:
        return text
    cleaned = THINK_PATTERN.sub("", text)
    # 极少数样本可能 <think> 没闭合，保守做法是从 <think> 截到 </think> 或丢弃整个 think 之后到末尾
    # 但 R1 输出绝大多数闭合，这里只 warning 不处理
    if "<think>" in cleaned or "</think>" in cleaned:
        return cleaned  # 残留也直接返回，避免内容被误删
    return cleaned.strip()


def convert_sample(sample: Dict, keep_think: bool) -> Dict:
    """把 R1 样本转成 LlamaFactory sharegpt 格式。"""
    user_msg = sample.get("prompt") or ""
    assistant_msg = sample.get("response") or ""
    if not keep_think:
        assistant_msg = strip_think(assistant_msg)

    return {
        "messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--hf_path",
        default="oumi-ai/MetaMathQA-R1",
        help="HuggingFace dataset id (or local path).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use.",
    )
    parser.add_argument(
        "--output_path",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "metamathqa_r1_clean.json"),
        help="Output JSON path (LlamaFactory sharegpt format).",
    )
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 个样本（debug / 控制规模）")
    parser.add_argument("--keep_think", action="store_true", help="保留 <think>...</think> 段（默认删除）")
    parser.add_argument(
        "--filter_incomplete",
        action="store_true",
        default=True,
        help="跳过 metadata 标记为不完整（无 </think>）的样本，默认 True",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install datasets: pip install datasets", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.hf_path} (split={args.split}) ...")
    ds = load_dataset(args.hf_path, split=args.split)
    print(f"Loaded {len(ds)} samples")

    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
        print(f"Limited to {len(ds)} samples")

    out_records = []
    skipped_incomplete = 0
    skipped_empty = 0
    n_think_removed = 0

    for sample in ds:
        # 过滤不完整样本（R1 输出被截断、没有 </think>）
        if args.filter_incomplete:
            meta = sample.get("metadata") or {}
            # README 提到用 metadata 标记是否完整
            if isinstance(meta, dict) and meta.get("complete") is False:
                skipped_incomplete += 1
                continue

        rec = convert_sample(sample, keep_think=args.keep_think)
        # 跳过空 response（清洗后可能空）
        if not rec["messages"][1]["content"].strip():
            skipped_empty += 1
            continue

        # 统计有多少样本被删过 think
        if not args.keep_think and "<think>" in (sample.get("response") or ""):
            n_think_removed += 1

        out_records.append(rec)

    print(f"Done. Final: {len(out_records)} samples")
    print(f"  - skipped (incomplete): {skipped_incomplete}")
    print(f"  - skipped (empty response): {skipped_empty}")
    print(f"  - <think> removed from {n_think_removed} samples")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(out_records, f, ensure_ascii=False, indent=None)

    size_mb = os.path.getsize(args.output_path) / 1024 / 1024
    print(f"Saved to {args.output_path} ({size_mb:.1f} MB)")
    print("\nNext step: 在 data/dataset_info.json 中已注册 'metamathqa_r1'，可以直接 --dataset metamathqa_r1 训练。")


if __name__ == "__main__":
    main()

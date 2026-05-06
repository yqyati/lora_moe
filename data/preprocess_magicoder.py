"""下载 ise-uiuc/Magicoder-OSS-Instruct-75K，转为 LlamaFactory sharegpt 格式。

数据集 schema:
    - problem: str     编程指令/问题描述
    - solution: str    对应的代码解答

用法:
    python data/preprocess_magicoder.py
    python data/preprocess_magicoder.py --limit 10000
    python data/preprocess_magicoder.py --output_path data/magicoder_75k.json

需要设置:
    export HF_ENDPOINT=https://hf-mirror.com   # 国内加速
"""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Convert Magicoder-OSS-Instruct-75K to LlamaFactory sharegpt format")
    parser.add_argument("--limit", type=int, default=None, help="只保留前 N 条（debug 用）")
    parser.add_argument("--output_path", type=str, default="data/magicoder_oss_75k.json",
                        help="输出路径")
    args = parser.parse_args()

    from datasets import load_dataset

    print("Loading ise-uiuc/Magicoder-OSS-Instruct-75K ...")
    ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
    print(f"Loaded {len(ds)} samples")

    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
        print(f"Limited to {len(ds)} samples")

    converted = []
    for sample in ds:
        # Magicoder 字段: problem, solution
        user_msg = sample.get("problem") or sample.get("instruction") or ""
        assistant_msg = sample.get("solution") or sample.get("output") or ""

        if not user_msg.strip() or not assistant_msg.strip():
            continue

        converted.append({
            "messages": [
                {"role": "user", "content": user_msg.strip()},
                {"role": "assistant", "content": assistant_msg.strip()},
            ]
        })

    # 保存
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=None)

    print(f"Saved {len(converted)} samples to {args.output_path}")


if __name__ == "__main__":
    main()

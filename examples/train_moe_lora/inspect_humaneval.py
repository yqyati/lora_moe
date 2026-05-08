"""诊断 /tmp/humaneval_debug.jsonl:
- 打印头 3 个样本的 raw 模型输出 / 抽取后代码 / 执行结果
- 统计失败原因分布

用法:
    python examples/train_moe_lora/inspect_humaneval.py
    python examples/train_moe_lora/inspect_humaneval.py /tmp/other_file.jsonl
"""

import json
import sys
from collections import Counter


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/humaneval_debug.jsonl"
    rs = [json.loads(l) for l in open(path)]

    print(f"File: {path}")
    print(f"Pass: {sum(r['passed'] for r in rs)}/{len(rs)}")

    print("\n=== 头 3 个样本的 RAW 输出(模型原文) ===")
    for r in rs[:3]:
        print(f"\n--- {r['task_id']} (round {r['sample_round']}) ---")
        print("RAW:", repr(r.get("raw", "<no raw field>")[:600]))
        print()
        print("EXTRACTED:", repr(r["completion"][:300]))
        print("RESULT:", r["exec_result"][:120])

    print("\n=== 失败原因分布 ===")
    fails = [r.get("exec_result", "")[:90] for r in rs if not r["passed"]]
    for reason, n in Counter(fails).most_common(10):
        print(f"{n:4d}  {reason!r}")


if __name__ == "__main__":
    main()

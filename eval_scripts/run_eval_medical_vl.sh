#!/bin/bash
# Run medical VL eval on all 3 benchmarks (VQA-RAD / SLAKE / PathVQA) for one adapter.
#
# Usage:
#   bash eval_scripts/run_eval_medical_vl.sh <adapter_path> <run_name>
#
# Examples:
#   bash eval_scripts/run_eval_medical_vl.sh saves/qwen3vl/moe_lora/baseline2_moelora_medical baseline2_moelora
#   bash eval_scripts/run_eval_medical_vl.sh ""                                                base   # base model only
#
# Saves predictions to eval_results/medical_vl/<run_name>_<dataset>.jsonl
# Prints metrics summary at end of each dataset.

set -e
cd "$(dirname "$0")/.."

ADAPTER="${1:-}"
RUN_NAME="${2:-eval_run}"
BASE_MODEL="${BASE_MODEL:-/data/android/yqy/work/lora_moe/model/Qwen3.5-35B-A3B}"
DATA_ROOT="${DATA_ROOT:-/data/android/yqy/work/lora_moe/data/medical_eval}"
OUT_ROOT="${OUT_ROOT:-eval_results/medical_vl}"
N_GPUS="${N_GPUS:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
BATCH_SIZE="${BATCH_SIZE:-4}"

mkdir -p "$OUT_ROOT"
PYBIN=/root/miniconda3/envs/qwen35/bin/python
TORCHRUN=/root/miniconda3/envs/qwen35/bin/torchrun

ADAPTER_ARG=""
[ -n "$ADAPTER" ] && ADAPTER_ARG="--adapter_path $ADAPTER"

for DS in vqa_rad slake pathvqa; do
    EVAL_JSON="$DATA_ROOT/$DS/test.json"
    SAVE_PATH="$OUT_ROOT/${RUN_NAME}_${DS}.jsonl"
    echo
    echo "================================================"
    echo "  EVAL: $DS  →  $SAVE_PATH"
    echo "================================================"
    $TORCHRUN --nproc_per_node=$N_GPUS eval_scripts/eval_medical_vl.py \
        --base_model "$BASE_MODEL" \
        $ADAPTER_ARG \
        --eval_json "$EVAL_JSON" \
        --save_path "$SAVE_PATH" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --batch_size "$BATCH_SIZE"
done

# 跨 3 个数据集打汇总表
echo
echo "================================================"
echo "  SUMMARY  (run_name = $RUN_NAME)"
echo "================================================"
$PYBIN <<EOF
import json, glob
from pathlib import Path
rows = []
for ds in ("vqa_rad", "slake", "pathvqa"):
    p = Path("$OUT_ROOT") / f"${RUN_NAME}_{ds}.metrics.json"
    if not p.exists():
        rows.append((ds, "MISSING", "-", "-", "-"))
        continue
    m = json.loads(p.read_text())
    rows.append((ds,
                 f"{m['closed_accuracy']*100:.2f}",
                 f"{m['open_recall']*100:.2f}",
                 f"{m['overall']*100:.2f}",
                 f"{m['n_total']}"))
print(f"{'dataset':<10} {'closed_acc':>11} {'open_recall':>12} {'overall':>9} {'n':>6}")
print("-" * 55)
for r in rows:
    print(f"{r[0]:<10} {r[1]:>11} {r[2]:>12} {r[3]:>9} {r[4]:>6}")
EOF

echo
echo "All done.  Predictions + metrics JSON in $OUT_ROOT/"

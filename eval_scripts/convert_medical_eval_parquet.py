"""Convert medical VQA parquet test sets (VQA-RAD / SLAKE-en / PathVQA) to a
unified eval-ready format.

Output per dataset:
  data/medical_eval/<ds>/test.json   — list of {id, image, question, answer, answer_type}
  data/medical_eval/<ds>/images/*.jpg

answer_type rule (跟 LLaVA-Med paper 对齐):
  - closed:  answer.lower().strip() in {"yes", "no"}
  - open:    otherwise

Run: python eval_scripts/convert_medical_eval_parquet.py
"""

import io
import json
import os
import glob
from pathlib import Path

import pandas as pd
from PIL import Image

BASE = "/data/android/yqy/work/lora_moe/data/medical_eval"

DATASETS = [
    ("vqa_rad", "vqa_rad/data/test-*.parquet"),
    ("slake",   "slake/data/test-*.parquet"),
    ("pathvqa", "pathvqa/data/test-*.parquet"),
]


def classify(ans: str) -> str:
    return "closed" if ans.lower().strip() in {"yes", "no"} else "open"


def convert(name: str, pattern: str):
    parquets = sorted(glob.glob(os.path.join(BASE, pattern)))
    df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)

    img_dir = Path(BASE) / name / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    type_count = {"closed": 0, "open": 0}
    for i, row in df.iterrows():
        img_dict = row["image"]
        img_bytes = img_dict["bytes"] if isinstance(img_dict, dict) else img_dict
        # 同一张物理图可能被多次复用(SLAKE: source.jpg per病例);为唯一性按 row index 命名
        img_path = img_dir / f"{i:06d}.jpg"
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img.save(img_path, "JPEG", quality=92)

        ans = str(row["answer"]).strip()
        a_type = classify(ans)
        type_count[a_type] += 1

        entries.append({
            "id": f"{name}_test_{i:06d}",
            "image": str(img_path),
            "question": str(row["question"]).strip(),
            "answer": ans,
            "answer_type": a_type,
        })

    out_json = Path(BASE) / name / "test.json"
    with open(out_json, "w") as f:
        json.dump(entries, f, ensure_ascii=False, indent=None)

    print(f"[{name}] {len(entries)} entries  (closed={type_count['closed']}, open={type_count['open']})  "
          f"→ {out_json}")


def main():
    for name, pattern in DATASETS:
        convert(name, pattern)


if __name__ == "__main__":
    main()

"""Retention eval on general VL benchmarks for Qwen3.5-VL medical-fine-tuned checkpoints.

Supports three benchmarks:
- MMBench DEV-EN (multi-choice, 4-option)
- MME (yes/no, 14 categories)
- RealWorldQA (open with inline options)

Forks load_model + generate_batch + left-pad fix from eval_medical_vl.py.

Usage:
    torchrun --nproc_per_node=8 eval_scripts/eval_vl_retention.py \\
        --base_model /data/.../Qwen3.5-35B-A3B \\
        --adapter_path saves/qwen3vl/moe_lora/v2_enhanced_global_medical_mixed \\
        --benchmark mmbench \\
        --bench_parquet /data/.../MMBench/en/dev-00000-of-00001.parquet \\
        --save_path eval_results/retention_mmbench_ours.jsonl \\
        --batch_size 4
"""

import argparse
import io
import json
import os
import re
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent / "src"))
from llamafactory.model.model_utils.moe_lora import load_moe_lora_state  # noqa: E402
from llamafactory.model.model_utils.mola import load_mola_state  # noqa: E402
from llamafactory.model.model_utils.das_lora import load_das_lora_state  # noqa: E402


# ============================================================
# Load model + adapter (fork from eval_medical_vl.py)
# ============================================================

def load_model(args):
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"  # critical for batched decoder-only generation

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        model = AutoModelForImageTextToText.from_pretrained(
            args.base_model, dtype=dtype, trust_remote_code=True,
        ).to(f"cuda:{local_rank}")
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            args.base_model, dtype=dtype, device_map="auto", trust_remote_code=True,
        )

    if args.adapter_path:
        ap = args.adapter_path
        if os.path.exists(os.path.join(ap, "das_lora_state.safetensors")):
            print(f"[load] DAS-LoRA  ← {ap}")
            model = load_das_lora_state(model, ap)
        elif os.path.exists(os.path.join(ap, "mola_state.safetensors")):
            print(f"[load] MoLA      ← {ap}")
            model = load_mola_state(model, ap)
        elif os.path.exists(os.path.join(ap, "moe_lora_state.safetensors")):
            print(f"[load] MoE-LoRA  ← {ap}")
            model = load_moe_lora_state(model, ap)
        elif os.path.exists(os.path.join(ap, "adapter_config.json")):
            print(f"[load] PEFT-LoRA ← {ap}")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, ap, dtype=dtype)
            model = model.merge_and_unload()
        else:
            raise FileNotFoundError(f"No recognized adapter file in {ap}")
    else:
        print("[load] BASE model only (no adapter)")
    model.eval()
    return processor, model


def build_prompt(question: str) -> str:
    """Qwen3.5-VL nothink prompt (matches training template + disables thinking)."""
    return (
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )


# ============================================================
# Per-benchmark prompt + parsing
# ============================================================

def make_mmbench_question(row) -> str:
    """MMBench: hint (optional) + question + 4 options."""
    parts = []
    if isinstance(row["hint"], str) and row["hint"] and row["hint"] != "nan":
        parts.append(f"Context: {row['hint']}")
    parts.append(f"Question: {row['question']}")
    for k in ["A", "B", "C", "D"]:
        v = row[k]
        if isinstance(v, str) and v and v != "nan":
            parts.append(f"{k}. {v}")
    parts.append("Answer with the option letter only.")
    return "\n".join(parts)


def make_mme_question(row) -> str:
    return f"{row['question']}\nAnswer with Yes or No only."


def make_realworldqa_question(row) -> str:
    return f"{row['question']}\nAnswer with the option letter only (e.g., A)."


def extract_choice_letter(text: str) -> str:
    """Extract a single A/B/C/D letter from model output."""
    m = re.search(r"\b([A-D])\b", text)
    return m.group(1) if m else ""


def extract_yesno(text: str) -> str:
    t = text.strip().lower()
    if re.search(r"\byes\b", t): return "Yes"
    if re.search(r"\bno\b", t): return "No"
    return ""


BENCHMARKS = {
    "mmbench": {
        "make_question": make_mmbench_question,
        "extract": extract_choice_letter,
        "max_new_tokens": 16,
    },
    "mme": {
        "make_question": make_mme_question,
        "extract": extract_yesno,
        "max_new_tokens": 8,
    },
    "realworldqa": {
        "make_question": make_realworldqa_question,
        "extract": extract_choice_letter,
        "max_new_tokens": 16,
    },
}


# ============================================================
# Generation
# ============================================================

@torch.inference_mode()
def generate_batch(processor, model, images, questions, max_new_tokens):
    texts = [build_prompt(q) for q in questions]
    inputs = processor(
        text=texts, images=images, return_tensors="pt", padding=True
    ).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
    )
    in_len = inputs.input_ids.shape[1]
    return [
        processor.tokenizer.decode(ids[in_len:], skip_special_tokens=True).strip()
        for ids in out
    ]


def img_from_bytes(item, max_side: int = 1024) -> Image.Image:
    """parquet image col is dict {'bytes': b'...'}.
    Resize so longest side <= max_side to avoid VL OOM on huge images."""
    if isinstance(item, dict) and "bytes" in item:
        img = Image.open(io.BytesIO(item["bytes"])).convert("RGB")
        w, h = img.size
        m = max(w, h)
        if m > max_side:
            scale = max_side / m
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return img
    raise ValueError(f"Unexpected image format: {type(item)}")


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapter_path", default=None)
    p.add_argument("--benchmark", required=True, choices=list(BENCHMARKS.keys()))
    p.add_argument("--bench_parquet", required=True, help="path to parquet file (or glob); can supply multiple via comma")
    p.add_argument("--save_path", required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_dist = world_size > 1
    is_main = local_rank == 0
    if is_dist:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    # Load benchmark data
    parquet_files = args.bench_parquet.split(",")
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    if args.limit:
        df = df.iloc[:args.limit].reset_index(drop=True)
    if is_main:
        print(f"[bench] {args.benchmark}: {len(df)} samples loaded")

    # Shard
    if is_dist:
        df = df.iloc[local_rank::world_size].reset_index(drop=True)

    bench = BENCHMARKS[args.benchmark]

    # Load model
    processor, model = load_model(args)

    # Generate
    results = []
    bs = args.batch_size
    n_batches = (len(df) + bs - 1) // bs
    iterator = range(0, len(df), bs)
    if is_main:
        iterator = tqdm(iterator, total=n_batches, desc=f"rank{local_rank}")

    for i in iterator:
        batch = df.iloc[i:i+bs]
        try:
            images = [img_from_bytes(item) for item in batch["image"].tolist()]
            questions = [bench["make_question"](row) for _, row in batch.iterrows()]
            preds_raw = generate_batch(
                processor, model, images, questions, bench["max_new_tokens"],
            )
        except Exception as e:
            if is_main:
                print(f"[ERR] batch {i}: {e}", flush=True)
            preds_raw = [""] * len(batch)

        for (_, row), pred_raw in zip(batch.iterrows(), preds_raw):
            extracted = bench["extract"](pred_raw)
            gold = str(row["answer"]).strip()
            correct = (extracted == gold) if extracted else False
            results.append({
                "idx": str(row.get("index", row.get("question_id", i))),
                "question": str(row["question"])[:200],
                "gold": gold,
                "pred_raw": pred_raw,
                "pred_extracted": extracted,
                "correct": bool(correct),
                "category": str(row.get("category", "all")),
            })

    # Gather results across ranks
    if is_dist:
        all_results = [None] * world_size
        dist.all_gather_object(all_results, results)
        if is_main:
            results = [r for sublist in all_results for r in sublist]
        dist.barrier()

    # Save + report
    if is_main:
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        n = len(results)
        n_correct = sum(r["correct"] for r in results)
        acc = n_correct / n if n else 0.0
        print(f"\n[{args.benchmark}] overall accuracy: {acc:.4f} ({n_correct}/{n})")

        # Per-category breakdown (MME has categories)
        from collections import defaultdict
        cat_stats = defaultdict(lambda: [0, 0])  # [n_correct, n_total]
        for r in results:
            cat_stats[r["category"]][0] += int(r["correct"])
            cat_stats[r["category"]][1] += 1
        print(f"\n[{args.benchmark}] per-category:")
        for cat, (c, t) in sorted(cat_stats.items()):
            print(f"  {cat:25s} {c/t:.4f} ({c}/{t})")

    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

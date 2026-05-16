"""Eval script for medical multimodal VQA on Qwen3.5-VL.

Supports VQA-RAD / SLAKE-en / PathVQA test sets (converted by
convert_medical_eval_parquet.py to {test.json + images/}).

Loads a Qwen3.5-VL base + optional adapter (MoE-LoRA / DAS-LoRA / MoLA / PEFT),
runs greedy generation per sample, computes LLaVA-Med-style metrics:
  - closed: exact-match accuracy on yes/no
  - open:   token-level recall  (∩(pred_toks, ref_toks)) / |ref_toks|

Usage:
  python eval_scripts/eval_medical_vl.py \\
      --base_model /data/.../Qwen3.5-35B-A3B \\
      --adapter_path saves/qwen3vl/moe_lora/baseline2_moelora_medical \\
      --eval_json data/medical_eval/vqa_rad/test.json \\
      --save_path eval_results/preds_vqa_rad_baseline2.jsonl \\
      --max_new_tokens 64

Multi-GPU (torchrun): each rank takes a slice of the eval set.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

# Make repo's src/ importable for adapter loading helpers
_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent / "src"))

from llamafactory.model.model_utils.moe_lora import load_moe_lora_state  # noqa: E402
from llamafactory.model.model_utils.mola import load_mola_state  # noqa: E402
from llamafactory.model.model_utils.das_lora import load_das_lora_state  # noqa: E402


# ============================================================
# 1. Load model + adapter (multimodal version)
# ============================================================

def load_model(args) -> tuple:
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)

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


# ============================================================
# 2. Inference (per-sample, batch_size=1 for multimodal simplicity)
# ============================================================

def build_prompt_qwen3_5_nothink(question: str) -> str:
    """手动构造跟训练 (template: qwen3_5_nothink) 100% 一致的 prompt:
        <|im_start|>user
        <|vision_start|><|image_pad|><|vision_end|>{question}<|im_end|>
        <|im_start|>assistant

    绕开 processor.apply_chat_template — 该 chat_template 推理时会强加 <think>
    标记(即使 enable_thinking=False 也只是塞 <think></think> 占位),
    跟训练分布不符,导致 base/adapter model 胡言乱语 / 空 pred / 泄漏 'user\\n'。
    单个 <|image_pad|> 会被 processor 在 __call__ 时按 image patch 数自动展开。
    """
    return (
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


@torch.inference_mode()
def generate_batch(processor, model, image_paths, questions, max_new_tokens):
    """Batched multimodal generation. Returns list[str] of length len(image_paths)."""
    images = [Image.open(p).convert("RGB") for p in image_paths]
    texts = [build_prompt_qwen3_5_nothink(q) for q in questions]
    # processor 自动左 pad input_ids 并把多图的 patches 拼好,用 image_grid_thw 关联
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


@torch.inference_mode()
def generate_one(processor, model, image_path: str, question: str, max_new_tokens: int) -> str:
    return generate_batch(processor, model, [image_path], [question], max_new_tokens)[0]


# ============================================================
# 3. Metrics (LLaVA-Med protocol)
# ============================================================

_TOK_RE = re.compile(r"\w+")


def normalize(s: str) -> str:
    return _TOK_RE.findall(s.lower())


def open_recall(pred: str, ref: str) -> float:
    """Token-level recall: how many ref tokens appear in pred."""
    ref_toks = set(normalize(ref))
    if not ref_toks:
        return 0.0
    pred_toks = set(normalize(pred))
    return len(ref_toks & pred_toks) / len(ref_toks)


def closed_match(pred: str, ref: str) -> int:
    """1 if ref ('yes'/'no') appears as a token in pred, else 0.
    Uses substring-on-tokens to tolerate "Yes, ..." / "The answer is no."."""
    pred_toks = normalize(pred)
    return int(ref.lower().strip() in pred_toks)


def compute_metrics(records: List[Dict]) -> Dict:
    closed = [r for r in records if r["answer_type"] == "closed"]
    openq = [r for r in records if r["answer_type"] == "open"]

    closed_acc = sum(closed_match(r["pred"], r["answer"]) for r in closed) / max(len(closed), 1)
    open_rec = sum(open_recall(r["pred"], r["answer"]) for r in openq) / max(len(openq), 1)
    overall = (closed_acc * len(closed) + open_rec * len(openq)) / max(len(records), 1)

    return {
        "n_total":         len(records),
        "n_closed":        len(closed),
        "n_open":          len(openq),
        "closed_accuracy": round(closed_acc, 4),
        "open_recall":     round(open_rec, 4),
        "overall":         round(overall, 4),
    }


# ============================================================
# 4. Main loop
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapter_path", default=None)
    p.add_argument("--eval_json", required=True, help="data/medical_eval/<ds>/test.json")
    p.add_argument("--save_path", default=None, help="Save per-sample preds to JSONL")
    p.add_argument("--max_new_tokens", type=int, default=32, help="VQA 答案多数 <30 token,32 够用")
    p.add_argument("--batch_size", type=int, default=4, help="多模态 batch generation,80GB H100 上 4-8 都稳")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = p.parse_args()

    # ===== 关键: 多卡时第一步必须先绑定每个 rank 的物理 GPU,再初始化 NCCL =====
    # 否则所有 rank 默认 cuda:0,NCCL all_gather 时报 "Duplicate GPU detected"
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

    with open(args.eval_json) as f:
        data = json.load(f)
    if args.limit:
        data = data[: args.limit]

    data_shard = data[rank::world_size]
    print(f"[rank {rank}/{world_size}] processing {len(data_shard)}/{len(data)} samples")

    processor, model = load_model(args)

    records = []
    bs = args.batch_size
    pbar = tqdm(total=len(data_shard), disable=(rank != 0))
    for i in range(0, len(data_shard), bs):
        chunk = data_shard[i : i + bs]
        try:
            preds = generate_batch(
                processor, model,
                [c["image"] for c in chunk],
                [c["question"] for c in chunk],
                args.max_new_tokens,
            )
        except Exception as e:
            preds = [f"<ERROR: {type(e).__name__}: {str(e)[:200]}>" for _ in chunk]
        for ent, pred in zip(chunk, preds):
            records.append({
                "id":          ent["id"],
                "question":    ent["question"],
                "answer":      ent["answer"],
                "answer_type": ent["answer_type"],
                "pred":        pred,
            })
        pbar.update(len(chunk))
    pbar.close()

    # Multi-GPU: gather all shards on rank 0 (init done at top of main)
    if world_size > 1:
        import torch.distributed as dist
        gathered = [None] * world_size
        dist.all_gather_object(gathered, records)
        if rank == 0:
            records = [r for shard in gathered for r in shard]
            order = {ent["id"]: i for i, ent in enumerate(data)}
            records.sort(key=lambda r: order.get(r["id"], 1e9))

    if rank == 0:
        if args.save_path:
            Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(args.save_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[save] preds → {args.save_path}")

        metrics = compute_metrics(records)
        # 数据集名取自 eval_json 路径,方便汇总
        metrics["dataset"] = Path(args.eval_json).parent.name
        metrics["adapter"] = args.adapter_path or "base"

        print("\n=== Metrics ===")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

        # 同时写 .metrics.json,供 runner 汇总用
        if args.save_path:
            metrics_path = str(Path(args.save_path).with_suffix(".metrics.json"))
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print(f"[save] metrics → {metrics_path}")


if __name__ == "__main__":
    main()

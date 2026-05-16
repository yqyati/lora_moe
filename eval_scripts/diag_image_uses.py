"""Sanity test: does the trained checkpoint actually USE the image content,
or just ignore it and spit out a generic medical-sounding response?

Strategy: feed the SAME question with TWO clearly different images
(e.g. chest X-ray vs histology slide), then compare outputs.

  - If outputs are identical (or nearly so) → image is being ignored ❌
  - If outputs are clearly different → image is actually used ✅

Run:
  cd LlamaFactory && python eval_scripts/diag_image_uses.py \\
      --base_model /data/.../Qwen3.5-35B-A3B \\
      --adapter_path saves/qwen3vl/moe_lora/baseline2_moelora_medical
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from llamafactory.model.model_utils.moe_lora import load_moe_lora_state
from llamafactory.model.model_utils.das_lora import load_das_lora_state
from llamafactory.model.model_utils.mola import load_mola_state


# 从训练 JSON + 原始 LLaVA-Med JSON 关联出 domain → image 映射,确保文件存在
TRAIN_JSON = "/data/android/yqy/work/lora_moe/data/medical_train/llava_med/llava_med_train_filtered.json"
SRC_JSON = "/data/android/yqy/work/lora_moe/data/medical_train/llava_med/json/llava_med_instruct_60k_inline_mention.json"


def pick_images_per_domain():
    with open(TRAIN_JSON) as f:
        kept = json.load(f)
    with open(SRC_JSON) as f:
        src = json.load(f)
    fn_to_domain = {}
    for e in src:
        doms = [k for k, v in e["domain"].items() if v]
        if doms:
            fn_to_domain[e["image"]] = doms[0]

    by_domain = {}
    for e in kept:
        img_path = e["images"][0]
        d = fn_to_domain.get(Path(img_path).name, "unknown")
        by_domain.setdefault(d, []).append(img_path)
    print("[domains]", ", ".join(f"{k}:{len(v)}" for k, v in by_domain.items()))
    return by_domain


@torch.inference_mode()
def gen(processor, model, image_path, question, max_new=120):
    img = Image.open(image_path).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    try:
        text = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=max_new, do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
    )
    return processor.tokenizer.decode(
        out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapter_path", default=None)
    args = p.parse_args()

    by_domain = pick_images_per_domain()
    pairs = []
    if "chest_xray" in by_domain and "histology" in by_domain:
        pairs.append({
            "question": "Describe what you see in this medical image in one sentence.",
            "images": [by_domain["chest_xray"][0], by_domain["histology"][0]],
            "labels": ["chest_xray", "histology"],
        })
    if "mri" in by_domain and "ct_scan" in by_domain:
        pairs.append({
            "question": "What anatomical structure or region is shown?",
            "images": [by_domain["mri"][0], by_domain["ct_scan"][0]],
            "labels": ["mri", "ct_scan"],
        })
    if not pairs:
        keys = [k for k, v in by_domain.items() if v and k != "unknown"]
        if len(keys) >= 2:
            pairs.append({
                "question": "Describe what you see in this medical image.",
                "images": [by_domain[keys[0]][0], by_domain[keys[1]][0]],
                "labels": [keys[0], keys[1]],
            })

    print(f"\n[load] processor + base model from {args.base_model}")
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    # 单 GPU 加载,避免 device_map=auto 把 base model 跨卡分,
    # 和我们注入的 lora_pool/router 跨设备失败
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model, dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda:0")

    if args.adapter_path:
        ap = args.adapter_path
        if (Path(ap) / "das_lora_state.safetensors").exists():
            print(f"[load] DAS-LoRA  ← {ap}")
            model = load_das_lora_state(model, ap)
        elif (Path(ap) / "mola_state.safetensors").exists():
            print(f"[load] MoLA      ← {ap}")
            model = load_mola_state(model, ap)
        elif (Path(ap) / "moe_lora_state.safetensors").exists():
            print(f"[load] MoE-LoRA  ← {ap}")
            model = load_moe_lora_state(model, ap)
        else:
            raise FileNotFoundError(f"No adapter found in {ap}")
    else:
        print("[load] BASE model (no adapter)")
    model.eval()

    print("\n" + "=" * 70)
    print("SANITY: same question, different images → should give different outputs")
    print("=" * 70)

    import re
    for i, pair in enumerate(pairs, 1):
        q = pair["question"]
        print(f"\n--- Pair {i}: question = {q!r} ---")
        outs = []
        for img_path, label in zip(pair["images"], pair["labels"]):
            response = gen(processor, model, img_path, q)
            outs.append(response)
            print(f"\n  [{label}]  {Path(img_path).name}")
            print(f"  → {response[:300]}{'...' if len(response) > 300 else ''}")

        print()
        if outs[0].strip() == outs[1].strip():
            print(f"  ❌  IDENTICAL outputs — image likely IGNORED")
        else:
            t0 = set(re.findall(r"\w+", outs[0].lower()))
            t1 = set(re.findall(r"\w+", outs[1].lower()))
            overlap = len(t0 & t1) / max(len(t0 | t1), 1)
            print(f"  ✅  DIFFERENT outputs (token Jaccard overlap = {overlap:.2%})")


if __name__ == "__main__":
    main()

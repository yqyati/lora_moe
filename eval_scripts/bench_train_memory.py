#!/usr/bin/env python
"""Peak GPU training memory benchmark for PEFT methods on DeepSeek-V2-Lite.

For each method: load model + adapter, enable training mode, run 2 forward
+ backward + Adam step on a dummy batch, record peak GPU memory.

Single GPU (no ZeRO), batch size 1, seq len 2048 (matches training config).
"""
import os, sys, time, json, types
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from _common import load


def bench_train_mem(label, adapter, base, seq_len=2048, n_steps=2):
    print(f"\n=== {label} ===")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    args = types.SimpleNamespace(base_model=base, adapter_path=adapter, dtype="bfloat16")
    tokenizer, model = load(args)
    model.train()  # enable train mode (activations saved for backward)

    # Identify trainable params (adapter-injected ones)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable) / 1e6
    print(f"  trainable params: {n_params:.2f}M")

    if n_params == 0:
        print("  no trainable params (base model only), skipping training step")
        peak_load_gb = torch.cuda.max_memory_allocated() / 1e9
        del model, tokenizer; torch.cuda.empty_cache()
        return {"label": label, "trainable_M": 0.0,
                "peak_load_gb": peak_load_gb, "peak_train_gb": None}

    peak_load_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"  peak after load:  {peak_load_gb:.2f} GB")

    # AdamW optimizer (matches our training config)
    optim = torch.optim.AdamW(trainable, lr=4e-4, betas=(0.9, 0.999))

    vocab = tokenizer.vocab_size
    input_ids = torch.randint(low=0, high=vocab, size=(1, seq_len)).to(model.device)
    labels = input_ids.clone()

    for step in range(n_steps):
        optim.zero_grad()
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        optim.step()
        torch.cuda.synchronize()
        print(f"  step {step+1}: loss={out.loss.item():.3f}  "
              f"peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB")

    peak_train_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"  peak after train: {peak_train_gb:.2f} GB")

    res = {
        "label": label,
        "trainable_M": n_params,
        "peak_load_gb": peak_load_gb,
        "peak_train_gb": peak_train_gb,
    }
    del model, tokenizer, optim
    torch.cuda.empty_cache()
    return res


def main():
    base = "/data/android/yqy/work/lora_moe/model/DeepSeek-V2-Lite"
    root = "/data/android/yqy/work/LlamaFactory/saves/deepseek_v2_lite/moe_lora"

    runs = [
        ("Base (no adapter)",        None),
        ("ours (N=128 r=34 matched)",f"{root}/rcp_global_N128_r34_matched"),
        ("MoELoRA",                  f"{root}/baseline2_moelora_math"),
        ("MoLA inc matched",         f"{root}/mola_inc_match_math"),
        ("DAS-LoRA r=52 matched",    f"{root}/das_lora_math"),
    ]
    results = []
    for label, adapter in runs:
        try:
            results.append(bench_train_mem(label, adapter, base))
        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM: {e}")
            results.append({"label": label, "OOM": True})
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}")
            results.append({"label": label, "error": str(e)[:200]})
            torch.cuda.empty_cache()

    print("\n" + "=" * 75)
    print(f"{'method':28s} {'trainable':>10s} {'load(GB)':>10s} {'train(GB)':>11s}")
    print("-" * 75)
    for r in results:
        if "OOM" in r or "error" in r:
            print(f"{r['label']:28s} ---  {r.get('OOM', r.get('error',''))}")
            continue
        tp = f"{r['trainable_M']:.1f}M"
        ld = f"{r['peak_load_gb']:.2f}"
        tr = f"{r['peak_train_gb']:.2f}" if r["peak_train_gb"] else "  —"
        print(f"{r['label']:28s} {tp:>10s} {ld:>10s} {tr:>11s}")

    out = "/data/android/yqy/work/lora_moe/data/training_memory_deepseek.json"
    json.dump(results, open(out, "w"), indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

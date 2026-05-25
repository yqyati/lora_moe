#!/usr/bin/env python
"""OLMoE-1B-7B single-GPU inference latency benchmark."""
import os, sys, time, json, types
import torch

sys.path.insert(0, os.path.dirname(__file__))
from _common import load


PROMPT = (
    "Solve the following math problem step by step. "
    "End your answer with '#### <number>'.\n\n"
    "Question: A coffee shop sells 240 cups of coffee on Monday. "
    "On Tuesday they sell 1/3 more than Monday. How many cups did they sell on Tuesday?\n"
    "Answer:"
)


def bench(adapter, label, base, max_new_tokens=256, warmup=2, n_runs=5):
    print(f"\n=== {label} ===")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    args = types.SimpleNamespace(base_model=base, adapter_path=adapter, dtype="bfloat16")
    tokenizer, model = load(args)
    model.eval()

    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    in_len = inputs.input_ids.shape[1]

    for _ in range(warmup):
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize()

    ttfts, tps_list = [], []
    for _ in range(n_runs):
        torch.cuda.synchronize(); t0 = time.time()
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
        torch.cuda.synchronize()
        ttfts.append(time.time() - t0)

        t1 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        elapsed = time.time() - t1
        tps_list.append((out.shape[1] - in_len) / elapsed)

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    res = {
        "label": label,
        "tps_mean": sum(tps_list) / len(tps_list),
        "ttft_mean": sum(ttfts) / len(ttfts),
        "peak_mem_gb": peak_gb,
    }
    print(f"  tokens/sec : {res['tps_mean']:.2f}")
    print(f"  TTFT (s)   : {res['ttft_mean']:.3f}")
    print(f"  peak mem GB: {res['peak_mem_gb']:.2f}")
    del model, tokenizer
    torch.cuda.empty_cache()
    return res


def main():
    base = "allenai/OLMoE-1B-7B-0924"
    root = "/data/android/yqy/work/LlamaFactory/saves/olmoe/moe_lora"

    runs = [
        ("Base (no adapter)", None),
        ("ours (N=128 r=16 K=4)",   f"{root}/v2_enhanced_h16_128"),
        ("ours nobal (N=128 r=16)", f"{root}/v2_enhanced_h16_nobal"),
        ("MoELoRA",                 f"{root}/baseline2_independent"),
        ("MoLA",                    f"{root}/mola_math"),
        ("DAS-LoRA",                f"{root}/das_lora_math"),
    ]
    results = []
    for label, adapter in runs:
        try:
            results.append(bench(adapter, label, base))
        except Exception as e:
            print(f"FAILED {label}: {e}")

    print("\n" + "=" * 72)
    print(f"{'method':28s} {'tps':>10s} {'TTFT(s)':>10s} {'peakGB':>10s} {'tps_rel':>10s}")
    print("-" * 72)
    base_tps = results[0]["tps_mean"] if results else 1.0
    for r in results:
        rel = r["tps_mean"] / base_tps * 100
        print(f"{r['label']:28s} {r['tps_mean']:10.2f} {r['ttft_mean']:10.3f} "
              f"{r['peak_mem_gb']:10.2f} {rel:9.1f}%")

    out = "/data/android/yqy/work/lora_moe/data/inference_latency_olmoe.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

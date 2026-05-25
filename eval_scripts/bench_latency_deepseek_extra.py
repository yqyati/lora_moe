#!/usr/bin/env python
"""Bench inference latency for missing N_L values on DeepSeek-V2-Lite."""
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
        tps_list.append((out.shape[1] - in_len) / (time.time() - t1))

    res = {
        "label": label,
        "tps_mean": sum(tps_list) / len(tps_list),
        "ttft_mean": sum(ttfts) / len(ttfts),
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    print(f"  tps={res['tps_mean']:.2f}  TTFT={res['ttft_mean']:.3f}s  peakGB={res['peak_mem_gb']:.2f}")
    del model, tokenizer
    torch.cuda.empty_cache()
    return res


def main():
    base = "/data/android/yqy/work/lora_moe/model/DeepSeek-V2-Lite"
    root = "/data/android/yqy/work/LlamaFactory/saves/deepseek_v2_lite/moe_lora"
    runs = [
        ("N=64  r=70",  f"{root}/rcp_global_N64_r70_matched"),
        ("N=256 r=17",  f"{root}/rcp_global_N256_r17_matched"),
    ]
    results = []
    for label, adapter in runs:
        try:
            results.append(bench(adapter, label, base))
        except Exception as e:
            print(f"FAILED {label}: {e}")

    print("\n" + "=" * 50)
    for r in results:
        print(f"{r['label']:15s} tps={r['tps_mean']:6.2f}  TTFT={r['ttft_mean']:.3f}  peak={r['peak_mem_gb']:.2f}GB")

    out = "/data/android/yqy/work/lora_moe/data/inference_latency_deepseek_extra.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

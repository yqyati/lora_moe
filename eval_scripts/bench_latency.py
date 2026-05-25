#!/usr/bin/env python
"""Single-GPU inference latency benchmark on DeepSeek-V2-Lite.

For each (method, adapter) pair, measure:
- throughput (tokens / sec)
- time-to-first-token (TTFT, seconds)
- peak GPU memory (GB)

Same prompt, fixed max_new_tokens, greedy decoding, batch=1 (single-sequence
latency, the worst case for any router overhead).
"""
import argparse, os, sys, time, json
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from _common import load, common_arg_parser  # noqa


PROMPT = (
    "Solve the following math problem step by step. "
    "End your answer with '#### <number>'.\n\n"
    "Question: A coffee shop sells 240 cups of coffee on Monday. "
    "On Tuesday they sell 1/3 more than Monday. How many cups did they sell on Tuesday?\n"
    "Answer:"
)


def bench(adapter, label, base, max_new_tokens=256, warmup=2, n_runs=5):
    import types
    args = types.SimpleNamespace(
        base_model=base, adapter_path=adapter, dtype="bfloat16"
    )
    print(f"\n=== {label} ===")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    tokenizer, model = load(args)
    model.eval()

    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    in_len = inputs.input_ids.shape[1]

    # warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=8, do_sample=False)

    torch.cuda.synchronize()

    # measure
    ttfts, tps_list = [], []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            # TTFT = first forward
            out_first = model.generate(**inputs, max_new_tokens=1, do_sample=False)
        torch.cuda.synchronize()
        ttft = time.time() - t0

        t1 = time.time()
        with torch.no_grad():
            out_full = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        elapsed = time.time() - t1
        gen_len = out_full.shape[1] - in_len
        tps = gen_len / elapsed

        ttfts.append(ttft)
        tps_list.append(tps)

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
    base = "/data/android/yqy/work/lora_moe/model/DeepSeek-V2-Lite"
    root = "/data/android/yqy/work/LlamaFactory/saves/deepseek_v2_lite/moe_lora"

    runs = [
        ("Base (no adapter)", None),
        ("ours (N=512 r=8)",  f"{root}/rcp_global_best"),
        ("ours (N=128 r=34)", f"{root}/rcp_global_N128_r34_matched"),
        ("MoELoRA",           f"{root}/baseline2_moelora_math"),
        ("MoLA (rank=16)",    f"{root}/mola_math"),
        ("DAS-LoRA (r=52)",   f"{root}/das_lora_math"),
    ]
    results = []
    for label, adapter in runs:
        try:
            r = bench(adapter, label, base)
            results.append(r)
        except Exception as e:
            print(f"FAILED {label}: {e}")

    # final table
    print("\n" + "=" * 72)
    print(f"{'method':25s} {'tps':>10s} {'TTFT(s)':>10s} {'peakGB':>10s} {'tps_rel':>10s}")
    print("-" * 72)
    base_tps = results[0]["tps_mean"] if results else 1.0
    for r in results:
        rel = r["tps_mean"] / base_tps * 100
        print(f"{r['label']:25s} {r['tps_mean']:10.2f} {r['ttft_mean']:10.3f} "
              f"{r['peak_mem_gb']:10.2f} {rel:9.1f}%")

    out = "/data/android/yqy/work/lora_moe/data/inference_latency_deepseek.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

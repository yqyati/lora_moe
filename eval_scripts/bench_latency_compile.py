#!/usr/bin/env python
"""Compare inference latency with and without torch.compile.

Tests the same DeepSeek configs as bench_latency.py, but applies
torch.compile(mode='reduce-overhead') to each model and re-measures.
Reports both raw and compiled tokens/sec to quantify the speedup.
"""
import os, sys, time, json, types, warnings
import torch

sys.path.insert(0, os.path.dirname(__file__))
from _common import load

# torch.compile can throw a lot of recompile warnings during warmup
warnings.filterwarnings("ignore", category=UserWarning)


PROMPT = (
    "Solve the following math problem step by step. "
    "End your answer with '#### <number>'.\n\n"
    "Question: A coffee shop sells 240 cups of coffee on Monday. "
    "On Tuesday they sell 1/3 more than Monday. How many cups did they sell on Tuesday?\n"
    "Answer:"
)


def measure(model, tokenizer, max_new_tokens=256, warmup=2, n_runs=5):
    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    in_len = inputs.input_ids.shape[1]
    for _ in range(warmup):
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize()
    tps_list, ttfts = [], []
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
    return sum(tps_list) / len(tps_list), sum(ttfts) / len(ttfts)


def bench_one(label, adapter, base):
    print(f"\n=== {label} ===")
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    args = types.SimpleNamespace(base_model=base, adapter_path=adapter, dtype="bfloat16")
    tokenizer, model = load(args)
    model.eval()

    # uncompiled
    tps_raw, ttft_raw = measure(model, tokenizer)
    print(f"  raw:       tps={tps_raw:.2f}  TTFT={ttft_raw:.3f}s")

    # try torch.compile
    try:
        compiled = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        # extra warmup for compile (first call traces & compiles)
        inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
        print("  compiling (first calls will be slow)...")
        with torch.no_grad():
            _ = compiled.generate(**inputs, max_new_tokens=8, do_sample=False)
        with torch.no_grad():
            _ = compiled.generate(**inputs, max_new_tokens=8, do_sample=False)
        torch.cuda.synchronize()
        tps_c, ttft_c = measure(compiled, tokenizer, warmup=2, n_runs=5)
        print(f"  compiled:  tps={tps_c:.2f}  TTFT={ttft_c:.3f}s  speedup={tps_c/tps_raw:.2f}x")
        compiled_ok = True
    except Exception as e:
        print(f"  compile FAILED: {type(e).__name__}: {str(e)[:200]}")
        tps_c, ttft_c, compiled_ok = None, None, False

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    res = {
        "label": label,
        "tps_raw": tps_raw, "ttft_raw": ttft_raw,
        "tps_compiled": tps_c, "ttft_compiled": ttft_c,
        "speedup": (tps_c / tps_raw if compiled_ok else None),
        "peak_mem_gb": peak_gb,
    }
    del model, tokenizer
    torch.cuda.empty_cache()
    return res


def main():
    base = "/data/android/yqy/work/lora_moe/model/DeepSeek-V2-Lite"
    root = "/data/android/yqy/work/LlamaFactory/saves/deepseek_v2_lite/moe_lora"

    runs = [
        ("Base (no adapter)",           None),
        ("ours (N=128 r=34)",           f"{root}/rcp_global_N128_r34_matched"),
        ("MoLA (rank=16)",              f"{root}/mola_math"),
        ("DAS-LoRA (rank=52)",          f"{root}/das_lora_math"),
    ]
    results = []
    for label, adapter in runs:
        try:
            results.append(bench_one(label, adapter, base))
        except Exception as e:
            print(f"FAILED {label}: {type(e).__name__}: {e}")

    print("\n" + "=" * 85)
    print(f"{'method':25s} {'raw tps':>8s} {'compiled':>8s} {'speedup':>8s} {'peakGB':>8s}")
    print("-" * 85)
    for r in results:
        tc = f"{r['tps_compiled']:.2f}" if r['tps_compiled'] is not None else " FAIL"
        sp = f"{r['speedup']:.2f}x" if r['speedup'] else "  —"
        print(f"{r['label']:25s} {r['tps_raw']:8.2f} {tc:>8s} {sp:>8s} {r['peak_mem_gb']:8.2f}")

    out = "/data/android/yqy/work/lora_moe/data/inference_latency_compile.json"
    json.dump(results, open(out, "w"), indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

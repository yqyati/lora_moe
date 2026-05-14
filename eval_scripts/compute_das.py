"""Compute Domain Advantage Score (DAS) for MoE experts.

Simplified from "Exploring Expert Concentration" (ICLR 2026 submission):
- Skips Stage 1 (attention + router fine-tuning)
- Computes DAS on base model's pre-trained routing
- DAS(D_d, D_g) = mean(routing_score_d) - mean(routing_score_g)
- Selects top-k experts per layer by DAS

Usage:
    python eval_scripts/compute_das.py \\
        --base_model allenai/OLMoE-1B-7B-0924 \\
        --domain_dataset oumi-ai/MetaMathQA-R1 \\
        --general_dataset cais/mmlu \\
        --top_k 4 \\
        --max_samples 200 \\
        --output eval_results/das_math.json
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def collect_routing_scores(model, tokenizer, samples, max_seq_len=512, desc=""):
    """Run forward on samples; for each MoE block, accumulate per-expert
    mean routing score. Returns tensor [n_layers, n_experts]."""
    moe_layers = []
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
            moe_layers.append((i, layer.mlp))

    if not moe_layers:
        raise RuntimeError("No MoE layers with .gate found.")

    # n_experts from gate weight shape
    n_experts = moe_layers[0][1].gate.weight.shape[0]
    n_layers = len(moe_layers)
    device = next(model.parameters()).device

    score_sum = torch.zeros(n_layers, n_experts, device=device, dtype=torch.float32)
    token_count = torch.zeros(n_layers, device=device, dtype=torch.float32)

    # Forward pre-hook: 直接挂在 MoE block(`mlp`)上,从 hidden_states 算 routing scores。
    # 这种方式兼容多种 MoE 实现:
    #   - OLMoE / Qwen3-MoE: 它们的 MoE block forward 里也会先调 gate.linear
    #   - DeepSeek-V2 (transformers 5.x native): forward 直接用 F.linear(h, gate.weight)
    #     不走 gate.forward(),所以不能挂在 mlp.gate 上(hook 永远不触发)
    # 改挂在 mlp 上,统一处理。
    handles = []
    for li, (orig_idx, mlp) in enumerate(moe_layers):
        def make_hook(layer_idx, gate_module):
            def hook(module, input):
                h = input[0]
                if h.dim() == 3:
                    h = h.reshape(-1, h.shape[-1])
                # 用 gate 的 weight 自己算 logits(跟 native MoE block 的内部计算一致)
                bias = getattr(gate_module, "bias", None)
                logits = F.linear(h.float(), gate_module.weight.float(), bias)
                scores = F.softmax(logits, dim=-1)
                # device_map="auto" 下,该层可能在 cuda:N 而 score_sum 在 cuda:0,需 to(device)
                score_sum[layer_idx] += scores.sum(dim=0).to(score_sum.device)
                token_count[layer_idx] += scores.shape[0]
            return hook
        h = mlp.register_forward_pre_hook(make_hook(li, mlp.gate))
        handles.append(h)

    try:
        for text in tqdm(samples, desc=desc):
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_seq_len
            ).to(device)
            _ = model(**inputs)
    finally:
        for h in handles:
            h.remove()

    # Normalize: mean score per expert across all tokens
    mean_scores = score_sum / token_count.unsqueeze(-1).clamp(min=1)
    return mean_scores.cpu()


def extract_text(sample, dataset_name):
    """Extract a representative text field from various dataset formats."""
    if "messages" in sample:
        # ShareGPT / chat format
        msgs = sample["messages"]
        if isinstance(msgs, list) and len(msgs) > 0:
            return " ".join(str(m.get("content", "")) for m in msgs if isinstance(m, dict))
    # MMLU-style (question + choices,先检查 choices,因为 answer 可能是 int label)
    if "question" in sample and "choices" in sample:
        choices = sample["choices"]
        q = str(sample["question"])
        if isinstance(choices, list):
            return q + " " + " ".join(str(c) for c in choices)
        return q
    # Generic question + answer(string answer 才行)
    if "question" in sample and "answer" in sample:
        ans = sample["answer"]
        if isinstance(ans, str):
            return str(sample["question"]) + " " + ans
        return str(sample["question"])
    if "prompt" in sample and "response" in sample:
        return str(sample["prompt"]) + " " + str(sample["response"])
    if "instruction" in sample:
        return str(sample["instruction"]) + " " + str(sample.get("output", ""))
    if "text" in sample:
        return str(sample["text"])
    if "problem" in sample:
        return str(sample["problem"])
    # Fallback: stringify whole sample
    return str(sample)[:1000]


def load_samples(dataset_name, split, n, dataset_config=None):
    """Load N text samples from a dataset.

    Supports:
    - Local JSON file path (.json / .jsonl) — sharegpt 格式 {"messages": [...]} 或 alpaca
    - HuggingFace dataset name (e.g. "cais/mmlu")
    """
    # Local file path
    if os.path.exists(dataset_name) and dataset_name.endswith((".json", ".jsonl")):
        import json as _json
        print(f"  [local file] {dataset_name}")
        if dataset_name.endswith(".jsonl"):
            with open(dataset_name) as f:
                data = [_json.loads(line) for line in f]
        else:
            with open(dataset_name) as f:
                data = _json.load(f)
        # Shuffle and select
        import random
        random.seed(42)
        if len(data) > n:
            data = random.sample(data, n)
        return [extract_text(s, dataset_name) for s in data]

    # HuggingFace dataset
    if dataset_config:
        ds = load_dataset(dataset_name, dataset_config, split=split, trust_remote_code=True)
    else:
        ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
    if len(ds) > n:
        ds = ds.shuffle(seed=42).select(range(n))
    return [extract_text(s, dataset_name) for s in ds]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--domain_dataset", required=True,
                        help="HF dataset name for domain data, e.g. oumi-ai/MetaMathQA-R1")
    parser.add_argument("--domain_config", default=None,
                        help="Optional dataset config name")
    parser.add_argument("--domain_split", default="train")
    parser.add_argument("--general_dataset", default="cais/mmlu")
    parser.add_argument("--general_config", default="all")
    parser.add_argument("--general_split", default="test")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=4,
                        help="Top-K experts to select per layer (DAS-LoRA hyperparameter)")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    print(f"Loading model {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()

    print(f"Loading domain dataset: {args.domain_dataset} (split={args.domain_split})")
    domain_samples = load_samples(args.domain_dataset, args.domain_split,
                                   args.max_samples, args.domain_config)
    print(f"Loaded {len(domain_samples)} domain samples")

    print(f"Loading general dataset: {args.general_dataset} (split={args.general_split})")
    general_samples = load_samples(args.general_dataset, args.general_split,
                                    args.max_samples, args.general_config)
    print(f"Loaded {len(general_samples)} general samples")

    print("\n=== Computing routing scores on DOMAIN data ===")
    domain_scores = collect_routing_scores(
        model, tokenizer, domain_samples, args.max_seq_len, desc="domain"
    )

    print("\n=== Computing routing scores on GENERAL data ===")
    general_scores = collect_routing_scores(
        model, tokenizer, general_samples, args.max_seq_len, desc="general"
    )

    # DAS = domain - general
    das = domain_scores - general_scores  # [n_layers, n_experts]

    # Per-layer top-k selection
    top_k_experts = []
    for layer_idx in range(das.shape[0]):
        top_indices = das[layer_idx].argsort(descending=True)[: args.top_k].tolist()
        top_k_experts.append(top_indices)

    # Cumulative DAS@k (a sanity / informativeness metric from the paper)
    pos_das = das.clamp(min=0)
    cumulative_das_at_k = []
    for layer_idx in range(das.shape[0]):
        layer_pos = pos_das[layer_idx]
        total = layer_pos.sum().item()
        if total > 0:
            top_pos = layer_pos.sort(descending=True).values[: args.top_k].sum().item()
            cumulative_das_at_k.append(top_pos / total)
        else:
            cumulative_das_at_k.append(0.0)

    output = {
        "base_model": args.base_model,
        "domain_dataset": args.domain_dataset,
        "general_dataset": args.general_dataset,
        "max_samples": args.max_samples,
        "top_k": args.top_k,
        "selected_experts_per_layer": top_k_experts,
        "das_per_layer": das.tolist(),
        "cumulative_das_at_k": cumulative_das_at_k,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n=== DAS computation complete ===")
    print(f"Saved to: {args.output}")
    print(f"Selected experts per layer (top-{args.top_k}):")
    for li, experts in enumerate(top_k_experts):
        print(f"  Layer {li:2d}: {experts}")
    avg_cdas = sum(cumulative_das_at_k) / len(cumulative_das_at_k)
    print(f"\nAverage C-DAS@{args.top_k}: {avg_cdas:.3f}")
    print(f"  (higher = more concentrated; ~0.5+ is typical for fine-tunable signal)")


if __name__ == "__main__":
    main()

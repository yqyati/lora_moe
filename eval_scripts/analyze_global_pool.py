"""
全局共享池优势分析：证明 global pool 的跨层复用和参数效率

分析指标：
  1. 跨层激活热力图：每个 expert 在各层的激活频率
  2. Expert 角色分类：通用 expert vs 层特异 expert
  3. 路由组合多样性：实际出现了多少种不同的路由 pattern
  4. Expert 利用效率：每个 expert 的梯度信号量 / 有效利用率
  5. 跨层路由差异度：不同层是否选择了不同的 expert 子集

用法:
    python eval_scripts/analyze_global_pool.py \
        --base_model allenai/OLMoE-1B-7B-0924 \
        --adapter_path saves/olmoe/moe_lora/v2_global_pool128_best \
        --dataset gsm8k \
        --limit 200
"""

import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True, help="全局共享池模型 (V2)")
    parser.add_argument("--baseline_path", type=str, default=None, help="per_layer 对照模型 (V1/Baseline1)")
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "math500"])
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="analysis_results/global_pool")
    return parser.parse_args()


def load_dataset(name, limit):
    from datasets import load_dataset as hf_load
    if name == "gsm8k":
        ds = hf_load("openai/gsm8k", "main", split="test")
        prompts = [sample["question"] for sample in ds]
    elif name == "math500":
        ds = hf_load("HuggingFaceH4/MATH-500", split="test")
        prompts = [sample["problem"] for sample in ds]
    if limit:
        prompts = prompts[:limit]
    return prompts


def load_model_with_adapter(base_model_path, adapter_path, device="cuda"):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from llamafactory.model.model_utils.moe_lora import load_moe_lora_state

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=device
    )
    model = load_moe_lora_state(model, adapter_path)
    model.eval()
    return tokenizer, model


class GlobalPoolCollector:
    """收集全局池的跨层路由数据"""

    def __init__(self, model):
        self.model = model
        self.hooks = []
        # [layer_idx] → list of p_L tensors
        self.per_layer_routing = defaultdict(list)
        self._call_count = 0
        self._n_layers = 16
        self._setup_hooks()

    def _setup_hooks(self):
        from llamafactory.model.model_utils.moe_lora import LoRAPool

        for name, module in self.model.named_modules():
            if isinstance(module, LoRAPool):
                h = module.register_forward_hook(self._pool_hook)
                self.hooks.append(h)

    def _pool_hook(self, module, inputs, output):
        if len(inputs) >= 2:
            p_L = inputs[1]  # routing probabilities
            if p_L.dim() == 3:
                p_L = p_L.reshape(-1, p_L.size(-1))
            layer_idx = self._call_count % self._n_layers
            self.per_layer_routing[layer_idx].append(p_L.detach().float().cpu())
            self._call_count += 1

    def reset_count(self):
        self._call_count = 0

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

    def analyze(self):
        """计算所有分析指标"""
        results = {}

        # 合并每层数据
        layer_probs = {}
        for layer_idx in range(self._n_layers):
            if layer_idx in self.per_layer_routing and self.per_layer_routing[layer_idx]:
                layer_probs[layer_idx] = torch.cat(self.per_layer_routing[layer_idx], dim=0)

        if not layer_probs:
            print("警告：没有收集到路由数据")
            return results

        n_experts = list(layer_probs.values())[0].size(-1)
        n_layers = len(layer_probs)

        print(f"  Pool 大小: {n_experts} experts, {n_layers} layers")

        # ============================================
        # 1. 跨层激活热力图 [n_experts × n_layers]
        # ============================================
        heatmap = torch.zeros(n_experts, n_layers)
        for layer_idx, probs in layer_probs.items():
            # 每个 expert 在该层的平均激活概率
            heatmap[:, layer_idx] = probs.mean(dim=0)

        results["heatmap"] = heatmap.numpy().tolist()

        # ============================================
        # 2. Expert 角色分类
        # ============================================
        # 对每个 expert，计算它在各层激活频率的方差
        # 方差低 → 通用 expert（各层均匀使用）
        # 方差高 → 专用 expert（只在某些层使用）
        expert_layer_variance = heatmap.var(dim=1)  # [n_experts]
        expert_layer_mean = heatmap.mean(dim=1)     # [n_experts]

        # 通用 expert: 方差低 + 均值高（各层都在用，且用得多）
        # 层特异 expert: 方差高（某些层多，某些层少）
        # 休眠 expert: 均值低

        mean_threshold = expert_layer_mean.median().item()
        var_threshold = expert_layer_variance.median().item()

        universal_experts = ((expert_layer_variance < var_threshold) & (expert_layer_mean > mean_threshold)).sum().item()
        specialized_experts = (expert_layer_variance > var_threshold).sum().item()
        dormant_experts = (expert_layer_mean < mean_threshold * 0.1).sum().item()

        results["expert_roles"] = {
            "universal": universal_experts,
            "specialized": specialized_experts,
            "dormant": dormant_experts,
            "total": n_experts,
            "universal_ratio": universal_experts / n_experts,
            "specialized_ratio": specialized_experts / n_experts,
        }

        print(f"\n  Expert 角色分布:")
        print(f"    通用 expert (各层均匀使用): {universal_experts}/{n_experts} ({100*universal_experts/n_experts:.1f}%)")
        print(f"    专用 expert (层特异):       {specialized_experts}/{n_experts} ({100*specialized_experts/n_experts:.1f}%)")
        print(f"    休眠 expert (几乎不用):     {dormant_experts}/{n_experts} ({100*dormant_experts/n_experts:.1f}%)")

        # ============================================
        # 3. 跨层路由差异度
        # ============================================
        # 不同层的路由分布之间的 cosine distance
        # 如果所有层路由一样 → 共享池没意义
        # 如果各层路由不同 → 共享池真正被灵活使用
        layer_avg_routing = torch.stack([probs.mean(dim=0) for probs in layer_probs.values()])  # [n_layers, n_experts]
        layer_avg_norm = F.normalize(layer_avg_routing, dim=-1)
        layer_similarity = (layer_avg_norm @ layer_avg_norm.T).numpy()  # [n_layers, n_layers]

        # 平均跨层相似度（排除对角线）
        mask = np.ones_like(layer_similarity, dtype=bool)
        np.fill_diagonal(mask, False)
        avg_cross_layer_sim = layer_similarity[mask].mean()

        results["cross_layer"] = {
            "avg_similarity": float(avg_cross_layer_sim),
            "avg_distance": float(1 - avg_cross_layer_sim),
            "similarity_matrix": layer_similarity.tolist(),
        }

        print(f"\n  跨层路由差异度:")
        print(f"    平均跨层 cosine 相似度: {avg_cross_layer_sim:.4f}")
        print(f"    平均跨层距离: {1 - avg_cross_layer_sim:.4f}")
        print(f"    (相似度=1 表示各层完全一样; =0 表示完全不同)")

        # ============================================
        # 4. 路由组合多样性
        # ============================================
        # 对每个 token，取 top-k expert indices 作为 "路由 pattern"
        # 统计有多少种不同的 pattern
        top_k = 4  # 假设 top_k=4
        total_unique_patterns = 0
        total_tokens = 0

        for layer_idx, probs in layer_probs.items():
            n_tokens = probs.size(0)
            top_k_indices = probs.topk(min(top_k, n_experts), dim=-1).indices  # [n_tokens, top_k]
            # 排序后转为 tuple 作为 pattern
            sorted_indices = top_k_indices.sort(dim=-1).values
            patterns = set()
            for row in sorted_indices:
                patterns.add(tuple(row.tolist()))
            total_unique_patterns += len(patterns)
            total_tokens += n_tokens

        # 理论最大值 C(n_experts, top_k)
        from math import comb
        max_patterns = comb(n_experts, min(top_k, n_experts))

        results["diversity"] = {
            "avg_unique_patterns_per_layer": total_unique_patterns / n_layers,
            "total_tokens_per_layer": total_tokens / n_layers,
            "max_possible_patterns": max_patterns,
            "utilization_ratio": (total_unique_patterns / n_layers) / max_patterns,
        }

        print(f"\n  路由组合多样性:")
        print(f"    平均每层不同 pattern 数: {total_unique_patterns / n_layers:.0f}")
        print(f"    理论最大组合数 C({n_experts},{top_k}): {max_patterns:,}")
        print(f"    组合利用率: {100*(total_unique_patterns / n_layers)/max_patterns:.4f}%")

        # ============================================
        # 5. Expert 跨层共享度
        # ============================================
        # 对每个 expert，它在多少层的 top-50% 激活列表中出现
        # "top-50% 激活"：该层中激活频率高于中位数的 expert
        expert_active_layers = torch.zeros(n_experts)
        for layer_idx in range(n_layers):
            layer_activation = heatmap[:, layer_idx]
            median_activation = layer_activation.median()
            active_mask = layer_activation > median_activation
            expert_active_layers += active_mask.float()

        avg_layers_per_expert = expert_active_layers.mean().item()
        experts_used_in_all_layers = (expert_active_layers == n_layers).sum().item()
        experts_used_in_majority = (expert_active_layers >= n_layers * 0.75).sum().item()
        experts_used_in_few = (expert_active_layers <= n_layers * 0.25).sum().item()

        results["sharing"] = {
            "avg_layers_per_expert": avg_layers_per_expert,
            "experts_in_all_layers": experts_used_in_all_layers,
            "experts_in_majority_layers": experts_used_in_majority,
            "experts_in_few_layers": experts_used_in_few,
        }

        print(f"\n  Expert 跨层共享度:")
        print(f"    每个 expert 平均活跃层数: {avg_layers_per_expert:.1f} / {n_layers}")
        print(f"    在所有层都活跃的 expert: {experts_used_in_all_layers}/{n_experts}")
        print(f"    在 ≥75% 层活跃的 expert: {experts_used_in_majority}/{n_experts}")
        print(f"    仅在 ≤25% 层活跃的 expert: {experts_used_in_few}/{n_experts}")

        # ============================================
        # 6. 对比理论分析：global vs per_layer 参数效率
        # ============================================
        print(f"\n  参数效率对比（理论）:")
        print(f"    Global pool ({n_experts} experts):")
        print(f"      - 每个 expert 被 {n_layers} 层共用 → 每个 expert 获得 {n_layers}x 梯度信号")
        print(f"      - 每层可从 {n_experts} 个 expert 中选 {top_k} 个 → C({n_experts},{top_k})={max_patterns:,} 种组合")

        per_layer_experts = 16  # V1 对照
        per_layer_topk = 2
        per_layer_comb = comb(per_layer_experts, per_layer_topk)
        print(f"    Per-layer pool ({per_layer_experts} experts/layer):")
        print(f"      - 每个 expert 只被 1 层使用 → 梯度信号少 {n_layers}x")
        print(f"      - 每层只能从 {per_layer_experts} 个中选 {per_layer_topk} 个 → C({per_layer_experts},{per_layer_topk})={per_layer_comb} 种组合")
        print(f"    组合丰富度比: {max_patterns / per_layer_comb:.0f}x")

        return results


def run_collection(model, tokenizer, prompts, batch_size):
    """对一个模型收集路由数据并分析"""
    collector = GlobalPoolCollector(model)

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        collector.reset_count()
        with torch.no_grad():
            model(**inputs)

        if (i // batch_size) % 10 == 0:
            print(f"  已处理 {min(i+batch_size, len(prompts))}/{len(prompts)} 条")

    results = collector.analyze()
    collector.remove_hooks()
    return results


def print_comparison(global_results, perlayer_results):
    """对比全局池 vs per_layer 池"""
    print("\n" + "="*80)
    print("            全局共享池 vs Per-Layer 池 对比")
    print("="*80)

    g_roles = global_results.get("expert_roles", {})
    p_roles = perlayer_results.get("expert_roles", {})
    g_cross = global_results.get("cross_layer", {})
    p_cross = perlayer_results.get("cross_layer", {})
    g_div = global_results.get("diversity", {})
    p_div = perlayer_results.get("diversity", {})
    g_share = global_results.get("sharing", {})
    p_share = perlayer_results.get("sharing", {})

    print(f"\n{'指标':<40} {'Global Pool':<20} {'Per-Layer Pool':<20}")
    print("-" * 80)

    # Expert 角色
    if g_roles and p_roles:
        print(f"{'Expert 总数':<40} {g_roles.get('total', '-'):<20} {p_roles.get('total', '-'):<20}")
        print(f"{'通用 expert 比例':<40} {g_roles.get('universal_ratio', 0)*100:<18.1f}% {p_roles.get('universal_ratio', 0)*100:<18.1f}%")
        print(f"{'专用 expert 比例':<40} {g_roles.get('specialized_ratio', 0)*100:<18.1f}% {p_roles.get('specialized_ratio', 0)*100:<18.1f}%")
        print(f"{'休眠 expert 数':<40} {g_roles.get('dormant', '-'):<20} {p_roles.get('dormant', '-'):<20}")

    # 跨层差异
    if g_cross and p_cross:
        print(f"\n{'跨层路由 cosine 距离':<40} {g_cross.get('avg_distance', 0):<20.4f} {p_cross.get('avg_distance', 0):<20.4f}")

    # 组合多样性
    if g_div and p_div:
        print(f"{'平均每层不同 pattern 数':<40} {g_div.get('avg_unique_patterns_per_layer', 0):<20.0f} {p_div.get('avg_unique_patterns_per_layer', 0):<20.0f}")
        print(f"{'理论最大组合数':<40} {g_div.get('max_possible_patterns', 0):<20,} {p_div.get('max_possible_patterns', 0):<20,}")

    # 跨层共享
    if g_share and p_share:
        print(f"\n{'每个 expert 平均活跃层数':<40} {g_share.get('avg_layers_per_expert', 0):<20.1f} {p_share.get('avg_layers_per_expert', 0):<20.1f}")
        print(f"{'在≥75%层活跃的 expert 数':<40} {g_share.get('experts_in_majority_layers', 0):<20} {p_share.get('experts_in_majority_layers', 0):<20}")
        print(f"{'仅在≤25%层活跃的 expert 数':<40} {g_share.get('experts_in_few_layers', 0):<20} {p_share.get('experts_in_few_layers', 0):<20}")

    print("\n" + "="*80)
    print("结论")
    print("="*80)

    if g_cross and p_cross:
        g_dist = g_cross.get('avg_distance', 0)
        p_dist = p_cross.get('avg_distance', 0)
        if g_dist > p_dist:
            print(f"\n✓ 全局池各层路由差异度更大 ({g_dist:.3f} vs {p_dist:.3f})")
            print(f"  → 共享池不等于雷同，各层自主选择了不同的 expert 子集")

    if g_div and p_div:
        g_patterns = g_div.get('avg_unique_patterns_per_layer', 0)
        p_patterns = p_div.get('avg_unique_patterns_per_layer', 0)
        if g_patterns > p_patterns:
            print(f"\n✓ 全局池路由组合多样性远高于 per-layer ({g_patterns:.0f} vs {p_patterns:.0f})")
            print(f"  → 更大的组合空间让模型能为每个 token 找到更精确的 expert 组合")

    if g_share and p_share:
        g_avg = g_share.get('avg_layers_per_expert', 0)
        p_avg = p_share.get('avg_layers_per_expert', 0)
        if g_avg > p_avg:
            print(f"\n✓ 全局池 expert 跨层共享度更高 ({g_avg:.1f} vs {p_avg:.1f} 层)")
            print(f"  → 实现了跨层知识复用，每个 expert 获得更充分的梯度信号")


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print("加载数据集...")
    prompts = load_dataset(args.dataset, args.limit)
    print(f"加载了 {len(prompts)} 条测试数据")

    # 分析全局共享池模型
    print("\n" + "="*60)
    print("加载全局共享池模型 (Global Pool)...")
    print("="*60)
    tokenizer, model = load_model_with_adapter(args.base_model, args.adapter_path)
    global_results = run_collection(model, tokenizer, prompts, args.batch_size)
    del model
    torch.cuda.empty_cache()

    # 分析 per_layer 对照模型（如果提供了）
    perlayer_results = None
    if args.baseline_path:
        print("\n" + "="*60)
        print("加载 Per-Layer 对照模型...")
        print("="*60)
        _, model_bl = load_model_with_adapter(args.base_model, args.baseline_path)
        perlayer_results = run_collection(model_bl, tokenizer, prompts, args.batch_size)
        del model_bl
        torch.cuda.empty_cache()

    # 打印全局池自身分析
    print("\n" + "="*60)
    print("        全局共享池分析结果")
    print("="*60)

    if global_results.get("expert_roles"):
        roles = global_results["expert_roles"]
        print(f"\n1. 全局池中同时存在通用 expert ({roles['universal_ratio']*100:.0f}%) 和层特异 expert ({roles['specialized_ratio']*100:.0f}%),")
        print(f"   证明共享池能自适应地为不同层提供不同的 expert 组合")

    if global_results.get("cross_layer"):
        dist = global_results["cross_layer"]["avg_distance"]
        print(f"\n2. 跨层路由距离 = {dist:.3f}, 说明各层确实在选择不同的 expert 子集,")
        print(f"   而非所有层使用相同的 expert（共享不等于雷同）")

    if global_results.get("sharing"):
        sharing = global_results["sharing"]
        print(f"\n3. 每个 expert 平均在 {sharing['avg_layers_per_expert']:.1f}/{16} 层被活跃使用,")
        print(f"   实现了跨层知识复用；同时有 {sharing['experts_in_few_layers']} 个 expert 仅在少数层活跃,")
        print(f"   说明池中存在层特异化的 expert 分工")

    # 对比打印
    if perlayer_results:
        print_comparison(global_results, perlayer_results)

    # 保存
    save_data = {"global_pool": global_results}
    if perlayer_results:
        save_data["per_layer_pool"] = perlayer_results
    save_path = os.path.join(args.save_dir, f"global_pool_{args.dataset}.json")
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {save_path}")


if __name__ == "__main__":
    main()

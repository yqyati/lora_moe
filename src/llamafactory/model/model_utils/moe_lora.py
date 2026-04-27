# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MoE-LoRA Routing Compression: 在 MoE Transformer 主干上引入 LoRA 分支，
LoRA 的 routing 不独立学习，而是直接复用 MoE 的 router_logits 投影压缩而来:

    p_L = softmax(W @ router_logits),  W: N_E -> N_L

详见工程实现大纲: lora_moe/工程实现大纲.md
"""

import json
import math
import os
from typing import TYPE_CHECKING, List, Optional

import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrainerCallback

from ...extras import logging


if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


# OLMoE 第一阶段支持，后续按需扩展（见工程实现大纲 Step 6）
SUPPORTED_MOE_BLOCK_NAMES = {
    "OlmoeSparseMoeBlock",        # OLMoE
    "Qwen3MoeSparseMoeBlock",     # Qwen3-MoE
    "DeepseekV2MoE",              # DeepSeek-V2
}


# ---------------------------------------------------------------------------
# 核心组件
# ---------------------------------------------------------------------------


class RoutingProjection(nn.Module):
    """投影 raw router_logits 到 LoRA routing space。

    输入: router_logits in R^{..., N_E}（稠密、未 softmax）
    流程: Linear(N_E -> N_L) -> softmax
    输出: p_L in R^{..., N_L}

    内置 entropy 监控 buffer，forward 时累加（no_grad 不影响训练）。
    """

    def __init__(self, n_experts: int, n_lora: int):
        super().__init__()
        self.proj = nn.Linear(n_experts, n_lora, bias=False)
        # 监控用 buffer（long 类型不会被 .to(dtype=bfloat16) 转换）
        self.register_buffer("entropy_sum", torch.zeros(1), persistent=False)
        self.register_buffer("entropy_count", torch.zeros(1, dtype=torch.long), persistent=False)

    def forward(self, router_logits: torch.Tensor) -> torch.Tensor:
        p_L = torch.softmax(self.proj(router_logits), dim=-1)
        with torch.no_grad():
            ent = -(p_L * torch.log(p_L + 1e-8)).sum(dim=-1).mean()
            self.entropy_sum += ent.detach()
            self.entropy_count += 1
        return p_L


class LoRAExpert(nn.Module):
    """单个 LoRA adapter
    A: d -> r（Kaiming 初始化）, B: r -> d（零初始化）
    forward(h) -> (alpha / rank) * B(A(h))

    零初始化 B 保证训练初期 lora_output ≈ 0，不破坏 base model。
    """

    def __init__(self, d_model: int, rank: int, alpha: int):
        super().__init__()
        self.A = nn.Linear(d_model, rank, bias=False)
        self.B = nn.Linear(rank, d_model, bias=False)
        self.scaling = alpha / rank

        # 标准 LoRA 初始化
        #nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        #nn.init.zeros_(self.B.weight)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))                                                                                
        nn.init.normal_(self.B.weight, std=1e-4)  

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.scaling * self.B(self.A(h))


class LoRAPool(nn.Module):
    """N_L 个 LoRAExpert 的池
    forward(h, p_L) -> top-k 选择 + 加权和

    h:    [..., d_model]      MoE block 入口的 hidden_states
    p_L:  [..., n_experts]    routing 概率分布
    返回: [..., d_model]      LoRA 分支的输出（与 h 同形状）

    内置激活计数 buffer，监控每个 expert 的使用频率（防 dead expert）。
    """

    def __init__(self, n_experts: int, d_model: int, rank: int, alpha: int, top_k: int):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.experts = nn.ModuleList(
            [LoRAExpert(d_model, rank, alpha) for _ in range(n_experts)]
        )
        # 监控用 buffer（long 类型不会被 .to(dtype=bfloat16) 转换，避免精度丢失）
        self.register_buffer("activation_count", torch.zeros(n_experts, dtype=torch.long), persistent=False)
        self.register_buffer("total_tokens", torch.zeros(1, dtype=torch.long), persistent=False)
        # 全生命周期累计（不被 on_log 清零，用于训练后剪枝 dead experts）
        self.register_buffer("lifetime_activation_count", torch.zeros(n_experts, dtype=torch.long), persistent=False)

    def forward(self, h: torch.Tensor, p_L: torch.Tensor) -> torch.Tensor:
        # 对齐 dtype：p_L 在 autocast 下的 softmax 会被升到 float32，但 h / out / lora
        # expert 输出都是 bfloat16，下游 out[active] = ... 要求 dtype 一致，否则报
        # "Index put requires the source and destination dtypes match"。
        p_L_orig = p_L
        p_L = p_L.to(h.dtype)
        topk_vals, topk_idx = p_L.topk(self.top_k, dim=-1)  # [..., k]

        with torch.no_grad():
            flat_idx = topk_idx.reshape(-1)
            ones = torch.ones_like(flat_idx)
            self.activation_count.scatter_add_(0, flat_idx, ones)
            self.lifetime_activation_count.scatter_add_(0, flat_idx, ones)
            n_tokens = 1
            for s in h.shape[:-1]:
                n_tokens *= s
            self.total_tokens += n_tokens

        # load balancing loss: L = N * Σ(f_i * p_i)
        if self.training:
            n_tokens_f = float(h.shape[0]) if h.dim() == 2 else float(h.shape[0] * h.shape[1])
            # f_i: expert i 被 top-k 选中的 token 比例（不可导）
            one_hot = torch.zeros(int(n_tokens_f), self.n_experts, device=h.device, dtype=p_L_orig.dtype)
            one_hot.scatter_(1, topk_idx.reshape(int(n_tokens_f), self.top_k), 1.0)
            f = one_hot.mean(dim=0)
            # p_i: expert i 的平均路由概率（可导，梯度从这里流）
            p = p_L_orig.reshape(int(n_tokens_f), -1).mean(dim=0)
            self._aux_loss = self.n_experts * (f * p).sum()
        else:
            self._aux_loss = None

        out = torch.zeros_like(h)
        # 对每个 expert，找出在 top-k 中选中它的位置，加权累加
        for i in range(self.n_experts):
            # mask: [..., k]，标记该 expert 在 top-k 第几位
            expert_mask = topk_idx == i
            if not expert_mask.any():
                continue
            # 对每个 token，计算该 expert 的总权重（如果在多个 top-k 位置都被选中）
            weight = (topk_vals * expert_mask).sum(dim=-1, keepdim=True)  # [..., 1]
            active = (weight.squeeze(-1) > 0)  # [...]
            if not active.any():
                continue
            # 只对该 expert 被激活的 token 跑 forward
            out_i = self.experts[i](h[active])  # [N_active, d_model]
            # 强制对齐到 out.dtype：trainer 的混合精度策略可能把 trainable 参数（含
            # LoRA expert）保留在 float32，输出也是 float32；而 out 是 bfloat16，
            # in-place index_put 要求两边 dtype 完全一致，这里兜底一下。
            update = (weight[active] * out_i).to(out.dtype)
            out[active] = out[active] + update
        return out


# ---------------------------------------------------------------------------
# Monkey-patch 注入逻辑
# ---------------------------------------------------------------------------


def patched_moe_forward(self, hidden_states: torch.Tensor):
    """注入到 OLMoE / Qwen3-MoE 等 MoE block 上的 forward。

    依赖以下属性（由 inject_moe_lora 在 monkey-patch 时挂上）：
        self._original_forward    原始 MoE forward
        self._finetuning_args     训练超参（用于读 detach_p_e 等开关）
        self.routing_projection   RoutingProjection 实例
        self.lora_pool            LoRAPool 实例
    """
    # 1. 原 MoE 计算 + 取出 router_logits
    #    旧版 transformers: forward 返回 (final_hidden_states, router_logits)
    #    新版 transformers (OLMoE): forward 只返回 final_hidden_states，
    #      需要额外调用 self.gate 拿 router_logits。注意 OlmoeTopKRouter.forward
    #      返回三元组 (router_logits, router_scores, router_indices)。
    original_result = self._original_forward(hidden_states)
    if isinstance(original_result, tuple):
        moe_output, router_logits, *rest = original_result
    else:
        moe_output = original_result
        rest = []
        # 自己跑 raw logits，绕开 OlmoeTopKRouter 内部的 float32 softmax
        # （否则 router_logits 会变成 float32 概率，下游和 bfloat16 的 h 不兼容）
        if hidden_states.dim() == 3:
            h_flat_for_gate = hidden_states.reshape(-1, hidden_states.shape[-1])
        else:
            h_flat_for_gate = hidden_states
        gate_bias = getattr(self.gate, "bias", None)
        router_logits = F.linear(h_flat_for_gate, self.gate.weight, gate_bias)

    # 2. LoRA 分支
    detach = getattr(self._finetuning_args, "moe_lora_detach_p_e", False)
    logits_for_lora = router_logits.detach() if detach else router_logits
    routing_mode = getattr(self._finetuning_args, "moe_lora_routing_mode", "learned")

    # OLMoE 内部把 hidden_states reshape 成 [B*T, D] 后再调 gate，
    # 所以 router_logits 形状是 [B*T, N_E]，而 hidden_states 仍是 [B, T, D]。
    # 让 LoRA 分支也用 flatten 形式，避免形状不匹配。
    if router_logits.dim() == 2 and hidden_states.dim() == 3:
        b, t, d = hidden_states.shape
        h_flat = hidden_states.reshape(-1, d)
        if routing_mode == "follow_moe":
            p_L = _follow_moe_p_L(self, h_flat, logits_for_lora)
        else:
            p_L = self.routing_projection(logits_for_lora)     # [B*T, N_L]
        lora_output = self.lora_pool(h_flat, p_L)              # [B*T, D]
        lora_output = lora_output.reshape(b, t, d)
    else:
        if routing_mode == "follow_moe":
            p_L = _follow_moe_p_L(self, hidden_states, logits_for_lora)
        else:
            p_L = self.routing_projection(logits_for_lora)
        lora_output = self.lora_pool(hidden_states, p_L)

    # 3. 相加，保持原签名
    final_output = moe_output + lora_output
    if rest:
        return (final_output, *rest)
    return final_output


def _follow_moe_p_L(moe_block, h_flat: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
    """对照组路由：直接复用原 MoE router 的 top_k 选择，跳过 RoutingProjection。

    输出 p_L: [n_tokens, n_experts_moe] 稀疏权重矩阵，只在原 router 选中的 top_k
    位置上有值（= router 的 softmax 权重），其他位置为 0。下游 LoRAPool 会按这个
    分布做 top-k（n_lora == n_experts_moe，top_k_lora 通常等于原 MoE 的 top_k）。

    实现说明：直接对 router_logits 做 softmax + topk，**等价于** OlmoeTopKRouter 内部
    那一套（softmax → topk → 可选 norm_topk_prob）。这样就完全避开了
    RoutingProjection，参数也不再有 W。
    """
    n_experts = router_logits.shape[-1]
    top_k = moe_block.lora_pool.top_k
    probs = torch.softmax(router_logits, dim=-1, dtype=router_logits.dtype)
    top_vals, top_idx = probs.topk(top_k, dim=-1)
    p_L = torch.zeros_like(probs)
    p_L.scatter_(1, top_idx, top_vals)
    return p_L


def _resolve_target_layer_indices(target_layers: str, n_layers: int) -> List[int]:
    """解析 moe_lora_target_layers 配置成具体层号列表。

    支持: "all" / "first_half" / "last_half" / "last_third" / "0,5,10,15"
    """
    target_layers = target_layers.strip()
    if target_layers == "all":
        return list(range(n_layers))
    if target_layers == "first_half":
        return list(range(n_layers // 2))
    if target_layers == "last_half":
        return list(range(n_layers // 2, n_layers))
    if target_layers == "last_third":
        return list(range(2 * n_layers // 3, n_layers))
    # 显式层号列表，逗号分隔
    return [int(s) for s in target_layers.split(",") if s.strip()]


def _find_moe_blocks(model: "PreTrainedModel") -> List[nn.Module]:
    """遍历 model.model.layers，找到所有 MoE block。"""
    moe_blocks = []
    layers = getattr(model, "model", None)
    if layers is None or not hasattr(layers, "layers"):
        raise ValueError(
            "Cannot locate model.model.layers. Make sure the model is a "
            "decoder-style LM (e.g. OlmoeForCausalLM)."
        )
    for layer in layers.layers:
        # OLMoE 在 layer.mlp，DeepSeek-V2 也常在 layer.mlp 或 layer.feed_forward
        block = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
        if block is not None and type(block).__name__ in SUPPORTED_MOE_BLOCK_NAMES:
            moe_blocks.append(block)
    if not moe_blocks:
        raise ValueError(
            f"No MoE block found in model. Supported types: {SUPPORTED_MOE_BLOCK_NAMES}. "
            f"Got layer types: {[type(l).__name__ for l in layers.layers[:3]]}"
        )
    return moe_blocks


def inject_moe_lora(model: "PreTrainedModel", finetuning_args: "FinetuningArguments") -> None:
    """主入口: 在所有 MoE block 旁注入 LoRA 分支。

    1. 收集所有 MoE block
    2. 按 target_layers 过滤
    3. 按 pool_share / w_share 创建 RoutingProjection + LoRAPool（per-layer 或 global）
    4. cast 到 base model 的 device + dtype
    5. monkey-patch forward
    6. 重置 requires_grad（base model 全 freeze，新模块全 trainable）
    7. 校验 trainable 比例
    """
    n_lora = finetuning_args.moe_lora_n_experts
    rank = finetuning_args.moe_lora_rank
    alpha = finetuning_args.moe_lora_alpha
    top_k = finetuning_args.moe_lora_top_k
    pool_share = finetuning_args.moe_lora_pool_share
    w_share = finetuning_args.moe_lora_w_share
    target_layers = finetuning_args.moe_lora_target_layers
    routing_mode = getattr(finetuning_args, "moe_lora_routing_mode", "learned")

    # 取 base model 的 device 和 dtype（为 cast 用）
    sample_param = next(model.parameters())
    device, dtype = sample_param.device, sample_param.dtype

    d_model = model.config.hidden_size
    n_experts_moe = getattr(model.config, "num_experts", None) or model.config.num_experts_per_tok * 8
    if hasattr(model.config, "num_experts"):
        n_experts_moe = model.config.num_experts

    # follow_moe 模式：复用原 MoE router，n_lora 必须等于 n_experts_moe，
    # top_k 通常等于原 MoE 的 top_k（OLMoE 是 num_experts_per_tok）。
    if routing_mode == "follow_moe":
        if n_lora != n_experts_moe:
            raise ValueError(
                f"routing_mode='follow_moe' requires moe_lora_n_experts == n_experts_moe ({n_experts_moe}), "
                f"got {n_lora}. Set moe_lora_n_experts={n_experts_moe} in your yaml."
            )
        # w_share 在 follow_moe 下没有意义（不创建 W），强制 per_layer 防误解
        w_share = "per_layer"

    # 1. 收集 MoE block + 按 target_layers 过滤
    all_moe_blocks = _find_moe_blocks(model)
    target_indices = set(_resolve_target_layer_indices(target_layers, len(all_moe_blocks)))
    moe_blocks = [b for i, b in enumerate(all_moe_blocks) if i in target_indices]

    logger.info_rank0(
        f"MoE-LoRA injecting into {len(moe_blocks)}/{len(all_moe_blocks)} MoE layers | "
        f"N_E={n_experts_moe} -> N_L={n_lora} (rank={rank}, alpha={alpha}, top_k={top_k}) | "
        f"pool_share={pool_share}, w_share={w_share}, routing_mode={routing_mode}"
    )

    # 2. 全局共享时在 model 顶层挂一份
    shared_pool = None
    shared_proj = None
    if pool_share == "global":
        shared_pool = LoRAPool(n_lora, d_model, rank, alpha, top_k).to(device=device, dtype=dtype)
        model.add_module("global_lora_pool", shared_pool)
    if w_share == "global" and routing_mode == "learned":
        shared_proj = RoutingProjection(n_experts_moe, n_lora).to(device=device, dtype=dtype)
        model.add_module("global_routing_projection", shared_proj)

    # 3. 给每个 MoE block 挂模块 + monkey-patch forward
    for moe_block in moe_blocks:
        # RoutingProjection（follow_moe 模式下不创建任何 W）
        if routing_mode == "learned":
            if w_share == "per_layer":
                proj = RoutingProjection(n_experts_moe, n_lora).to(device=device, dtype=dtype)
                moe_block.routing_projection = proj
            else:
                # 全局共享时用 object.__setattr__ 绕过 nn.Module 的子模块自动注册：
                # 否则同一个 shared_proj 会同时挂在 model.global_routing_projection 和
                # 16 个 model.layers.X.mlp.routing_projection 下，state_dict 出现 17 份
                # 重复 key，HuggingFace save_pretrained 会因 shared tensors 报错。
                # shared_proj 已通过 add_module 注册到 model 顶层，参数仍被 optimizer 看到。
                object.__setattr__(moe_block, "routing_projection", shared_proj)

        # LoRAPool
        if pool_share == "per_layer":
            pool = LoRAPool(n_lora, d_model, rank, alpha, top_k).to(device=device, dtype=dtype)
            moe_block.lora_pool = pool
        else:
            # 同上理由，避免 shared_pool 在 state_dict 里出现 17 份。
            object.__setattr__(moe_block, "lora_pool", shared_pool)

        # monkey-patch forward
        moe_block._original_forward = moe_block.forward
        moe_block._finetuning_args = finetuning_args
        moe_block.forward = patched_moe_forward.__get__(moe_block)

    # 4. 重置 requires_grad（base model 全 freeze，注入的模块全 trainable）
    for name, p in model.named_parameters():
        if "routing_projection" in name or "lora_pool" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False

    # 5. 校验 trainable 比例
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    ratio = trainable / total if total > 0 else 0.0
    assert 0.00001 < ratio < 0.10, (
        f"Trainable ratio {ratio:.5f} out of expected range [1e-5, 0.1]. "
        f"trainable={trainable}, total={total}. "
        "Likely a bug in inject_moe_lora or freeze step."
    )

    # 6. 注册 balance loss hook（coef > 0 时生效，线性衰减到 0）
    balance_coef = getattr(finetuning_args, "moe_lora_balance_loss_coef", 0.0)
    if balance_coef > 0:
        model._balance_loss_state = {"step": 0, "total_steps": 1}

        def _add_balance_loss_hook(module, input, output):
            if not module.training or not hasattr(output, "loss") or output.loss is None:
                return output
            aux = torch.zeros(1, device=output.loss.device, dtype=output.loss.dtype)
            for m in module.modules():
                if isinstance(m, LoRAPool) and getattr(m, "_aux_loss", None) is not None:
                    aux = aux + m._aux_loss.to(aux.dtype)
                    m._aux_loss = None
            state = module._balance_loss_state
            progress = state["step"] / max(state["total_steps"], 1)
            effective_coef = balance_coef * (1.0 - progress)
            output.loss = output.loss + effective_coef * aux
            return output

        model.register_forward_hook(_add_balance_loss_hook)
        logger.info_rank0(f"MoE-LoRA balance loss enabled (coef={balance_coef}, linear decay)")


# ---------------------------------------------------------------------------
# 保存 / 加载
# ---------------------------------------------------------------------------


def save_moe_lora_state(
    model: nn.Module,
    save_dir: str,
    finetuning_args: "FinetuningArguments",
) -> None:
    """只保存 trainable 参数 + 配置到 save_dir。

    产出:
        moe_lora_state.safetensors   仅 trainable 权重
        moe_lora_config.json         重建结构所需的配置
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. 提取所有 trainable 参数
    state = {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    if not state:
        logger.warning_rank0("No trainable params found, skip saving moe_lora state.")
        return
    safetensors.torch.save_file(state, os.path.join(save_dir, "moe_lora_state.safetensors"))

    # 2. 保存配置（重建模型结构必需）
    config = {
        "moe_lora_n_experts": finetuning_args.moe_lora_n_experts,
        "moe_lora_rank": finetuning_args.moe_lora_rank,
        "moe_lora_alpha": finetuning_args.moe_lora_alpha,
        "moe_lora_top_k": finetuning_args.moe_lora_top_k,
        "moe_lora_pool_share": finetuning_args.moe_lora_pool_share,
        "moe_lora_w_share": finetuning_args.moe_lora_w_share,
        "moe_lora_target_layers": finetuning_args.moe_lora_target_layers,
        "moe_lora_detach_p_e": finetuning_args.moe_lora_detach_p_e,
        "moe_lora_routing_mode": getattr(finetuning_args, "moe_lora_routing_mode", "learned"),
        "moe_lora_balance_loss_coef": getattr(finetuning_args, "moe_lora_balance_loss_coef", 0.0),
        # 元信息（不属于 finetuning_args，加载时要 pop 掉）
        "_meta_base_model": getattr(model.config, "name_or_path", "unknown"),
    }
    with open(os.path.join(save_dir, "moe_lora_config.json"), "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info_rank0(
        f"Saved MoE-LoRA state ({len(state)} tensors) to {save_dir}"
    )

    # 3. 保存全生命周期激活统计（用于训练后剪枝）
    pools = [(n, m) for n, m in model.named_modules() if isinstance(m, LoRAPool)]
    if pools:
        lifetime_stats = {}
        for name, pool in pools:
            counts = pool.lifetime_activation_count.cpu().tolist()
            dead = [i for i, c in enumerate(counts) if c == 0]
            lifetime_stats[name] = {
                "activation_count": counts,
                "dead_expert_indices": dead,
                "n_dead": len(dead),
                "n_total": len(counts),
            }
        with open(os.path.join(save_dir, "moe_lora_lifetime_stats.json"), "w") as f:
            json.dump(lifetime_stats, f, indent=2)
        total_dead = sum(s["n_dead"] for s in lifetime_stats.values())
        total_experts = sum(s["n_total"] for s in lifetime_stats.values())
        logger.info_rank0(
            f"Lifetime dead experts: {total_dead}/{total_experts}"
        )


def load_moe_lora_state(model: "PreTrainedModel", load_dir: str) -> "PreTrainedModel":
    """先重建结构（注入），再加载权重。用于评估 / 续训。"""
    from ...hparams import FinetuningArguments

    config_path = os.path.join(load_dir, "moe_lora_config.json")
    state_path = os.path.join(load_dir, "moe_lora_state.safetensors")
    if not os.path.exists(config_path) or not os.path.exists(state_path):
        raise FileNotFoundError(
            f"Missing moe_lora_config.json or moe_lora_state.safetensors in {load_dir}"
        )

    # 1. 读配置
    with open(config_path) as f:
        config = json.load(f)

    # 2. pop 元信息字段
    meta = {k: config.pop(k) for k in list(config.keys()) if k.startswith("_meta_")}

    # 3. 用配置重建结构
    finetuning_args = FinetuningArguments(finetuning_type="moe_lora", **config)
    inject_moe_lora(model, finetuning_args)

    # 4. 加载权重
    state = safetensors.torch.load_file(state_path)
    missing, unexpected = model.load_state_dict(state, strict=False)

    # 5. 双向校验（防 silent fail）
    assert len(unexpected) == 0, f"Unexpected weights in checkpoint: {unexpected[:5]}"
    unloaded_lora = [n for n in missing if "routing_projection" in n or "lora_pool" in n]
    assert not unloaded_lora, (
        f"Failed to load LoRA params (naming mismatch?): {unloaded_lora[:5]}"
    )

    # 6. base model 一致性（warning, 不阻断）
    base_model_meta = meta.get("_meta_base_model")
    actual_base = getattr(model.config, "name_or_path", "unknown")
    if base_model_meta and base_model_meta != actual_base:
        logger.warning_rank0(
            f"Base model mismatch: loaded {actual_base} but checkpoint was trained on "
            f"{base_model_meta}. Behaviour may be unexpected."
        )

    logger.info_rank0(f"Loaded MoE-LoRA state from {load_dir} ({len(state)} tensors)")
    return model


# ---------------------------------------------------------------------------
# Trainer Callbacks
# ---------------------------------------------------------------------------


class MoELoRABalanceDecayCallback(TrainerCallback):
    """同步训练进度到 model._balance_loss_state，供 balance loss hook 做线性衰减。"""

    def on_step_begin(self, args, state, control, model: Optional[nn.Module] = None, **kwargs):
        if model is not None and hasattr(model, "_balance_loss_state"):
            model._balance_loss_state["step"] = state.global_step
            model._balance_loss_state["total_steps"] = state.max_steps


class MoELoRASaveCallback(TrainerCallback):
    """每次 trainer 触发 checkpoint 保存时，同步保存 MoE-LoRA 自定义状态。

    注意: trainer 默认 checkpoint 路径是 {output_dir}/checkpoint-{global_step}，
    这个 callback 跟随该约定。如果改了 save_strategy 自定义路径，需要相应调整。
    """

    def __init__(self, finetuning_args: "FinetuningArguments"):
        self.finetuning_args = finetuning_args

    def on_save(self, args, state, control, model: Optional[nn.Module] = None, **kwargs):
        if model is None:
            return
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        save_moe_lora_state(model, ckpt_dir, self.finetuning_args)


class MoELoRAStatsCallback(TrainerCallback):
    """收集 MoE-LoRA 监控指标（每 logging_steps 触发），写入 trainer log
    （Trainer 会自动转发到 wandb / tensorboard / mlflow）。

    监控指标:
        moe_lora/{block}/activation_max         expert 最大激活频率
        moe_lora/{block}/activation_min         expert 最小激活频率
        moe_lora/{block}/dead_experts           频率 < 1e-4 的 expert 数
        moe_lora/{block}/activation_imbalance   max_freq / ideal_uniform 比值
        moe_lora/{block}/W_fro_norm             投影矩阵 W 的 Frobenius 范数
        moe_lora/{block}/p_L_entropy            p_L 分布熵（按 batch 平均）
        moe_lora/{*}_avg                         以上指标的全局平均
    """

    def on_log(self, args, state, control, model: Optional[nn.Module] = None,
               logs: Optional[dict] = None, **kwargs):
        if logs is None or model is None:
            return

        pools = [(n, m) for n, m in model.named_modules() if isinstance(m, LoRAPool)]
        projs = [(n, m) for n, m in model.named_modules() if isinstance(m, RoutingProjection)]

        # ① 激活频率
        for name, pool in pools:
            if pool.total_tokens.item() == 0:
                continue
            freq = pool.activation_count.float() / pool.total_tokens.float()
            ideal = pool.top_k / pool.n_experts
            logs[f"moe_lora/{name}/activation_max"] = freq.max().item()
            logs[f"moe_lora/{name}/activation_min"] = freq.min().item()
            logs[f"moe_lora/{name}/dead_experts"] = (freq < 1e-4).sum().item()
            logs[f"moe_lora/{name}/activation_imbalance"] = (freq.max() / (ideal + 1e-8)).item()
            # wandb histogram：每个 expert 的激活频率分布
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        f"moe_lora/{name}/activation_freq": wandb.Histogram(freq.cpu().numpy()),
                        f"moe_lora/{name}/lifetime_activation": wandb.Histogram(
                            pool.lifetime_activation_count.float().cpu().numpy()
                        ),
                    }, commit=False)
            except ImportError:
                pass
            pool.activation_count.zero_()
            pool.total_tokens.zero_()

        # ② W 的 Frobenius 范数
        w_norms = []
        for name, proj in projs:
            w_norm = proj.proj.weight.detach().float().norm().item()
            logs[f"moe_lora/{name}/W_fro_norm"] = w_norm
            w_norms.append(w_norm)
        if w_norms:
            logs["moe_lora/W_fro_norm_avg"] = sum(w_norms) / len(w_norms)

        # ③ p_L entropy（直接读 RoutingProjection 内置 buffer）
        entropies = []
        for name, proj in projs:
            if proj.entropy_count.item() == 0:
                continue
            avg_ent = (proj.entropy_sum / proj.entropy_count).item()
            logs[f"moe_lora/{name}/p_L_entropy"] = avg_ent
            entropies.append(avg_ent)
            proj.entropy_sum.zero_()
            proj.entropy_count.zero_()
        if entropies:
            logs["moe_lora/p_L_entropy_avg"] = sum(entropies) / len(entropies)

        # ④ balance loss（读取最近一次 forward 的值）
        aux_losses = []
        for name, pool in pools:
            if getattr(pool, "_aux_loss", None) is not None:
                aux_losses.append(pool._aux_loss.item())
        if aux_losses:
            logs["moe_lora/balance_loss"] = sum(aux_losses) / len(aux_losses)

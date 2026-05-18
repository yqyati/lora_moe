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
    "DeepseekV2MoE",              # DeepSeek-V2 (custom modeling_deepseek.py)
    "DeepseekV2Moe",              # DeepSeek-V2 (transformers 5.x native, 注意大小写)
    "DeepseekV3MoE",              # DeepSeek-V3 (custom)
    "DeepseekV3Moe",              # DeepSeek-V3 (transformers 5.x native)
    "Qwen3_5MoeSparseMoeBlock",   # Qwen3.5-MoE (multimodal, transformers 5.2+)
}


# ---------------------------------------------------------------------------
# 核心组件
# ---------------------------------------------------------------------------


class RoutingProjection(nn.Module):
    """投影 raw router_logits 到 LoRA routing space。

    输入: router_logits in R^{..., N_E}（稠密、未 softmax）
    流程: Linear(N_E -> N_L) -> softmax
    输出: p_L in R^{..., N_L}

    可选: hidden_bottleneck_dim > 0 时,从 hidden state 抽 token-specific 信号
    与 router_logits 拼接后喂给 W_l(突破 64-d 瓶颈)。

    内置 entropy 监控 buffer，forward 时累加（no_grad 不影响训练）。
    """

    def __init__(self, n_experts: int, n_lora: int,
                 d_model: int = 0, hidden_bottleneck_dim: int = 0,
                 das_init: Optional[torch.Tensor] = None,
                 das_init_strength: float = 1.0,
                 wl_init: str = "default"):
        super().__init__()
        self.hidden_bottleneck_dim = hidden_bottleneck_dim
        if hidden_bottleneck_dim > 0:
            assert d_model > 0, "d_model required when hidden_bottleneck_dim > 0"
            # 从 hidden state 抽 token-specific 信号
            self.h_proj = nn.Linear(d_model, hidden_bottleneck_dim, bias=False)
            input_dim = n_experts + hidden_bottleneck_dim
        else:
            self.h_proj = None
            input_dim = n_experts
        self.proj = nn.Linear(input_dim, n_lora, bias=False)

        # DAS-informed initialization (optional, takes precedence over wl_init):
        # hard-partition the N_E base experts by DAS rank into N_L groups,
        # assign each group to a distinct LoRA expert via strong positive weight.
        # 其余位置保留 Kaiming default init,确保训练后仍可演化。
        if das_init is not None:
            assert das_init.shape[0] == n_experts, \
                f"das_init dim {das_init.shape[0]} != n_experts {n_experts}"
            with torch.no_grad():
                sorted_idx = das_init.argsort(descending=True)  # high to low
                if n_lora <= n_experts:
                    # 多对一: 每个 LoRA expert 拥有 (N_E / N_L) 个 base expert
                    n_per_group = n_experts // n_lora
                    for lora_i in range(n_lora):
                        start = lora_i * n_per_group
                        end = start + n_per_group if lora_i < n_lora - 1 else n_experts
                        group = sorted_idx[start:end]
                        self.proj.weight.data[lora_i, group] = das_init_strength
                else:
                    # n_lora > n_experts: 每 base expert 被多个 LoRA "共享"(round-robin)
                    # LoRA i 映射到 DAS 排名 (i mod N_E) 的 base expert
                    for lora_i in range(n_lora):
                        base_idx = sorted_idx[lora_i % n_experts].item()
                        self.proj.weight.data[lora_i, base_idx] = das_init_strength
        elif wl_init == "anchor":
            # Anchor init (DAS-free): identity-like on z-channel, zero on h-channel.
            # 让 LoRA routing 在 step 0 镜像 base routing 的一个 chunked 版本——
            # 利用 base router 的 pretrained 结构作为"锚",避免随机 W_l 在 day 1
            # 触发 routing collapse;training 中 W_l 渐进偏移学 task-specific routing。
            with torch.no_grad():
                self.proj.weight.data.zero_()
                if n_lora <= n_experts:
                    n_per_group = n_experts // n_lora
                    for lora_i in range(n_lora):
                        start = lora_i * n_per_group
                        end = start + n_per_group if lora_i < n_lora - 1 else n_experts
                        self.proj.weight.data[lora_i, start:end] = 1.0
                else:
                    for lora_i in range(n_lora):
                        base_idx = lora_i % n_experts
                        self.proj.weight.data[lora_i, base_idx] = 1.0
                # h-channel (columns n_experts:) 已 zero,无需再设

        # 监控用 buffer（long 类型不会被 .to(dtype=bfloat16) 转换）
        self.register_buffer("entropy_sum", torch.zeros(1), persistent=False)
        self.register_buffer("entropy_count", torch.zeros(1, dtype=torch.long), persistent=False)

    def forward(self, router_logits: torch.Tensor,
                hidden_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.h_proj is not None:
            assert hidden_state is not None, (
                "hidden_state required when hidden_bottleneck_dim > 0"
            )
            h_c = self.h_proj(hidden_state)
            routing_input = torch.cat([router_logits, h_c], dim=-1)
        else:
            routing_input = router_logits
        p_L = torch.softmax(self.proj(routing_input), dim=-1)
        with torch.no_grad():
            ent = -(p_L * torch.log(p_L + 1e-8)).sum(dim=-1).mean()
            self.entropy_sum += ent.detach()
            self.entropy_count += 1
        return p_L


class IndependentRouter(nn.Module):
    """独立 router (Baseline2 / LoRAMoE-style).

    直接从 hidden_states 学路由，不依赖 MoE 的 router_logits。
    输入: h in R^{..., d_model}
    输出: p_L in R^{..., n_lora} (softmax 概率)
    """

    def __init__(self, d_model: int, n_lora: int):
        super().__init__()
        self.gate = nn.Linear(d_model, n_lora, bias=False)
        self.register_buffer("entropy_sum", torch.zeros(1), persistent=False)
        self.register_buffer("entropy_count", torch.zeros(1, dtype=torch.long), persistent=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        p_L = torch.softmax(self.gate(h), dim=-1)
        with torch.no_grad():
            ent = -(p_L * torch.log(p_L + 1e-8)).sum(dim=-1).mean()
            self.entropy_sum += ent.detach()
            self.entropy_count += 1
        return p_L


class ResidualRouter(nn.Module):
    """残差叠加路由：W1 @ router_logits + W2 @ h

    结合 MoE routing 先验（64 维）和 hidden state 信息（2048 维）。
    训练后通过 W1/W2 的 norm 比值可量化两路信号的相对贡献。
    """

    def __init__(self, n_experts: int, d_model: int, n_lora: int):
        super().__init__()
        self.proj_moe = nn.Linear(n_experts, n_lora, bias=False)
        self.proj_h = nn.Linear(d_model, n_lora, bias=False)
        self.register_buffer("entropy_sum", torch.zeros(1), persistent=False)
        self.register_buffer("entropy_count", torch.zeros(1, dtype=torch.long), persistent=False)

    def forward(self, router_logits: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        logits = self.proj_moe(router_logits) + self.proj_h(h)
        p_L = torch.softmax(logits, dim=-1)
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
        self.rank = rank
        self.scaling = alpha / rank
        self.experts = nn.ModuleList(
            [LoRAExpert(d_model, rank, alpha) for _ in range(n_experts)]
        )
        self._use_grouped_mm = hasattr(torch.nn.functional, "grouped_mm")
        # 监控用 buffer（long 类型不会被 .to(dtype=bfloat16) 转换，避免精度丢失）
        self.register_buffer("activation_count", torch.zeros(n_experts, dtype=torch.long), persistent=False)
        self.register_buffer("total_tokens", torch.zeros(1, dtype=torch.long), persistent=False)
        # 全生命周期累计（不被 on_log 清零，用于训练后剪枝 dead experts）
        self.register_buffer("lifetime_activation_count", torch.zeros(n_experts, dtype=torch.long), persistent=False)

    def forward(self, h: torch.Tensor, p_L: torch.Tensor) -> torch.Tensor:
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
        # 注意: 多层共享同一 LoRAPool 时(global / grouped pool 默认),后续层会
        # 覆盖前面层的 _aux_loss,实际只反映"本 batch 最后一层"的 balance 信号。
        # 这是历史行为(OLMoE 训练时如此),保留以便复现。
        # 累加版会让 balance loss 数量级 ≈ layers_per_pool× 放大,需要相应调整 coef。
        if self.training:
            n_tokens_f = float(h.shape[0]) if h.dim() == 2 else float(h.shape[0] * h.shape[1])
            one_hot = torch.zeros(int(n_tokens_f), self.n_experts, device=h.device, dtype=p_L_orig.dtype)
            one_hot.scatter_(1, topk_idx.reshape(int(n_tokens_f), self.top_k), 1.0)
            f = one_hot.mean(dim=0)
            p = p_L_orig.reshape(int(n_tokens_f), -1).mean(dim=0)
            self._aux_loss = self.n_experts * (f * p).sum()
        else:
            self._aux_loss = None

        if self._use_grouped_mm:
            return self._forward_grouped_mm(h, topk_vals, topk_idx)
        else:
            return self._forward_loop(h, topk_vals, topk_idx)

    def dispatch(self, h: torch.Tensor, topk_vals: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
        """跳过内部 top-K, 直接用外部传入的 (topk_vals, topk_idx) 做 expert dispatch.

        用于 joint top-K 场景(per-layer + cross-layer global pool 联合竞争).
        topk_idx 必须在 [0, self.n_experts) 范围内; 没选中的 slot 应填 idx=0,
        vals=0(权重 0 时输出 = 0 * expert(h) = 0, 不污染结果).

        不更新 activation_count 也不算 balance loss (joint 模式由外部 forward 统筹).
        """
        topk_vals = topk_vals.to(h.dtype)
        if self._use_grouped_mm:
            return self._forward_grouped_mm(h, topk_vals, topk_idx)
        else:
            return self._forward_loop(h, topk_vals, topk_idx)

    def _forward_grouped_mm(self, h: torch.Tensor, topk_vals: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
        """使用 torch.nn.functional.grouped_mm 的高效实现。"""
        orig_shape = h.shape
        d = h.shape[-1]
        h_flat = h.reshape(-1, d)  # [N, d]
        N = h_flat.shape[0]
        k = topk_idx.shape[-1]   # 从 topk_idx 推断, 兼容 dispatch 模式(K 可能 != self.top_k)

        # 展开: 每个 token 复制 k 份，对应 k 个选中的 expert
        h_expanded = h_flat.unsqueeze(1).expand(N, k, d).reshape(N * k, d)  # [N*k, d]
        expert_ids = topk_idx.reshape(N * k)  # [N*k]
        weights = topk_vals.reshape(N * k)  # [N*k]

        # 按 expert_id 排序，grouped_mm 要求同组 token 连续
        sorted_order = expert_ids.argsort(stable=True)
        h_sorted = h_expanded[sorted_order]  # [N*k, d]
        expert_ids_sorted = expert_ids[sorted_order]
        weights_sorted = weights[sorted_order]  # [N*k]

        # 计算每个 expert 的 token 数量 -> cumulative offsets
        counts = torch.zeros(self.n_experts, dtype=torch.int32, device=h.device)
        counts.scatter_add_(0, expert_ids_sorted.to(torch.int64), torch.ones(N * k, dtype=torch.int32, device=h.device))
        offs = counts.cumsum(0).to(torch.int32)  # [n_experts]

        # 堆叠所有 expert 的权重
        A_weight = torch.stack([e.A.weight for e in self.experts])  # [n_experts, rank, d]
        B_weight = torch.stack([e.B.weight for e in self.experts])  # [n_experts, d, rank]

        # grouped_mm 要求 stride 是 16 字节对齐；rank 较小时需要 pad
        elem_size = A_weight.element_size()  # 2 for bf16/fp16
        align_n = 16 // elem_size  # elements per 16B boundary
        padded_rank = (self.rank + align_n - 1) // align_n * align_n
        if padded_rank != self.rank:
            pad_n = padded_rank - self.rank
            A_weight = F.pad(A_weight, (0, 0, 0, pad_n))         # [n_experts, padded_rank, d]
            B_weight = F.pad(B_weight, (0, pad_n))               # [n_experts, d, padded_rank]

        # grouped_mm: h_sorted @ A^T -> [N*k, padded_rank]
        compute_dtype = h_sorted.dtype
        intermediate = torch.nn.functional.grouped_mm(
            h_sorted.to(A_weight.dtype), A_weight.transpose(-2, -1).contiguous(), offs=offs
        )
        # grouped_mm: intermediate @ B^T -> [N*k, d]
        output_sorted = torch.nn.functional.grouped_mm(
            intermediate, B_weight.transpose(-2, -1).contiguous(), offs=offs
        )

        # 应用 scaling 和 routing weight
        output_sorted = (self.scaling * weights_sorted.unsqueeze(-1)) * output_sorted

        # 反排序 + 聚合 k 个 expert 的输出
        output_expanded = torch.zeros(N * k, d, device=h.device, dtype=output_sorted.dtype)
        output_expanded[sorted_order] = output_sorted
        out = output_expanded.reshape(N, k, d).sum(dim=1)  # [N, d]

        return out.to(compute_dtype).reshape(orig_shape)

    def _forward_loop(self, h: torch.Tensor, topk_vals: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
        """Fallback: 逐 expert for 循环（兼容旧版 PyTorch）。"""
        out = torch.zeros_like(h)
        for i in range(self.n_experts):
            expert_mask = topk_idx == i
            if not expert_mask.any():
                continue
            weight = (topk_vals * expert_mask).sum(dim=-1, keepdim=True)
            active = (weight.squeeze(-1) > 0)
            if not active.any():
                continue
            out_i = self.experts[i](h[active])
            update = (weight[active] * out_i).to(out.dtype)
            out[active] = out[active] + update
        return out




def _apply_lora_with_optional_cross_pool(moe_block, h_input, p_L):
    """LoRA dispatch helper:
    - 无 cross-layer pool: 走原 LoRAPool.forward 路径(单 pool 自己 top-K + balance loss).
    - 有 cross-layer pool: joint top-K (over n_local+n_global), 按 idx<n_local 拆分
      dispatch 到 per-layer 和 cross-layer pool, 各自 dispatch 后加和.
      Joint 模式不算 balance loss(由 LoRAPool.dispatch 跳过).
    """
    cross_pool = getattr(moe_block, "cross_layer_lora_pool", None)
    if cross_pool is None:
        return moe_block.lora_pool(h_input, p_L)

    n_local = moe_block.lora_pool.n_experts
    K = moe_block.lora_pool.top_k  # joint top-K (per-layer 和 cross-layer 共用 K)

    p_L_h = p_L.to(h_input.dtype)
    topk_vals, topk_idx = p_L_h.topk(K, dim=-1)               # (..., K)
    local_mask = topk_idx < n_local                            # bool, (..., K)
    local_mask_f = local_mask.to(topk_vals.dtype)

    local_idx = topk_idx.masked_fill(~local_mask, 0)
    local_vals = topk_vals * local_mask_f
    global_idx = (topk_idx - n_local).masked_fill(local_mask, 0)
    global_vals = topk_vals * (1.0 - local_mask_f)

    out_local = moe_block.lora_pool.dispatch(h_input, local_vals, local_idx)
    out_global = cross_pool.dispatch(h_input, global_vals, global_idx)
    return out_local + out_global


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
        # detach gate.weight 以避免 DeepSpeed ZeRO 把它误认为被 trainable 两次访问。
        # 在 PEFT 场景下 gate.weight 是 frozen 的,这次 F.linear 只是"读取"它来计算
        # router_logits 给 RoutingProjection,不需要梯度流经它。detach 保证 DeepSpeed
        # 的 grad reduce hook 不会被错误地触发两次。
        router_logits = F.linear(
            h_flat_for_gate, self.gate.weight.detach(),
            gate_bias.detach() if gate_bias is not None else None,
        )

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
        elif routing_mode == "independent":
            p_L = self.independent_router(h_flat)
        elif routing_mode == "residual":
            p_L = self.residual_router(logits_for_lora, h_flat)
        else:
            # learned: optional hidden state bottleneck
            if getattr(self.routing_projection, "h_proj", None) is not None:
                p_L = self.routing_projection(logits_for_lora, h_flat)
            else:
                p_L = self.routing_projection(logits_for_lora)     # [B*T, N_L (+ N_global)]
        lora_output = _apply_lora_with_optional_cross_pool(self, h_flat, p_L)   # [B*T, D]
        lora_output = lora_output.reshape(b, t, d)
    else:
        if routing_mode == "follow_moe":
            p_L = _follow_moe_p_L(self, hidden_states, logits_for_lora)
        elif routing_mode == "independent":
            p_L = self.independent_router(hidden_states)
        elif routing_mode == "residual":
            p_L = self.residual_router(logits_for_lora, hidden_states)
        else:
            if getattr(self.routing_projection, "h_proj", None) is not None:
                p_L = self.routing_projection(logits_for_lora, hidden_states)
            else:
                p_L = self.routing_projection(logits_for_lora)
        lora_output = _apply_lora_with_optional_cross_pool(self, hidden_states, p_L)

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


def _locate_decoder_layers(model: "PreTrainedModel") -> Optional[nn.Module]:
    """定位 decoder 层列表，兼容纯文本 LM 和多模态 LM。

    优先级:
    1. model.model.layers          (OLMoE / Qwen3-MoE / DeepSeek-V2 等纯文本)
    2. model.model.text_model.layers / model.model.language_model.layers  (Qwen3.5-VL 等多模态)
    3. model.language_model.model.layers  (其他多模态 wrapper)
    返回 layers 容器 (有 .layers 属性的 module)，找不到返回 None。
    """
    base = getattr(model, "model", None)
    if base is not None and hasattr(base, "layers"):
        return base
    # 多模态: model.model.text_model / language_model
    if base is not None:
        for sub_name in ("text_model", "language_model"):
            sub = getattr(base, sub_name, None)
            if sub is not None and hasattr(sub, "layers"):
                return sub
            # 再下一层 wrapper: text_model.model.layers
            if sub is not None:
                inner = getattr(sub, "model", None)
                if inner is not None and hasattr(inner, "layers"):
                    return inner
    # 顶层 language_model wrapper
    lm = getattr(model, "language_model", None)
    if lm is not None:
        if hasattr(lm, "layers"):
            return lm
        inner = getattr(lm, "model", None)
        if inner is not None and hasattr(inner, "layers"):
            return inner
    return None


def _find_moe_blocks(model: "PreTrainedModel") -> List[nn.Module]:
    """遍历 decoder layers，找到所有 MoE block。"""
    moe_blocks = []
    layers = _locate_decoder_layers(model)
    if layers is None:
        raise ValueError(
            "Cannot locate decoder layers. Tried model.model.layers, "
            "model.model.text_model.layers, model.model.language_model.layers, "
            "model.language_model.(model.)layers."
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

    # 多模态模型 (Qwen3.5-VL 等) 把 LM 配置嵌在 text_config 下，纯文本模型直接放顶层。
    # 先在顶层找，找不到再 fallback 到 text_config，保持对 OLMoE/Qwen3/DeepSeek 完全向后兼容。
    text_config = getattr(model.config, "text_config", None)

    def _cfg_get(name: str):
        v = getattr(model.config, name, None)
        if v is None and text_config is not None:
            v = getattr(text_config, name, None)
        return v

    d_model = _cfg_get("hidden_size")
    if d_model is None:
        raise RuntimeError("Could not determine hidden_size from model.config (or text_config).")

    # transformers 4.x: num_experts; transformers 5.x: num_local_experts (Qwen3-MoE);
    # DeepSeek-V2 / V3: n_routed_experts (shared experts 不参与 routing,不算 N_E)
    n_experts_moe = (
        _cfg_get("num_experts")
        or _cfg_get("num_local_experts")
        or _cfg_get("n_routed_experts")
    )
    if n_experts_moe is None:
        fields = list(vars(model.config).keys())[:30]
        if text_config is not None:
            fields += ["text_config." + k for k in list(vars(text_config).keys())[:30]]
        raise RuntimeError(
            "Could not determine N_E from config. Tried num_experts, num_local_experts, n_routed_experts "
            f"(in both top-level and text_config). Available config fields: {fields}"
        )

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

    # Hidden state bottleneck(可选,默认 0 = 关闭,vanilla V2)
    hidden_bottleneck_dim = getattr(finetuning_args, "moe_lora_hidden_bottleneck_dim", 0)

    # Hierarchical/Grouped Pool(可选,默认 1 = 单一 global pool)
    n_groups = getattr(finetuning_args, "moe_lora_n_groups", 1)

    # DAS-informed initialization for RoutingProjection (可选,默认关闭)
    # 提供 das_qwen3_math.json / das_qwen3_code.json 等文件,按每层 DAS rank 把
    # base experts hard-partition 给 N_L 个 LoRA expert,防止训练初期 expert collapse。
    das_init_path = getattr(finetuning_args, "moe_lora_das_init_path", None)
    das_init_strength = getattr(finetuning_args, "moe_lora_das_init_strength", 1.0)
    wl_init = getattr(finetuning_args, "moe_lora_wl_init", "default")
    n_global = getattr(finetuning_args, "moe_lora_n_global", 0)
    das_per_layer_tensor = None
    if das_init_path:
        import json as _json
        with open(das_init_path) as f:
            das_data = _json.load(f)
        das_per_layer_tensor = torch.tensor(das_data["das_per_layer"], dtype=torch.float32)
        logger.info_rank0(
            f"Loaded DAS init from {das_init_path}: "
            f"shape={tuple(das_per_layer_tensor.shape)}, strength={das_init_strength}"
        )

    # 1. 收集 MoE block + 按 target_layers 过滤
    all_moe_blocks = _find_moe_blocks(model)
    target_indices = set(_resolve_target_layer_indices(target_layers, len(all_moe_blocks)))
    moe_blocks = [b for i, b in enumerate(all_moe_blocks) if i in target_indices]

    # n_groups 校验
    if n_groups > 1:
        if pool_share != "global":
            raise ValueError(f"n_groups={n_groups} requires pool_share='global', got '{pool_share}'.")
        if n_lora % n_groups != 0:
            raise ValueError(f"n_lora={n_lora} must be divisible by n_groups={n_groups}.")
        if len(moe_blocks) % n_groups != 0:
            logger.warning_rank0(
                f"layer_count={len(moe_blocks)} not divisible by n_groups={n_groups}; "
                "groups will be slightly unbalanced."
            )

    n_lora_per_group = n_lora // n_groups  # = n_lora when n_groups=1

    bottleneck_str = f", hidden_bottleneck={hidden_bottleneck_dim}" if hidden_bottleneck_dim > 0 else ""
    groups_str = f", n_groups={n_groups} (n_lora_per_group={n_lora_per_group})" if n_groups > 1 else ""
    logger.info_rank0(
        f"MoE-LoRA injecting into {len(moe_blocks)}/{len(all_moe_blocks)} MoE layers | "
        f"N_E={n_experts_moe} -> N_L={n_lora} (rank={rank}, alpha={alpha}, top_k={top_k}) | "
        f"pool_share={pool_share}, w_share={w_share}, routing_mode={routing_mode}{bottleneck_str}{groups_str}"
    )

    # 2. 全局共享时在 model 顶层挂 pool(支持 grouped pool: n_groups > 1 时挂多个)
    shared_pool = None         # n_groups=1 时使用
    grouped_pools = None       # n_groups>1 时使用,长度 n_groups
    shared_proj = None
    shared_indep_router = None
    shared_residual_router = None

    if pool_share == "global":
        if n_groups == 1:
            # 单一 global pool(原始行为)
            shared_pool = LoRAPool(n_lora, d_model, rank, alpha, top_k).to(device=device, dtype=dtype)
            model.add_module("global_lora_pool", shared_pool)
        else:
            # Grouped Pool: n_groups 个独立 LoRAPool,每个有 n_lora_per_group 个 expert
            grouped_pools = []
            for g in range(n_groups):
                pool = LoRAPool(n_lora_per_group, d_model, rank, alpha, top_k).to(device=device, dtype=dtype)
                model.add_module(f"global_lora_pool_g{g}", pool)
                grouped_pools.append(pool)

    # 2c. (可选) cross-layer global pool: 跨层共享一个 routed LoRA pool,
    # 跟 per-layer pool 联合 joint top-K 竞争(joint 模式不算 balance loss).
    # global pool 的 rank / alpha 可以跟 per-layer pool 不同 (通过 moe_lora_global_rank/alpha 设置)
    cross_layer_pool = None
    if n_global > 0:
        if pool_share != "per_layer":
            raise ValueError(
                f"moe_lora_n_global={n_global} requires moe_lora_pool_share='per_layer' "
                f"(got '{pool_share}'). Cross-layer pool is meant to complement per-layer pool, "
                f"not stack on top of an existing global pool."
            )
        if routing_mode != "learned":
            raise ValueError(
                f"moe_lora_n_global={n_global} requires moe_lora_routing_mode='learned' "
                f"(got '{routing_mode}'). Joint top-K requires W_l to output joint logits."
            )
        if n_groups != 1:
            raise ValueError(f"moe_lora_n_global={n_global} requires n_groups=1, got {n_groups}.")
        global_rank = getattr(finetuning_args, "moe_lora_global_rank", 0) or rank
        global_alpha = getattr(finetuning_args, "moe_lora_global_alpha", 0) or alpha
        cross_layer_pool = LoRAPool(n_global, d_model, global_rank, global_alpha, top_k).to(device=device, dtype=dtype)
        model.add_module("cross_layer_lora_pool", cross_layer_pool)
        logger.info_rank0(
            f"MoE-LoRA: cross-layer global pool enabled with n_global={n_global} "
            f"(rank={global_rank}, alpha={global_alpha}). Per-layer W_l will output (N_local={n_lora}+N_global={n_global}) logits, "
            f"joint top-K={top_k} competes across both pools. Joint mode disables balance loss."
        )

    if w_share == "global" and routing_mode == "learned":
        # global w_share 不支持 grouped(每层 W_l 必须独立才能路由到自己组)
        if n_groups > 1:
            raise ValueError("w_share='global' is incompatible with n_groups>1; use w_share='per_layer'.")
        if das_per_layer_tensor is not None:
            logger.warning_rank0(
                "DAS init is set but w_share='global'; per-layer DAS cannot be applied to "
                "a shared W_l. Falling back to using layer-0 DAS as init prior."
            )
            global_das_init = das_per_layer_tensor[0]
        else:
            global_das_init = None
        shared_proj = RoutingProjection(
            n_experts_moe, n_lora, d_model=d_model, hidden_bottleneck_dim=hidden_bottleneck_dim,
            das_init=global_das_init, das_init_strength=das_init_strength,
            wl_init=wl_init,
        ).to(device=device, dtype=dtype)
        model.add_module("global_routing_projection", shared_proj)
    if w_share == "global" and routing_mode == "independent":
        shared_indep_router = IndependentRouter(d_model, n_lora).to(device=device, dtype=dtype)
        model.add_module("global_independent_router", shared_indep_router)
    if w_share == "global" and routing_mode == "residual":
        shared_residual_router = ResidualRouter(n_experts_moe, d_model, n_lora).to(device=device, dtype=dtype)
        model.add_module("global_residual_router", shared_residual_router)

    # 3. 给每个 MoE block 挂模块 + monkey-patch forward
    for layer_idx, moe_block in enumerate(moe_blocks):
        # 计算该层属于哪个 group(n_groups=1 时永远是 0)
        group_idx = layer_idx * n_groups // len(moe_blocks)

        # RoutingProjection / IndependentRouter (follow_moe 模式下不创建任何 router)
        # 注意: grouped pool 时 W_l 输出维度是 n_lora_per_group(而非 n_lora)
        # 注意: n_global > 0 时 W_l 输出 (n_lora_per_group + n_global), joint top-K
        proj_out_dim = n_lora_per_group + n_global  # n_global=0 时退化为原行为

        if routing_mode == "learned":
            if w_share == "per_layer":
                # 取该层对应的 DAS 信号(per-layer init prior)
                layer_das = (
                    das_per_layer_tensor[layer_idx]
                    if das_per_layer_tensor is not None and layer_idx < das_per_layer_tensor.shape[0]
                    else None
                )
                proj = RoutingProjection(
                    n_experts_moe, proj_out_dim, d_model=d_model, hidden_bottleneck_dim=hidden_bottleneck_dim,
                    das_init=layer_das, das_init_strength=das_init_strength,
                    wl_init=wl_init,
                ).to(device=device, dtype=dtype)
                moe_block.routing_projection = proj
            else:
                object.__setattr__(moe_block, "routing_projection", shared_proj)
        elif routing_mode == "independent":
            if w_share == "per_layer":
                indep_router = IndependentRouter(d_model, proj_out_dim).to(device=device, dtype=dtype)
                moe_block.independent_router = indep_router
            else:
                object.__setattr__(moe_block, "independent_router", shared_indep_router)
        elif routing_mode == "residual":
            if w_share == "per_layer":
                res_router = ResidualRouter(n_experts_moe, d_model, proj_out_dim).to(device=device, dtype=dtype)
                moe_block.residual_router = res_router
            else:
                object.__setattr__(moe_block, "residual_router", shared_residual_router)

        # LoRAPool
        if pool_share == "per_layer":
            pool = LoRAPool(n_lora, d_model, rank, alpha, top_k).to(device=device, dtype=dtype)
            moe_block.lora_pool = pool
        elif n_groups > 1:
            # Grouped pool: 该层指向所属 group 的 pool
            object.__setattr__(moe_block, "lora_pool", grouped_pools[group_idx])
        else:
            # 单一 global pool
            object.__setattr__(moe_block, "lora_pool", shared_pool)

        # (可选) cross-layer global pool: 每层 reference 同一个 model 级 pool
        if cross_layer_pool is not None:
            object.__setattr__(moe_block, "cross_layer_lora_pool", cross_layer_pool)

        # monkey-patch forward
        moe_block._original_forward = moe_block.forward
        moe_block._finetuning_args = finetuning_args
        moe_block.forward = patched_moe_forward.__get__(moe_block)

    # 4. 重置 requires_grad（base model 全 freeze，注入的模块全 trainable）
    for name, p in model.named_parameters():
        if "routing_projection" in name or "lora_pool" in name or "independent_router" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False

    # 5. 校验 trainable 比例(兼容 DeepSpeed ZeRO-3:用 ds_numel 而非 numel)
    def _real_numel(p):
        # ZeRO-3 sharded params 使用 ds_numel 拿到全量,否则 numel() 只能拿到本地 shard
        return getattr(p, "ds_numel", None) or p.numel()
    trainable = sum(_real_numel(p) for p in model.parameters() if p.requires_grad)
    total = sum(_real_numel(p) for p in model.parameters())
    ratio = trainable / total if total > 0 else 0.0
    if not (0.00001 < ratio < 0.10):
        # ZeRO-3 早期阶段 base model 可能还没 materialize,降级为 warning
        logger.warning(
            f"Trainable ratio {ratio:.5f} out of expected range [1e-5, 0.1]. "
            f"trainable={trainable}, total={total}. "
            "Under ZeRO-3 this may be due to params not yet materialized; verify after first step."
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

    # ZeRO-3 兼容: 在 save 前 gather 所有 trainable params 的全量副本
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    try:
        import deepspeed
        gathered_ctx = deepspeed.zero.GatheredParameters(trainable_params, modifier_rank=0)
    except Exception:
        gathered_ctx = None

    # 1. 提取所有 trainable 参数(在 gather 上下文内)
    state = {}
    if gathered_ctx is not None:
        with gathered_ctx:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    state[name] = param.detach().cpu().clone()
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                state[name] = param.detach().cpu()

    if not state:
        logger.warning_rank0("No trainable params found, skip saving moe_lora state.")
        return

    # 只 rank 0 写入磁盘,避免多 rank 同时写同一文件
    is_rank0 = int(os.environ.get("LOCAL_RANK", "0")) == 0
    if not is_rank0:
        return
    safetensors.torch.save_file(state, os.path.join(save_dir, "moe_lora_state.safetensors"))

    # 2. 保存配置(重建模型结构必需)
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
        "moe_lora_hidden_bottleneck_dim": getattr(finetuning_args, "moe_lora_hidden_bottleneck_dim", 0),
        "moe_lora_n_groups": getattr(finetuning_args, "moe_lora_n_groups", 1),
        "moe_lora_wl_init": getattr(finetuning_args, "moe_lora_wl_init", "default"),
        "moe_lora_n_global": getattr(finetuning_args, "moe_lora_n_global", 0),
        "moe_lora_global_rank": getattr(finetuning_args, "moe_lora_global_rank", 0),
        "moe_lora_global_alpha": getattr(finetuning_args, "moe_lora_global_alpha", 0),
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
    unloaded_lora = [n for n in missing if "routing_projection" in n or "lora_pool" in n or "independent_router" in n]
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
        # 删除 Trainer 自动保存的完整基座模型,只保留 LoRA 状态
        import glob
        for pattern in ("model.safetensors", "model-*.safetensors", "model.safetensors.index.json"):
            for f in glob.glob(os.path.join(ckpt_dir, pattern)):
                try:
                    os.remove(f)
                    logger.info_rank0(f"Removed redundant base model file: {f}")
                except FileNotFoundError:
                    # 多进程可能已被其他 rank 删除
                    pass


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
        indep_routers = [(n, m) for n, m in model.named_modules() if isinstance(m, IndependentRouter)]

        # ① 激活频率
        # 跨 rank 聚合 activation_count / total_tokens,否则每个 rank 只看到 1/N batch,
        # 真实非 dead 的 expert 可能在本 rank 上看起来 dead → dead_experts 严重高估
        try:
            import torch.distributed as dist
            _dist_ok = dist.is_available() and dist.is_initialized()
        except Exception:
            _dist_ok = False

        for name, pool in pools:
            if _dist_ok:
                # all_reduce sum 到所有 rank(in-place),拿到全局真实计数
                # 用 clone 避免改 buffer 本身(后续还要 zero_)
                act_global = pool.activation_count.clone().to(torch.long)
                tot_global = pool.total_tokens.clone().to(torch.long)
                dist.all_reduce(act_global, op=dist.ReduceOp.SUM)
                dist.all_reduce(tot_global, op=dist.ReduceOp.SUM)
            else:
                act_global = pool.activation_count
                tot_global = pool.total_tokens

            if tot_global.item() == 0:
                continue
            freq = act_global.float() / tot_global.float()
            ideal = pool.top_k / pool.n_experts
            logs[f"moe_lora/{name}/activation_max"] = freq.max().item()
            logs[f"moe_lora/{name}/activation_min"] = freq.min().item()
            logs[f"moe_lora/{name}/dead_experts"] = (freq < 1e-4).sum().item()
            logs[f"moe_lora/{name}/activation_imbalance"] = (freq.max() / (ideal + 1e-8)).item()
            # wandb histogram：每个 expert 的激活频率分布
            try:
                import wandb
                if wandb.run is not None:
                    # wandb.Histogram 在所有值相同时(早期训练 freq 全 0)会崩
                    # → 用 try 兜底,数据无效就跳过 histogram
                    try:
                        wandb.log({
                            f"moe_lora/{name}/activation_freq": wandb.Histogram(freq.cpu().numpy()),
                            f"moe_lora/{name}/lifetime_activation": wandb.Histogram(
                                pool.lifetime_activation_count.float().cpu().numpy()
                            ),
                        }, commit=False)
                    except (ValueError, RuntimeError):
                        pass
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

        # ③ p_L entropy（直接读 RoutingProjection / IndependentRouter 内置 buffer）
        entropies = []
        for name, proj in projs:
            if proj.entropy_count.item() == 0:
                continue
            avg_ent = (proj.entropy_sum / proj.entropy_count).item()
            logs[f"moe_lora/{name}/p_L_entropy"] = avg_ent
            entropies.append(avg_ent)
            proj.entropy_sum.zero_()
            proj.entropy_count.zero_()
        for name, router in indep_routers:
            if router.entropy_count.item() == 0:
                continue
            avg_ent = (router.entropy_sum / router.entropy_count).item()
            logs[f"moe_lora/{name}/p_L_entropy"] = avg_ent
            entropies.append(avg_ent)
            router.entropy_sum.zero_()
            router.entropy_count.zero_()
        if entropies:
            logs["moe_lora/p_L_entropy_avg"] = sum(entropies) / len(entropies)

        # ④ balance loss（读取最近一次 forward 的值）
        aux_losses = []
        for name, pool in pools:
            if getattr(pool, "_aux_loss", None) is not None:
                aux_losses.append(pool._aux_loss.item())
        if aux_losses:
            logs["moe_lora/balance_loss"] = sum(aux_losses) / len(aux_losses)

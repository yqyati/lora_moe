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
        # 监控用 buffer（不算参数，不参与 state_dict 保存）
        self.register_buffer("entropy_sum", torch.zeros(1), persistent=False)
        self.register_buffer("entropy_count", torch.zeros(1), persistent=False)

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
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

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
        # 监控用 buffer
        self.register_buffer("activation_count", torch.zeros(n_experts), persistent=False)
        self.register_buffer("total_tokens", torch.zeros(1), persistent=False)

    def forward(self, h: torch.Tensor, p_L: torch.Tensor) -> torch.Tensor:
        topk_vals, topk_idx = p_L.topk(self.top_k, dim=-1)  # [..., k]

        with torch.no_grad():
            flat_idx = topk_idx.reshape(-1)
            ones = torch.ones_like(flat_idx, dtype=self.activation_count.dtype)
            self.activation_count.scatter_add_(0, flat_idx, ones)
            # token 数 = h 的所有 leading dim 乘积
            n_tokens = 1
            for s in h.shape[:-1]:
                n_tokens *= s
            self.total_tokens += n_tokens

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
            out[active] = out[active] + weight[active] * out_i
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
    #    OLMoE / Qwen3-MoE 都返回 (final_hidden_states, router_logits)
    #    DeepSeek-V2 等可能含额外字段，用保护性解包
    original_result = self._original_forward(hidden_states)
    if isinstance(original_result, tuple):
        moe_output, router_logits, *rest = original_result
    else:
        # 极少见情况：原 forward 只返回单个 tensor，我们就只能再调一次 gate
        moe_output = original_result
        rest = []
        router_logits = self.gate(hidden_states)

    # 2. LoRA 分支
    detach = getattr(self._finetuning_args, "moe_lora_detach_p_e", False)
    logits_for_lora = router_logits.detach() if detach else router_logits

    # OLMoE 内部把 hidden_states reshape 成 [B*T, D] 后再调 gate，
    # 所以 router_logits 形状是 [B*T, N_E]，而 hidden_states 仍是 [B, T, D]。
    # 让 LoRA 分支也用 flatten 形式，避免形状不匹配。
    if router_logits.dim() == 2 and hidden_states.dim() == 3:
        b, t, d = hidden_states.shape
        h_flat = hidden_states.reshape(-1, d)
        p_L = self.routing_projection(logits_for_lora)         # [B*T, N_L]
        lora_output = self.lora_pool(h_flat, p_L)              # [B*T, D]
        lora_output = lora_output.reshape(b, t, d)
    else:
        p_L = self.routing_projection(logits_for_lora)
        lora_output = self.lora_pool(hidden_states, p_L)

    # 3. 相加，保持原签名
    final_output = moe_output + lora_output
    if rest:
        return (final_output, *rest)
    return final_output


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

    # 取 base model 的 device 和 dtype（为 cast 用）
    sample_param = next(model.parameters())
    device, dtype = sample_param.device, sample_param.dtype

    d_model = model.config.hidden_size
    n_experts_moe = getattr(model.config, "num_experts", None) or model.config.num_experts_per_tok * 8
    if hasattr(model.config, "num_experts"):
        n_experts_moe = model.config.num_experts

    # 1. 收集 MoE block + 按 target_layers 过滤
    all_moe_blocks = _find_moe_blocks(model)
    target_indices = set(_resolve_target_layer_indices(target_layers, len(all_moe_blocks)))
    moe_blocks = [b for i, b in enumerate(all_moe_blocks) if i in target_indices]

    logger.info_rank0(
        f"MoE-LoRA injecting into {len(moe_blocks)}/{len(all_moe_blocks)} MoE layers | "
        f"N_E={n_experts_moe} -> N_L={n_lora} (rank={rank}, alpha={alpha}, top_k={top_k}) | "
        f"pool_share={pool_share}, w_share={w_share}"
    )

    # 2. 全局共享时在 model 顶层挂一份
    shared_pool = None
    shared_proj = None
    if pool_share == "global":
        shared_pool = LoRAPool(n_lora, d_model, rank, alpha, top_k).to(device=device, dtype=dtype)
        model.add_module("global_lora_pool", shared_pool)
    if w_share == "global":
        shared_proj = RoutingProjection(n_experts_moe, n_lora).to(device=device, dtype=dtype)
        model.add_module("global_routing_projection", shared_proj)

    # 3. 给每个 MoE block 挂模块 + monkey-patch forward
    for moe_block in moe_blocks:
        # RoutingProjection
        if w_share == "per_layer":
            proj = RoutingProjection(n_experts_moe, n_lora).to(device=device, dtype=dtype)
            moe_block.routing_projection = proj
        else:
            moe_block.routing_projection = shared_proj

        # LoRAPool
        if pool_share == "per_layer":
            pool = LoRAPool(n_lora, d_model, rank, alpha, top_k).to(device=device, dtype=dtype)
            moe_block.lora_pool = pool
        else:
            moe_block.lora_pool = shared_pool

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
        # 元信息（不属于 finetuning_args，加载时要 pop 掉）
        "_meta_base_model": getattr(model.config, "name_or_path", "unknown"),
    }
    with open(os.path.join(save_dir, "moe_lora_config.json"), "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info_rank0(
        f"Saved MoE-LoRA state ({len(state)} tensors) to {save_dir}"
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
            freq = pool.activation_count / pool.total_tokens
            ideal = pool.top_k / pool.n_experts
            logs[f"moe_lora/{name}/activation_max"] = freq.max().item()
            logs[f"moe_lora/{name}/activation_min"] = freq.min().item()
            logs[f"moe_lora/{name}/dead_experts"] = (freq < 1e-4).sum().item()
            logs[f"moe_lora/{name}/activation_imbalance"] = (freq.max() / (ideal + 1e-8)).item()
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

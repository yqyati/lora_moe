"""MoLA (NAACL 2024): Layer-wise expert allocation for LoRA-MoE.

复用 moe_lora.py 的 LoRAPool / IndependentRouter / patched_moe_forward,
只重新实现 inject 函数,支持 per-layer 不同的 LoRA expert 数量。

参考: Gao et al. 2024. "MoLA: Mixture of LoRA Experts for Large Language Models"
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, List, Optional

import safetensors.torch
import torch
import torch.nn as nn
from transformers import TrainerCallback

from ...extras.logging import get_logger
from .moe_lora import (
    IndependentRouter,
    LoRAPool,
    _find_moe_blocks,
    patched_moe_forward,
)

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)


def _parse_per_layer(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def inject_mola(model: "PreTrainedModel", finetuning_args: "FinetuningArguments") -> None:
    """MoLA 注入: 每层独立 LoRA pool,不同层用不同数量的 expert。

    1. 解析 per-layer expert 列表
    2. 校验长度等于 MoE 层数
    3. 每层创建对应大小的 IndependentRouter + LoRAPool
    4. 复用 moe_lora 的 patched_moe_forward
    """
    n_experts_list = _parse_per_layer(finetuning_args.mola_n_experts_per_layer)
    rank = finetuning_args.mola_rank
    alpha = finetuning_args.mola_alpha
    top_k = finetuning_args.mola_top_k

    sample_param = next(model.parameters())
    device, dtype = sample_param.device, sample_param.dtype
    d_model = model.config.hidden_size

    all_moe_blocks = _find_moe_blocks(model)
    if len(n_experts_list) != len(all_moe_blocks):
        raise ValueError(
            f"mola_n_experts_per_layer length {len(n_experts_list)} "
            f"!= number of MoE layers {len(all_moe_blocks)}. "
            f"Got list: {n_experts_list}"
        )

    total_experts = sum(n_experts_list)
    logger.info_rank0(
        f"MoLA injecting into {len(all_moe_blocks)} MoE layers | "
        f"per-layer experts: {n_experts_list} (sum={total_experts}) | "
        f"rank={rank}, alpha={alpha}, top_k={top_k}"
    )

    # 给每层挂上 layer-specific router + pool
    for layer_idx, moe_block in enumerate(all_moe_blocks):
        n_lora_l = n_experts_list[layer_idx]

        # MoLA 用 independent router from hidden state
        moe_block.independent_router = IndependentRouter(d_model, n_lora_l).to(
            device=device, dtype=dtype
        )
        moe_block.lora_pool = LoRAPool(n_lora_l, d_model, rank, alpha, top_k).to(
            device=device, dtype=dtype
        )

        # 给 patched_moe_forward 一个 fake finetuning_args(MoLA 用 independent 路由)
        moe_block._original_forward = moe_block.forward
        moe_block._finetuning_args = SimpleNamespace(
            moe_lora_routing_mode="independent",
            moe_lora_balance_loss_coef=getattr(finetuning_args, "mola_balance_loss_coef", 0.0),
            moe_lora_detach_p_e=False,
        )
        moe_block.forward = patched_moe_forward.__get__(moe_block)

    # 重置 requires_grad: base model freeze,新模块 trainable
    for name, p in model.named_parameters():
        if "independent_router" in name or "lora_pool" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False

    # 校验(兼容 DeepSpeed ZeRO-3:用 ds_numel 拿全量)
    def _real_numel(p):
        return getattr(p, "ds_numel", None) or p.numel()
    trainable = sum(_real_numel(p) for p in model.parameters() if p.requires_grad)
    total = sum(_real_numel(p) for p in model.parameters())
    logger.info_rank0(
        f"MoLA setup complete | trainable: {trainable / 1e6:.2f}M "
        f"({100 * trainable / total:.3f}%) | total: {total / 1e6:.2f}M"
    )
    if not (0.0001 < trainable / total < 0.05):
        logger.warning(
            f"MoLA trainable ratio {trainable / total:.4f} out of expected range (0.01% - 5%). "
            "Under ZeRO-3 this may be due to params not yet materialized."
        )


def save_mola_state(model: nn.Module, save_dir: str, finetuning_args: "FinetuningArguments") -> None:
    """保存 MoLA trainable 参数 + 配置。"""
    os.makedirs(save_dir, exist_ok=True)

    # ZeRO-3 兼容: gather 所有 trainable params 全量副本
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    try:
        import deepspeed
        gathered_ctx = deepspeed.zero.GatheredParameters(trainable_params, modifier_rank=0)
    except Exception:
        gathered_ctx = None

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
        logger.warning_rank0("No trainable params found, skip saving MoLA state.")
        return

    is_rank0 = int(os.environ.get("LOCAL_RANK", "0")) == 0
    if not is_rank0:
        return
    safetensors.torch.save_file(state, os.path.join(save_dir, "mola_state.safetensors"))

    config = {
        "mola_n_experts_per_layer": finetuning_args.mola_n_experts_per_layer,
        "mola_rank": finetuning_args.mola_rank,
        "mola_alpha": finetuning_args.mola_alpha,
        "mola_top_k": finetuning_args.mola_top_k,
        "mola_balance_loss_coef": getattr(finetuning_args, "mola_balance_loss_coef", 0.0),
        "_meta_base_model": getattr(model.config, "name_or_path", "unknown"),
    }
    with open(os.path.join(save_dir, "mola_config.json"), "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info_rank0(f"Saved MoLA state ({len(state)} tensors) to {save_dir}")


def load_mola_state(model: "PreTrainedModel", load_dir: str) -> "PreTrainedModel":
    """加载 MoLA checkpoint: 重建结构 + 加载权重。"""
    from ...hparams import FinetuningArguments

    config_path = os.path.join(load_dir, "mola_config.json")
    state_path = os.path.join(load_dir, "mola_state.safetensors")
    if not os.path.exists(config_path) or not os.path.exists(state_path):
        raise FileNotFoundError(
            f"Missing mola_config.json or mola_state.safetensors in {load_dir}"
        )

    with open(config_path) as f:
        config = json.load(f)

    # pop 元信息
    {k: config.pop(k) for k in list(config.keys()) if k.startswith("_meta_")}

    finetuning_args = FinetuningArguments(finetuning_type="mola", **config)
    inject_mola(model, finetuning_args)

    state = safetensors.torch.load_file(state_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert len(unexpected) == 0, f"Unexpected weights in checkpoint: {unexpected[:5]}"
    unloaded = [n for n in missing if "independent_router" in n or "lora_pool" in n]
    assert not unloaded, f"Failed to load MoLA weights: {unloaded[:5]}"

    logger.info_rank0(f"Loaded MoLA state from {load_dir}")
    return model


class MoLASaveCallback(TrainerCallback):
    """中间 checkpoint 保存(类似 MoELoRASaveCallback)。"""

    def __init__(self, finetuning_args: "FinetuningArguments"):
        self.finetuning_args = finetuning_args

    def on_save(self, args, state, control, model: Optional[nn.Module] = None, **kwargs):
        if model is None:
            return
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        save_mola_state(model, ckpt_dir, self.finetuning_args)
        # 删除 Trainer 自动保存的完整基座
        import glob
        for pattern in ("model.safetensors", "model-*.safetensors", "model.safetensors.index.json"):
            for f in glob.glob(os.path.join(ckpt_dir, pattern)):
                try:
                    os.remove(f)
                    logger.info_rank0(f"Removed redundant base model file: {f}")
                except FileNotFoundError:
                    pass

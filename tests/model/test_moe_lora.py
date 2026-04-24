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

"""moe_lora.py 单元测试。

不依赖真实 OLMoE/Qwen3-MoE 模型，使用 mock MoE block 测试核心行为。
"""

import os
import tempfile

import pytest
import torch
import torch.nn as nn

from llamafactory.model.model_utils.moe_lora import (
    LoRAExpert,
    LoRAPool,
    MoELoRASaveCallback,
    MoELoRAStatsCallback,
    RoutingProjection,
    inject_moe_lora,
    load_moe_lora_state,
    save_moe_lora_state,
)


D_MODEL = 32
N_EXPERTS_MOE = 16
N_LORA = 8
RANK = 4
ALPHA = 8
TOP_K = 2
N_LAYERS = 4
BATCH = 2
SEQ = 5


# ---------------------------------------------------------------------------
# Mock OLMoE-like block 和 model（避免依赖真实 transformers 模型）
# ---------------------------------------------------------------------------


class OlmoeSparseMoeBlock(nn.Module):
    """模拟 OLMoE 的 MoE block，名字必须和真实类名一致以触发 inject 识别。"""

    def __init__(self, d_model: int, n_experts: int, top_k: int = 8):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(
            [nn.Linear(d_model, d_model, bias=False) for _ in range(n_experts)]
        )
        self.top_k = top_k

    def forward(self, hidden_states):
        b, t, d = hidden_states.shape
        h_flat = hidden_states.reshape(-1, d)
        router_logits = self.gate(h_flat)  # [B*T, N_E]
        weights = torch.softmax(router_logits, dim=-1)
        topk_vals, topk_idx = weights.topk(self.top_k, dim=-1)

        out = torch.zeros_like(h_flat)
        for k in range(self.top_k):
            for i in range(len(self.experts)):
                mask = topk_idx[:, k] == i
                if mask.any():
                    out[mask] += topk_vals[mask, k : k + 1] * self.experts[i](h_flat[mask])
        return out.reshape(b, t, d), router_logits


class FakeLayer(nn.Module):
    def __init__(self, d_model, n_experts):
        super().__init__()
        self.mlp = OlmoeSparseMoeBlock(d_model, n_experts)


class FakeModel(nn.Module):
    """模拟 OlmoeForCausalLM 结构：model.model.layers[i].mlp"""

    class _Inner(nn.Module):
        def __init__(self, n_layers, d_model, n_experts):
            super().__init__()
            self.layers = nn.ModuleList([FakeLayer(d_model, n_experts) for _ in range(n_layers)])

    class _Cfg:
        def __init__(self, d_model, n_experts):
            self.hidden_size = d_model
            self.num_experts = n_experts
            self.name_or_path = "fake/olmoe"

    def __init__(self, n_layers=N_LAYERS, d_model=D_MODEL, n_experts=N_EXPERTS_MOE):
        super().__init__()
        self.model = self._Inner(n_layers, d_model, n_experts)
        self.config = self._Cfg(d_model, n_experts)

    def forward(self, x):
        h = x
        for layer in self.model.layers:
            out = layer.mlp(h)
            h = out[0] if isinstance(out, tuple) else out
        return h


class FakeFTArgs:
    """简化版 FinetuningArguments，字段必须与 moe_lora 用到的对齐"""

    def __init__(self, **overrides):
        defaults = dict(
            finetuning_type="moe_lora",
            moe_lora_n_experts=N_LORA,
            moe_lora_rank=RANK,
            moe_lora_alpha=ALPHA,
            moe_lora_top_k=TOP_K,
            moe_lora_pool_share="per_layer",
            moe_lora_w_share="per_layer",
            moe_lora_target_layers="all",
            moe_lora_detach_p_e=False,
        )
        defaults.update(overrides)
        for k, v in defaults.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# 组件单元测试
# ---------------------------------------------------------------------------


class TestRoutingProjection:
    def test_forward_shape(self):
        rp = RoutingProjection(N_EXPERTS_MOE, N_LORA)
        logits = torch.randn(BATCH, SEQ, N_EXPERTS_MOE)
        p_L = rp(logits)
        assert p_L.shape == (BATCH, SEQ, N_LORA)
        # softmax 后每行加和 ≈ 1
        torch.testing.assert_close(p_L.sum(dim=-1), torch.ones(BATCH, SEQ))

    def test_grad_flow(self):
        rp = RoutingProjection(N_EXPERTS_MOE, N_LORA)
        logits = torch.randn(BATCH, SEQ, N_EXPERTS_MOE, requires_grad=True)
        p_L = rp(logits)
        p_L.sum().backward()
        assert rp.proj.weight.grad is not None
        assert rp.proj.weight.grad.abs().sum() > 0

    def test_entropy_buffer(self):
        rp = RoutingProjection(N_EXPERTS_MOE, N_LORA)
        # 跑两次 forward，buffer 应该累加
        rp(torch.randn(BATCH, SEQ, N_EXPERTS_MOE))
        rp(torch.randn(BATCH, SEQ, N_EXPERTS_MOE))
        assert rp.entropy_count.item() == 2
        assert rp.entropy_sum.item() > 0


class TestLoRAExpert:
    def test_forward_shape(self):
        e = LoRAExpert(D_MODEL, RANK, ALPHA)
        h = torch.randn(BATCH, SEQ, D_MODEL)
        out = e(h)
        assert out.shape == h.shape

    def test_zero_init_b(self):
        """B 必须零初始化，保证训练初期 lora_output ≈ 0。"""
        e = LoRAExpert(D_MODEL, RANK, ALPHA)
        assert torch.allclose(e.B.weight, torch.zeros_like(e.B.weight))
        # 因此 forward 输出全 0
        h = torch.randn(BATCH, SEQ, D_MODEL)
        torch.testing.assert_close(e(h), torch.zeros_like(h))

    def test_scaling(self):
        e = LoRAExpert(D_MODEL, RANK, ALPHA)
        assert e.scaling == ALPHA / RANK


class TestLoRAPool:
    def test_forward_shape(self):
        pool = LoRAPool(N_LORA, D_MODEL, RANK, ALPHA, TOP_K)
        h = torch.randn(BATCH, SEQ, D_MODEL)
        p_L = torch.softmax(torch.randn(BATCH, SEQ, N_LORA), dim=-1)
        out = pool(h, p_L)
        assert out.shape == h.shape

    def test_zero_at_init(self):
        """所有 LoRAExpert.B 都是零，pool 输出也应该是零。"""
        pool = LoRAPool(N_LORA, D_MODEL, RANK, ALPHA, TOP_K)
        h = torch.randn(BATCH, SEQ, D_MODEL)
        p_L = torch.softmax(torch.randn(BATCH, SEQ, N_LORA), dim=-1)
        out = pool(h, p_L)
        torch.testing.assert_close(out, torch.zeros_like(out))

    def test_activation_buffer(self):
        pool = LoRAPool(N_LORA, D_MODEL, RANK, ALPHA, TOP_K)
        h = torch.randn(BATCH, SEQ, D_MODEL)
        p_L = torch.softmax(torch.randn(BATCH, SEQ, N_LORA), dim=-1)
        pool(h, p_L)
        # top-k 选了 BATCH*SEQ*TOP_K 次
        assert pool.activation_count.sum().item() == BATCH * SEQ * TOP_K
        assert pool.total_tokens.item() == BATCH * SEQ

    def test_grad_flow(self):
        pool = LoRAPool(N_LORA, D_MODEL, RANK, ALPHA, TOP_K)
        # 先扰动一下 B（不然全 0 没梯度）
        for e in pool.experts:
            nn.init.normal_(e.B.weight, std=0.01)
        h = torch.randn(BATCH, SEQ, D_MODEL, requires_grad=True)
        p_L = torch.softmax(torch.randn(BATCH, SEQ, N_LORA, requires_grad=True), dim=-1)
        out = pool(h, p_L)
        out.sum().backward()
        # 至少有一个 expert 的参数有梯度
        any_grad = any(
            (e.A.weight.grad is not None and e.A.weight.grad.abs().sum() > 0)
            for e in pool.experts
        )
        assert any_grad


# ---------------------------------------------------------------------------
# inject_moe_lora 测试
# ---------------------------------------------------------------------------


class TestInjectMoELoRA:
    def test_inject_per_layer(self):
        """V1: per_layer pool + per_layer W"""
        model = FakeModel()
        args = FakeFTArgs(moe_lora_pool_share="per_layer", moe_lora_w_share="per_layer")
        inject_moe_lora(model, args)

        # 每层都该挂上模块
        for layer in model.model.layers:
            assert isinstance(layer.mlp.routing_projection, RoutingProjection)
            assert isinstance(layer.mlp.lora_pool, LoRAPool)

        # 每个 pool / proj 应该是不同对象
        pools = [l.mlp.lora_pool for l in model.model.layers]
        projs = [l.mlp.routing_projection for l in model.model.layers]
        assert len(set(id(p) for p in pools)) == N_LAYERS
        assert len(set(id(p) for p in projs)) == N_LAYERS

    def test_inject_global_pool(self):
        """V2: 全局 pool + per-layer W"""
        model = FakeModel()
        args = FakeFTArgs(moe_lora_pool_share="global", moe_lora_w_share="per_layer")
        inject_moe_lora(model, args)

        # 所有层共享同一个 pool 对象
        pools = [l.mlp.lora_pool for l in model.model.layers]
        assert len(set(id(p) for p in pools)) == 1
        # W 仍然每层独立
        projs = [l.mlp.routing_projection for l in model.model.layers]
        assert len(set(id(p) for p in projs)) == N_LAYERS
        # global pool 应该挂在 model 顶层
        assert hasattr(model, "global_lora_pool")

    def test_inject_global_w(self):
        """V2': 全局 pool + 全局 W"""
        model = FakeModel()
        args = FakeFTArgs(moe_lora_pool_share="global", moe_lora_w_share="global")
        inject_moe_lora(model, args)

        # pool 和 W 都共享同一个对象
        pools = [l.mlp.lora_pool for l in model.model.layers]
        projs = [l.mlp.routing_projection for l in model.model.layers]
        assert len(set(id(p) for p in pools)) == 1
        assert len(set(id(p) for p in projs)) == 1

    def test_trainable_only_lora(self):
        model = FakeModel()
        args = FakeFTArgs()
        inject_moe_lora(model, args)

        for name, p in model.named_parameters():
            should_train = "routing_projection" in name or "lora_pool" in name
            assert p.requires_grad == should_train, f"{name} requires_grad mismatch"

    def test_trainable_ratio_reasonable(self):
        model = FakeModel()
        args = FakeFTArgs()
        inject_moe_lora(model, args)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        ratio = trainable / total
        assert 0.001 < ratio < 0.5, f"Trainable ratio {ratio} unexpected"

    def test_forward_after_inject_preserves_shape(self):
        """注入后 model forward 形状不变（且初期输出 ≈ base，因为 LoRA B 是 0）。"""
        model = FakeModel()
        x = torch.randn(BATCH, SEQ, D_MODEL)
        with torch.no_grad():
            base_out = model(x)

        args = FakeFTArgs()
        inject_moe_lora(model, args)

        with torch.no_grad():
            new_out = model(x)
        assert new_out.shape == base_out.shape
        # B 全零，LoRA 分支输出 0 → final 应该等于 base
        torch.testing.assert_close(new_out, base_out, rtol=1e-4, atol=1e-4)

    def test_forward_after_inject_grad_flow(self):
        """LoRA 参数能拿到非零梯度。"""
        model = FakeModel()
        args = FakeFTArgs()
        inject_moe_lora(model, args)

        # 扰动 LoRA expert B（否则全 0，loss 对 LoRA 无梯度）
        for layer in model.model.layers:
            for e in layer.mlp.lora_pool.experts:
                nn.init.normal_(e.B.weight, std=0.01)

        x = torch.randn(BATCH, SEQ, D_MODEL)
        out = model(x)
        out.sum().backward()

        # routing_projection 的 weight 必须有梯度
        for layer in model.model.layers:
            grad = layer.mlp.routing_projection.proj.weight.grad
            assert grad is not None and grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Save / Load round-trip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        model = FakeModel()
        args = FakeFTArgs()
        inject_moe_lora(model, args)

        # 训练几步：扰动一下 LoRA 参数
        for layer in model.model.layers:
            for e in layer.mlp.lora_pool.experts:
                nn.init.normal_(e.A.weight, std=0.05)
                nn.init.normal_(e.B.weight, std=0.05)
            nn.init.normal_(layer.mlp.routing_projection.proj.weight, std=0.05)

        save_moe_lora_state(model, str(tmp_path), args)
        assert (tmp_path / "moe_lora_state.safetensors").exists()
        assert (tmp_path / "moe_lora_config.json").exists()

        # 在新 model 上加载
        new_model = FakeModel()
        load_moe_lora_state(new_model, str(tmp_path))

        # trainable 参数应该一致
        old_state = {n: p for n, p in model.named_parameters() if p.requires_grad}
        new_state = {n: p for n, p in new_model.named_parameters() if p.requires_grad}
        assert set(old_state.keys()) == set(new_state.keys())
        for k in old_state:
            torch.testing.assert_close(old_state[k], new_state[k])

    def test_state_dict_no_duplicate_global_pool(self, tmp_path):
        """全局共享时，state_dict 里 pool 只出现一次（验证 add_module 起作用）。"""
        model = FakeModel()
        args = FakeFTArgs(moe_lora_pool_share="global", moe_lora_w_share="global")
        inject_moe_lora(model, args)

        trainable_keys = [n for n, p in model.named_parameters() if p.requires_grad]
        # global pool 的某个 expert A 权重应该只出现一次
        a_weights = [k for k in trainable_keys if "lora_pool.experts.0.A" in k]
        # 期望只有 model.global_lora_pool.experts.0.A.weight 一个键
        assert len(a_weights) == 1, f"Expected 1 occurrence, got {len(a_weights)}: {a_weights}"


# ---------------------------------------------------------------------------
# Callbacks（仅 smoke test）
# ---------------------------------------------------------------------------


class TestCallbacks:
    def test_stats_callback_writes_logs(self):
        model = FakeModel()
        args = FakeFTArgs()
        inject_moe_lora(model, args)

        # 跑一次 forward 触发 buffer 累加
        x = torch.randn(BATCH, SEQ, D_MODEL)
        model(x)

        cb = MoELoRAStatsCallback()
        logs: dict = {}
        cb.on_log(args=None, state=None, control=None, model=model, logs=logs)

        # 应该有这些指标
        assert any("activation_max" in k for k in logs)
        assert any("W_fro_norm" in k for k in logs)
        assert any("p_L_entropy" in k for k in logs)
        assert "moe_lora/W_fro_norm_avg" in logs
        assert "moe_lora/p_L_entropy_avg" in logs

    def test_stats_callback_resets_buffers(self):
        model = FakeModel()
        args = FakeFTArgs()
        inject_moe_lora(model, args)

        x = torch.randn(BATCH, SEQ, D_MODEL)
        model(x)

        cb = MoELoRAStatsCallback()
        cb.on_log(args=None, state=None, control=None, model=model, logs={})

        # 调用后 buffer 应该被重置
        for layer in model.model.layers:
            assert layer.mlp.lora_pool.total_tokens.item() == 0
            assert layer.mlp.routing_projection.entropy_count.item() == 0

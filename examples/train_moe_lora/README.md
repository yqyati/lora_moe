# MoE-LoRA: Routing Compression

在 MoE Transformer 主干上引入 LoRA 分支，**LoRA 的 routing 不独立学习**，而是直接复用 MoE router 的输出投影压缩而来：

```
p_L = softmax(W · router_logits),   W: N_E -> N_L
```

详细方案见 `lora_moe/工程实现大纲.md`。

---

## 快速开始（V1 + OLMoE-1B-7B + GSM8K）

```bash
# 1. 数据预处理：下载 oumi-ai/MetaMathQA-R1, 删 <think>...</think>
python data/preprocess_metamathqa_r1.py --limit 1000   # 先小规模 smoke test
python data/preprocess_metamathqa_r1.py                 # 全量 ~395K

# 2. 训练 V1
llamafactory-cli train examples/train_moe_lora/v1_olmoe.yaml

# 3. 评估 GSM8K（生成式）
python eval_scripts/eval_gsm8k.py \
    --base_model allenai/OLMoE-1B-7B-0924 \
    --adapter_path saves/olmoe/moe_lora/v1_per_layer/checkpoint-500 \
    --batch_size 4

# 多卡加速版（8 张 GPU 并行 + 大 batch + 减 max_new_tokens，速度快 ~30-50 倍）
torchrun --nproc_per_node=8 eval_scripts/eval_gsm8k.py \
    --base_model allenai/OLMoE-1B-7B-0924 \
    --batch_size 64 \
    --max_new_tokens 512

torchrun --nproc_per_node=8 eval_scripts/eval_gsm8k.py \
    --base_model allenai/OLMoE-1B-7B-0924 \
    --adapter_path saves/olmoe/moe_lora/v2_global_unlinear_1 \
    --batch_size 64 \
    --max_new_tokens 512

# 4. 评估 MMLU（选择题，走 LlamaFactory 自带 eval）
llamafactory-cli eval \
    --model_name_or_path allenai/OLMoE-1B-7B-0924 \
    --adapter_name_or_path saves/olmoe/moe_lora/v1_per_layer \
    --finetuning_type moe_lora \
    --task mmlu --lang en --n_shot 5 \
    --save_dir ./eval_results/v1_mmlu
```

---

## 5 种架构变体

| 变体 | 池结构 | 池大小 N_L | rank | 配置文件 |
|------|--------|----------|------|---------|
| **V1** | 每层独享 | 8 | 8 | `v1_olmoe.yaml` |
| **V2** | 全局共享 + W_l 独立 | 512 | 4 | `v2_olmoe.yaml` ⭐主推荐 |
| **V2'** | 全局共享 + W 也共享 | 512 | 4 | `v2prime_olmoe.yaml`（消融对照）|
| **V3** | 全局共享 + 高 rank | 64 | 32 | `v3_olmoe.yaml` |
| **V4** | 每层独享 + 高 rank 小池 | 4 | 16 | `v4_olmoe.yaml` |
| **VBase** | 每层独享 + follow_moe 路由 | 64（=OLMoE 原 expert 数）| 2 | `vbase_olmoe.yaml`（对照组：路由直接复用 OLMoE）|

所有变体的可训练参数量对齐到 V1 的 ±10% 范围内（公平对比）。

---

## 训练监控（wandb）

启用 `--report_to wandb` 后，下面这些指标会自动 log：

| 指标 | 异常信号 |
|------|---------|
| `moe_lora/.../activation_max` `_min` | 偏差大 → 路由不均衡 |
| `moe_lora/.../dead_experts` | > 0 → 有 expert 没被用过 |
| `moe_lora/.../activation_imbalance` | > 5× → 严重塌缩 |
| `moe_lora/.../W_fro_norm` | 爆炸 / 单调降到 0 |
| `moe_lora/.../p_L_entropy` | < 0.1 塌缩；接近 log(N_L) 没分化 |

启动日志会打印 trainable / total 参数比例，类似：
```
MoE-LoRA setup complete | trainable: 4.20M (0.299%) | total: 1404.32M
```

---

## 关键超参速查

| 参数 | 含义 | V1/V2/V3/V4 默认 |
|------|------|----|
| `moe_lora_n_experts` | LoRA pool 大小 N_L | 8 / 512 / 64 / 4 |
| `moe_lora_rank` | 每个 LoRA expert 的 rank | 8 / 4 / 32 / 16 |
| `moe_lora_alpha` | scaling = alpha / rank（建议 alpha = 2*rank） | 16 / 8 / 64 / 32 |
| `moe_lora_top_k` | LoRA top-k 激活 | 2 |
| `moe_lora_pool_share` | `per_layer` / `global` | per_layer / global |
| `moe_lora_w_share` | 投影矩阵 W 的共享方式 | per_layer |
| `moe_lora_target_layers` | `all` / `first_half` / `last_half` / `last_third` / `0,5,10` | all |
| `moe_lora_detach_p_e` | 是否切断 LoRA 到 router 的梯度（base model 全 freeze 时无实际效果） | false |

---

## 评估脚本

| 脚本 | benchmark | 评估方式 |
|------|-----------|---------|
| `eval_scripts/eval_gsm8k.py` | GSM8K | 生成 + `#### 数字` 抽取 |
| `eval_scripts/eval_math500.py` | MATH-500 | 生成 + `\boxed{...}` 抽取 |
| `eval_scripts/eval_humaneval.py` | HumanEval | 生成 + 执行 unit test（需 `pip install human-eval`）|

通用参数：
```bash
--base_model PATH        # HF base model id
--adapter_path PATH      # moe_lora checkpoint 目录（不传则评估 base）
--batch_size N           # 生成 batch
--limit N                # 只评估前 N 条（debug）
--save_path FILE.jsonl   # 保存 per-sample prediction
```

---

## checkpoint 文件布局

训练后 `output_dir` 里会有：
```
saves/olmoe/moe_lora/v1_per_layer/
├── moe_lora_state.safetensors    # 仅 trainable 参数（~8 MB for V1）
├── moe_lora_config.json          # 重建结构所需的配置
├── checkpoint-500/               # 中间 checkpoint（也包含上面两个文件）
├── checkpoint-1000/
└── ...
```

base model 不会保存（节省空间，加载时从 HF hub 拉）。

---

## 已支持的 MoE 模型

第一阶段验证：
- ✅ OLMoE-1B-7B（`OlmoeSparseMoeBlock`）

代码层面也识别但未充分测试：
- 🚧 Qwen3-MoE（`Qwen3MoeSparseMoeBlock`）
- 🚧 DeepSeek-V2-Lite（`DeepseekV2MoE`）

要扩展到其他 MoE 模型，把类名加到 `src/llamafactory/model/model_utils/moe_lora.py` 的 `SUPPORTED_MOE_BLOCK_NAMES`，并确认 router forward 返回 `(hidden_states, router_logits)` 二元组。

---

## 文件清单

| 文件 | 作用 |
|------|------|
| `src/llamafactory/model/model_utils/moe_lora.py` | 核心实现（组件 + 注入 + 保存 + callback） |
| `src/llamafactory/hparams/finetuning_args.py` | 8 个 `moe_lora_*` 超参 |
| `src/llamafactory/model/adapter.py` | `_setup_moe_lora` + `init_adapter` 分支 |
| `src/llamafactory/train/sft/workflow.py` | 注册 callback + 训练后保存 |
| `src/llamafactory/eval/evaluator.py` | 评估时加载 moe_lora checkpoint |
| `src/llamafactory/chat/hf_engine.py` | chat 时加载 moe_lora checkpoint |
| `data/preprocess_metamathqa_r1.py` | 数据清洗（删 `<think>`） |
| `data/dataset_info.json` | 注册 `metamathqa_r1` |
| `examples/train_moe_lora/*.yaml` | 5 个变体训练配置 |
| `eval_scripts/*.py` | 3 个生成式评估脚本 |
| `tests/model/test_moe_lora.py` | unit test |

---

## 常见问题

**Q: 训练启动报 `assert ratio < 0.05`？**  
A: trainable 参数比例超过 5%，通常是 freeze 没生效。检查 `_setup_moe_lora` 调用顺序。

**Q: wandb 看到 `dead_experts > 0` 怎么办？**  
A: 某些 LoRA expert 从未被激活。考虑减小 `moe_lora_n_experts`，或加大 learning rate 让 router 更新更激进。

**Q: 加载 checkpoint 时报 "未加载的 LoRA 参数"？**  
A: `inject_moe_lora` 的命名规则可能改了。检查 `moe_lora.py` 里 `routing_projection` / `lora_pool` 的属性名。

**Q: 训练完想 merge 到 base model 吗？**  
A: MoE-LoRA 是独立分支（不是叠加在 linear 上的 LoRA），无法 merge。inference 时必须保留 LoRA pool + RoutingProjection。

**Q: 评估生成很慢？**  
A: `eval_scripts/*` 是单卡顺序生成，没接 vLLM。第二/三阶段大量评估时建议接 vLLM 或 sglang。

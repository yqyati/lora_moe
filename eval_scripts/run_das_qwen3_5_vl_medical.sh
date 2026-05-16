#!/bin/bash
# Compute DAS (Domain Advantage Score) for Qwen3.5-VL on LLaVA-Med medical domain.
# Output: eval_results/das_qwen3_5_vl_medical.json
#
# Domain  = LLaVA-Med 医学多模态 SFT (text only, 已 strip <image>)
# General = MMLU (all subjects, test split)
# Top-K   = 4 specialized experts per layer
#
# Run: bash eval_scripts/run_das_qwen3_5_vl_medical.sh

set -e
cd "$(dirname "$0")/.."

# 用 hf-mirror 中国镜像下 MMLU
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

# Qwen3.5-VL 35B 在 8 GPU 上 device_map=auto 自动 shard
/root/miniconda3/envs/qwen35/bin/python eval_scripts/compute_das.py \
    --base_model /data/android/yqy/work/lora_moe/model/Qwen3.5-35B-A3B \
    --domain_dataset /data/android/yqy/work/lora_moe/data/medical_train/llava_med/llava_med_train_filtered.json \
    --general_dataset cais/mmlu \
    --general_config all \
    --general_split test \
    --top_k 4 \
    --max_samples 200 \
    --max_seq_len 512 \
    --output eval_results/das_qwen3_5_vl_medical.json

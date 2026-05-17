#!/bin/bash
# Compute DAS (Domain Advantage Score) for Qwen3.5-VL on medical_mixed domain.
# Output: eval_results/das_qwen3_5_vl_medical_mixed_top16.json
#
# Domain  = medical_mixed (llava_med 47k + medical_vqa_combined 26k)
# General = MMLU (all subjects, test split)
# Top-K   = 16

set -e
cd "$(dirname "$0")/.."

# 用 hf-mirror 中国镜像下 MMLU
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

# Qwen3.5-VL 35B 在 8 GPU 上 device_map=auto 自动 shard
/root/miniconda3/envs/qwen35/bin/python eval_scripts/compute_das.py \
    --base_model /data/android/yqy/work/lora_moe/model/Qwen3.5-35B-A3B \
    --domain_dataset /data/android/yqy/work/lora_moe/data/medical_train/medical_mixed/train.json \
    --general_dataset cais/mmlu \
    --general_config all \
    --general_split test \
    --top_k 16 \
    --max_samples 200 \
    --max_seq_len 512 \
    --output eval_results/das_qwen3_5_vl_medical_mixed_top16.json

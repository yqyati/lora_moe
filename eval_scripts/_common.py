"""通用工具：加载 base model + moe_lora checkpoint，给生成式评估脚本复用。"""

import argparse
import re
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 把仓库 src/ 加到 PYTHONPATH，以便 import llamafactory
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from llamafactory.model.model_utils.moe_lora import load_moe_lora_state  # noqa: E402


def common_arg_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--base_model", required=True, help="HF base model id, e.g. allenai/OLMoE-1B-7B-0924")
    p.add_argument("--adapter_path", default=None, help="moe_lora checkpoint dir; omit to evaluate base model only")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--limit", type=int, default=None, help="只评估前 N 个样本（debug 用）")
    p.add_argument("--batch_size", type=int, default=1, help="生成 batch size")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--save_path", default=None, help="把每个样本的 prediction 保存到 jsonl")
    return p


def load(args) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    if args.adapter_path:
        print(f"Loading MoE-LoRA adapter from {args.adapter_path}")
        model = load_moe_lora_state(model, args.adapter_path)
    else:
        print("No --adapter_path provided, evaluating BASE model")
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def generate_batch(model, tokenizer, prompts, max_new_tokens):
    """Batch 生成。返回纯 completion（不含 prompt）。"""
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    completions = []
    for i, ids in enumerate(out):
        prompt_len = enc.input_ids[i].ne(tokenizer.pad_token_id).sum().item()
        completions.append(tokenizer.decode(ids[prompt_len:], skip_special_tokens=True))
    return completions


def extract_gsm8k_answer(text: str):
    """GSM8K 标准答案抽取：优先 #### ，否则取最后一个数字。"""
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return m.group(1).strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None

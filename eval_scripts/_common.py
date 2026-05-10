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
    # decoder-only 模型 batch generate 必须 left-padding，否则模型会从 PAD 之后续写，输出全废
    tokenizer.padding_side = "left"

    # 多卡 (torchrun)：每个 rank 加载一份模型到自己的 GPU，避免 device_map="auto"
    # 把 shared moe_lora 模块和不同 layer 分到不同卡上引发跨设备调用。
    # 单卡：用 device_map="auto" 让 transformers 自己处理。
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        model.to(f"cuda:{local_rank}")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    if args.adapter_path:
        # 自动识别 adapter 类型:
        #   有 moe_lora_state.safetensors → MoE-LoRA(自家实现)
        #   有 adapter_config.json        → 标准 LoRA(PEFT)
        moe_state = os.path.join(args.adapter_path, "moe_lora_state.safetensors")
        peft_config = os.path.join(args.adapter_path, "adapter_config.json")
        if os.path.exists(moe_state):
            print(f"Loading MoE-LoRA adapter from {args.adapter_path}")
            model = load_moe_lora_state(model, args.adapter_path)
        elif os.path.exists(peft_config):
            print(f"Loading PEFT (standard) LoRA adapter from {args.adapter_path}")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter_path, torch_dtype=dtype)
            model = model.merge_and_unload()  # 合并 LoRA 到 base 权重,推理更快
        else:
            raise FileNotFoundError(
                f"{args.adapter_path} 既无 moe_lora_state.safetensors 也无 adapter_config.json,"
                "无法识别 adapter 类型"
            )
    else:
        print("No --adapter_path provided, evaluating BASE model")
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def generate_batch(model, tokenizer, prompts, max_new_tokens, temperature=0.0, top_p=0.95):
    """Batch 生成。返回纯 completion（不含 prompt）。

    temperature=0.0 时用 greedy；>0 时用 sampling。
    """
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
    gen_kwargs = dict(
        **enc,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kwargs.update(do_sample=False)
    out = model.generate(**gen_kwargs)
    # left-padding 下,真正的生成部分从 padded 总长之后开始;
    # 用"非 pad token 数"当起点会把 prompt 尾部泄露到 completion 里。
    input_len = enc.input_ids.shape[1]
    completions = []
    for ids in out:
        completions.append(tokenizer.decode(ids[input_len:], skip_special_tokens=True))
    return completions


def wrap_default_chat(tokenizer, user_content: str) -> str:
    """对齐 LlamaFactory `template: default` 训练格式:
        Human: {content}<EOS>
        Assistant:
    生成端不带 trailing EOS,模型从 'Assistant:' 后开始续写。
    """
    eos = tokenizer.eos_token or "<|endoftext|>"
    return f"Human: {user_content}{eos}\nAssistant:"


def extract_gsm8k_answer(text: str):
    """GSM8K 标准答案抽取(强化版):
    - 优先 #### 后的数字
    - 处理千分位逗号 ('1,234' → '1234')
    - 整数化浮点小数 ('5.0' → '5', '5.00' → '5')
    - fallback 取最后一个数字
    """
    # 1. 优先匹配 #### 后的数字(允许逗号)
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return _normalize_num(m.group(1))
    # 2. fallback: 取最后一个数字
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return _normalize_num(nums[-1]) if nums else None


def _normalize_num(s: str):
    """规范化数字字符串: 去千分位逗号,整数化浮点(5.0 → 5)。"""
    if s is None:
        return None
    s = s.replace(",", "").strip()
    # 末尾纯零的浮点 → 整数
    if "." in s:
        try:
            f = float(s)
            if f == int(f):
                return str(int(f))
            return s
        except ValueError:
            return s
    return s

"""Diagnose whether multimodal training is actually feeding pixel_values into the model.

Mimics LlamaFactory's training data pipeline for one batch and prints what tensors
end up in the collated batch.

Run: cd LlamaFactory && python eval_scripts/diag_mm_batch.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from transformers import HfArgumentParser

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.hparams import (
    DataArguments, FinetuningArguments, GeneratingArguments,
    ModelArguments, TrainingArguments,
)
from llamafactory.model import load_tokenizer

YAML = "examples/train_moe_lora_qwen3vl_medical/baseline2_moelora_qwen3vl_medical.yaml"


def main():
    import yaml
    with open(YAML) as f:
        cfg = yaml.safe_load(f)
    cfg["max_samples"] = 5  # 只取 5 条
    cfg["dataloader_num_workers"] = 0
    cfg["preprocessing_num_workers"] = 1
    cfg.pop("deepspeed", None)  # 不要起 ZeRO

    parser = HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments,
        FinetuningArguments, GeneratingArguments,
    ))
    model_args, data_args, training_args, ft_args, gen_args = (
        parser.parse_dict(cfg, allow_extra_keys=True)
    )

    tok_module = load_tokenizer(model_args)
    tokenizer = tok_module["tokenizer"]
    processor = tok_module.get("processor")
    print(f"[load] tokenizer={type(tokenizer).__name__}")
    print(f"[load] processor={type(processor).__name__ if processor else 'NONE ❌'}")

    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    print(f"[load] template={template.__class__.__name__}, mm_plugin={type(template.mm_plugin).__name__}")

    ds_module = get_dataset(template, model_args, data_args, training_args,
                             stage="sft", **tok_module)
    train_ds = ds_module["train_dataset"]
    print(f"[ds] train_ds size={len(train_ds)}")
    print(f"[ds] columns={list(train_ds.column_names) if hasattr(train_ds, 'column_names') else '?'}")

    # 看一条样本
    sample = train_ds[0]
    print("\n=== first sample keys ===")
    for k, v in sample.items():
        if isinstance(v, list):
            print(f"  {k}: list[{type(v[0]).__name__ if v else '?'}], len={len(v)}")
        elif hasattr(v, "shape"):
            print(f"  {k}: tensor shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            s = repr(v)
            print(f"  {k}: {type(v).__name__} {s[:80]}{'...' if len(s) > 80 else ''}")

    # 走 collator 拿一个 batch
    from llamafactory.data import SFTDataCollatorWith4DAttentionMask
    collator = SFTDataCollatorWith4DAttentionMask(
        template=template, model=None, label_pad_token_id=-100, **tok_module,
    )
    batch = collator([train_ds[i] for i in range(min(2, len(train_ds)))])
    print("\n=== collated batch keys ===")
    has_pixel = False
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: tensor shape={tuple(v.shape)} dtype={v.dtype}")
            if "pixel" in k or "image" in k:
                has_pixel = True
        else:
            print(f"  {k}: {type(v).__name__}")

    print()
    if has_pixel:
        print("✅ pixel_values IS in the batch — images do get to the model")
    else:
        print("❌ NO pixel_values in batch — images are being IGNORED")


if __name__ == "__main__":
    main()

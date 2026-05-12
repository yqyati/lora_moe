"""通用能力评测脚本（选择题，likelihood-based）。

评测 5 个 benchmark：CommonsenseQA, ARC-Challenge, StrategyQA, CEval, MMLU
用于检验领域微调后的灾难性遗忘程度。

用法（单 benchmark）:
    python eval_scripts/eval_general.py \
        --base_model allenai/OLMoE-1B-7B-0924 \
        --adapter_path saves/olmoe/moe_lora/v2_global_unlinear_1 \
        --benchmark arc_challenge \
        --batch_size 16

用法（全部 benchmark 一次跑完）:
    python eval_scripts/eval_general.py \
        --base_model allenai/OLMoE-1B-7B-0924 \
        --adapter_path saves/olmoe/moe_lora/v2_global_unlinear_1 \
        --benchmark all \
        --batch_size 16

多卡加速:
    torchrun --nproc_per_node=8 eval_scripts/eval_general.py \
        --base_model allenai/OLMoE-1B-7B-0924 \
        --adapter_path saves/olmoe/moe_lora/v2_global_unlinear_1 \
        --benchmark all \
        --batch_size 32
"""

import json
import os
import sys
import random

import torch
import torch.distributed as dist
from torch.nn.functional import log_softmax
from datasets import load_dataset
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from _common import common_arg_parser, load, wrap_for_model


# ─── 数据集加载与格式化 ────────────────────────────────────────────────────────

BENCHMARKS = ["commonsenseqa", "arc_challenge", "strategyqa", "hellaswag", "winogrande", "ceval", "mmlu"]


def load_commonsenseqa(limit=None):
    """CommonsenseQA: 5 选 1 (A-E)"""
    ds = load_dataset("tau/commonsense_qa", split="validation")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    samples = []
    for item in ds:
        question = item["question"]
        choices = item["choices"]["text"]
        labels = item["choices"]["label"]
        gold = item["answerKey"]
        options = [f"{l}. {t}" for l, t in zip(labels, choices)]
        samples.append({
            "question": question,
            "options": options,
            "labels": labels,
            "choices_text": choices,
            "gold": gold,
        })
    return samples


def load_arc_challenge(limit=None):
    """ARC-Challenge: 4 选 1 (A-D), 有时 (1-4)"""
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    samples = []
    for item in ds:
        question = item["question"]
        choices = item["choices"]["text"]
        labels = item["choices"]["label"]
        gold = item["answerKey"]
        options = [f"{l}. {t}" for l, t in zip(labels, choices)]
        samples.append({
            "question": question,
            "options": options,
            "labels": labels,
            "choices_text": choices,
            "gold": gold,
        })
    return samples


def load_strategyqa(limit=None):
    """StrategyQA: 二选一 (Yes/No)"""
    ds = load_dataset("ChilleD/StrategyQA", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    samples = []
    for item in ds:
        question = item["question"]
        gold = "A" if item["answer"] else "B"
        samples.append({
            "question": question,
            "options": ["A. Yes", "B. No"],
            "labels": ["A", "B"],
            "choices_text": ["Yes", "No"],
            "gold": gold,
        })
    return samples


def load_ceval(limit=None):
    """CEval: 中文 4 选 1 (A-D), 取 val 集全部科目"""
    subjects = [
        "computer_network", "operating_system", "computer_architecture",
        "college_programming", "college_physics", "college_chemistry",
        "advanced_mathematics", "probability_and_statistics", "discrete_mathematics",
        "electrical_engineer", "metrology_engineer", "high_school_mathematics",
        "high_school_physics", "high_school_chemistry", "high_school_biology",
        "middle_school_mathematics", "middle_school_physics", "middle_school_chemistry",
        "middle_school_biology", "high_school_history", "high_school_geography",
        "high_school_politics", "college_economics", "business_administration",
        "marxism", "mao_zedong_thought", "education_science", "teacher_qualification",
        "high_school_chinese", "logic", "law", "chinese_language_and_literature",
        "art_studies", "professional_tour_guide", "legal_professional",
        "modern_chinese_history", "ideological_and_moral_cultivation",
        "basic_medicine", "clinical_medicine", "urban_and_rural_planner",
        "accountant", "fire_engineer", "environmental_impact_assessment_engineer",
        "tax_accountant", "physician", "veterinary_medicine",
    ]
    samples = []
    for subj in subjects:
        try:
            ds = load_dataset("ceval/ceval-exam", subj, split="val", trust_remote_code=True)
        except Exception:
            continue
        for item in ds:
            question = item["question"]
            choices_text = [item["A"], item["B"], item["C"], item["D"]]
            labels = ["A", "B", "C", "D"]
            gold = item["answer"]
            options = [f"{l}. {t}" for l, t in zip(labels, choices_text)]
            samples.append({
                "question": question,
                "options": options,
                "labels": labels,
                "choices_text": choices_text,
                "gold": gold,
                "subject": subj,
            })
    random.seed(42)
    random.shuffle(samples)
    if limit:
        samples = samples[:limit]
    return samples


def load_mmlu(limit=None):
    """MMLU: 4 选 1 (A-D), 取 test 集"""
    ds = load_dataset("cais/mmlu", "all", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    samples = []
    label_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    for item in ds:
        question = item["question"]
        choices_text = item["choices"]
        labels = ["A", "B", "C", "D"]
        gold = label_map[item["answer"]]
        options = [f"{l}. {t}" for l, t in zip(labels, choices_text)]
        samples.append({
            "question": question,
            "options": options,
            "labels": labels,
            "choices_text": choices_text,
            "gold": gold,
            "subject": item.get("subject", "unknown"),
        })
    return samples


def load_hellaswag(limit=None):
    """HellaSwag: 4 选 1 句子补全（常识推理）"""
    ds = load_dataset("Rowan/hellaswag", split="validation")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    samples = []
    for item in ds:
        ctx = item["ctx"]
        endings = item["endings"]
        gold_idx = int(item["label"])
        labels = ["A", "B", "C", "D"]
        options = [f"{l}. {t}" for l, t in zip(labels, endings)]
        samples.append({
            "question": ctx,
            "options": options,
            "labels": labels,
            "choices_text": endings,
            "gold": labels[gold_idx],
        })
    return samples


def load_winogrande(limit=None):
    """WinoGrande: 2 选 1 代词消歧"""
    ds = load_dataset("allenai/winogrande", "winogrande_debiased", split="validation")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    samples = []
    for item in ds:
        sentence = item["sentence"]
        option1 = item["option1"]
        option2 = item["option2"]
        gold = "A" if item["answer"] == "1" else "B"
        labels = ["A", "B"]
        options = [f"A. {option1}", f"B. {option2}"]
        samples.append({
            "question": sentence,
            "options": options,
            "labels": labels,
            "choices_text": [option1, option2],
            "gold": gold,
        })
    return samples


def load_boolq(limit=None):
    """BoolQ: 是非问答（True/False）"""
    ds = load_dataset("google/boolq", split="validation")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    samples = []
    for item in ds:
        question = item["question"] + f"\nPassage: {item['passage']}"
        gold = "A" if item["answer"] else "B"
        samples.append({
            "question": question,
            "options": ["A. True", "B. False"],
            "labels": ["A", "B"],
            "choices_text": ["True", "False"],
            "gold": gold,
        })
    return samples


def load_piqa(limit=None):
    """PIQA: 物理直觉 2 选 1"""
    ds = load_dataset("piqa", split="validation", trust_remote_code=True)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    samples = []
    for item in ds:
        question = item["goal"]
        sol1 = item["sol1"]
        sol2 = item["sol2"]
        gold = "A" if item["label"] == 0 else "B"
        samples.append({
            "question": question,
            "options": [f"A. {sol1}", f"B. {sol2}"],
            "labels": ["A", "B"],
            "choices_text": [sol1, sol2],
            "gold": gold,
        })
    return samples


LOADER_MAP = {
    "commonsenseqa": load_commonsenseqa,
    "arc_challenge": load_arc_challenge,
    "strategyqa": load_strategyqa,
    "hellaswag": load_hellaswag,
    "winogrande": load_winogrande,
    "ceval": load_ceval,
    "mmlu": load_mmlu,
}


# ─── Prompt 模板 ──────────────────────────────────────────────────────────────

def format_prompt(sample, n_shot_examples=None):
    """格式化为标准选择题 prompt。"""
    parts = []
    if n_shot_examples:
        for ex in n_shot_examples:
            parts.append(format_single_qa(ex, include_answer=True))
        parts.append("")
    parts.append(format_single_qa(sample, include_answer=False))
    return "\n".join(parts)


def format_single_qa(sample, include_answer=False):
    """单条 QA 格式化。"""
    lines = [f"Question: {sample['question']}"]
    for opt in sample["options"]:
        lines.append(f"  {opt}")
    if include_answer:
        lines.append(f"Answer: {sample['gold']}")
    else:
        lines.append("Answer:")
    return "\n".join(lines)


# ─── Likelihood-based 评测核心 ─────────────────────────────────────────────────

@torch.inference_mode()
def compute_choice_logprobs(model, tokenizer, prompt, choices):
    """计算每个选项 token 的条件 log-probability。

    对于选择题，我们比较 "Answer: A", "Answer: B", ... 各选项的概率。
    只看选项 token 的 log-prob（不含 prompt 部分）。
    """
    logprobs = []
    for choice in choices:
        full_text = prompt + " " + choice
        enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = enc.input_ids.to(model.device)

        outputs = model(input_ids)
        logits = outputs.logits  # (1, seq_len, vocab_size)

        # prompt 部分长度
        prompt_enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        prompt_len = prompt_enc.input_ids.shape[1]

        # 计算 choice 部分每个 token 的 log-prob
        log_probs = log_softmax(logits[0], dim=-1)
        choice_logprob = 0.0
        choice_len = input_ids.shape[1] - prompt_len
        for i in range(prompt_len, input_ids.shape[1]):
            token_id = input_ids[0, i]
            choice_logprob += log_probs[i - 1, token_id].item()

        # 长度归一化
        if choice_len > 0:
            choice_logprob /= choice_len

        logprobs.append(choice_logprob)

    return logprobs


@torch.inference_mode()
def compute_choice_logprobs_batch(model, tokenizer, prompts_and_choices):
    """批量版本：每个 sample 有多个 choice，逐条算 logprob。

    prompts_and_choices: list of (prompt, choices_list)
    returns: list of logprobs_list (same shape)
    """
    results = []
    for prompt, choices in prompts_and_choices:
        logprobs = compute_choice_logprobs(model, tokenizer, prompt, choices)
        results.append(logprobs)
    return results


# ─── 主流程 ────────────────────────────────────────────────────────────────────

def evaluate_benchmark(model, tokenizer, benchmark_name, samples, args, is_main, local_rank, world_size, is_dist):
    """评测单个 benchmark，返回 (accuracy, records)。"""

    # 分配给当前 rank 的样本
    my_indices = list(range(local_rank, len(samples), world_size)) if is_dist else list(range(len(samples)))

    my_records = []
    batch_starts = list(range(0, len(my_indices), args.batch_size))
    iterator = tqdm(batch_starts, desc=f"{benchmark_name} rank{local_rank}") if is_main else batch_starts

    for i in iterator:
        batch_idx = my_indices[i : i + args.batch_size]
        for idx in batch_idx:
            sample = samples[idx]
            prompt = format_prompt(sample)
            # Qwen3: 套 chat template + 关闭 thinking; OLMoE: 原样返回 plain prompt。
            prompt = wrap_for_model(tokenizer, prompt, args.base_model)
            choices = sample["labels"]

            logprobs = compute_choice_logprobs(model, tokenizer, prompt, choices)
            pred_idx = logprobs.index(max(logprobs))
            pred = choices[pred_idx]
            ok = pred == sample["gold"]

            my_records.append({
                "idx": idx,
                "question": sample["question"],
                "gold": sample["gold"],
                "pred": pred,
                "logprobs": {c: lp for c, lp in zip(choices, logprobs)},
                "correct": ok,
            })

    # gather
    if is_dist:
        torch.cuda.empty_cache()  # 释放 likelihood 计算留下的 KV cache,给 NCCL 留 buffer
        gathered = [None] * world_size
        dist.all_gather_object(gathered, my_records)
        records = [r for sub in gathered for r in sub]
    else:
        records = my_records

    records.sort(key=lambda r: r["idx"])

    if is_main:
        correct = sum(int(r["correct"]) for r in records)
        acc = correct / len(records) if records else 0.0
        print(f"  {benchmark_name}: {acc:.4f} ({correct}/{len(records)})")
        return acc, records
    return None, records


def main():
    parser = common_arg_parser("通用能力评测：检验领域微调后的灾难性遗忘。")
    parser.add_argument(
        "--benchmark", type=str, default="all",
        help="评测哪个 benchmark: commonsenseqa, arc_challenge, strategyqa, ceval, mmlu, all"
    )
    parser.add_argument(
        "--n_shot", type=int, default=0,
        help="few-shot 数量（从训练集 / 额外样本中取）。MMLU 标准为 5-shot，其余通常 0-shot。"
    )
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_dist = world_size > 1
    is_main = local_rank == 0

    if is_dist:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    tokenizer, model = load(args)

    # 确定要评测的 benchmark 列表
    if args.benchmark == "all":
        benchmarks_to_eval = BENCHMARKS
    else:
        benchmarks_to_eval = [b.strip() for b in args.benchmark.split(",")]
        for b in benchmarks_to_eval:
            if b not in LOADER_MAP:
                raise ValueError(f"Unknown benchmark: {b}. Choose from: {BENCHMARKS}")

    results_summary = {}

    for bench_name in benchmarks_to_eval:
        if is_main:
            print(f"\n{'='*60}")
            print(f"Evaluating: {bench_name}")
            print(f"{'='*60}")

        loader = LOADER_MAP[bench_name]
        samples = loader(limit=args.limit)

        if is_main:
            print(f"  Loaded {len(samples)} samples")

        acc, records = evaluate_benchmark(
            model, tokenizer, bench_name, samples, args,
            is_main, local_rank, world_size, is_dist
        )

        if is_main and acc is not None:
            results_summary[bench_name] = acc

            # 保存每个 benchmark 的详细结果
            if args.save_path:
                bench_save = args.save_path.replace(".jsonl", f"_{bench_name}.jsonl")
                os.makedirs(os.path.dirname(bench_save) or ".", exist_ok=True)
                with open(bench_save, "w") as f:
                    for r in records:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 打印汇总
    if is_main and results_summary:
        print(f"\n{'='*60}")
        print("General Ability Summary")
        print(f"{'='*60}")
        print(f"{'Benchmark':<20} {'Accuracy':>10}")
        print(f"{'-'*30}")
        for bench, acc in results_summary.items():
            print(f"{bench:<20} {acc:>10.4f}")
        avg = sum(results_summary.values()) / len(results_summary)
        print(f"{'-'*30}")
        print(f"{'Average':<20} {avg:>10.4f}")
        print()

        # 保存汇总到 json
        if args.save_path:
            summary_path = args.save_path.replace(".jsonl", "_summary.json")
            os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
            with open(summary_path, "w") as f:
                results_summary["average"] = avg
                json.dump(results_summary, f, indent=2, ensure_ascii=False)
            print(f"Summary saved to {summary_path}")

    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

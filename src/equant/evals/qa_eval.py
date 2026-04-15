from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import string
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from equant.cache_factories import make_cache
from equant.kivi import patch_model_for_kivi
from equant.model_assets import ensure_model_assets
from equant.runtime import build_model_inputs, dtype_name, parse_device, resolve_torch_dtype, synchronize

REPO_ROOT = Path(__file__).resolve().parents[3]

BOOL_TRUE = {"true", "yes"}
BOOL_FALSE = {"false", "no"}
DEFAULT_SYSTEM_PROMPT = "Answer using only the shortest possible final answer."
QA_DATASETS = ["boolq", "squad", "coqa", "truthfulqa", "gsm8k"]
SINGLE_LINE_DATASETS = {"boolq", "squad", "coqa", "truthfulqa"}


def sanitize_model_id(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run open-source QA evaluations with dynamic/quanto/KIVI KV-cache modes.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--vendor-code-dir", default=None)
    parser.add_argument("--asset-manifest", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--cache", nargs="+", default=["dynamic", "kivi-int2", "quanto-int4"])
    parser.add_argument("--datasets", nargs="+", default=["boolq", "squad"], choices=QA_DATASETS)
    parser.add_argument("--max-samples-per-dataset", type=int, default=32)
    parser.add_argument("--max-answer-tokens", type=int, default=32)
    parser.add_argument("--residual-length", type=int, default=128)
    parser.add_argument("--q-group-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--output-dir", default=None)
    return parser


def load_model(model_path: str, device: torch.device, torch_dtype_arg, cache_modes: list[str]):
    load_kwargs = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": False,
        "torch_dtype": torch_dtype_arg,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    if any(cache_mode.lower().startswith("kivi-int") for cache_mode in cache_modes):
        patch_model_for_kivi(model)
    model.eval()
    model.to(device)
    return model, tokenizer


def sample_records(records, max_samples: int, seed: int):
    if max_samples < 0 or len(records) <= max_samples:
        return list(records)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(records)), max_samples))
    return [records[index] for index in indices]


def normalize_answer(text: str) -> str:
    lowered = text.lower()
    without_punc = "".join(ch for ch in lowered if ch not in string.punctuation)
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punc)
    return " ".join(without_articles.split())


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def token_f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0
    overlap = Counter(pred_tokens) & Counter(truth_tokens)
    matches = sum(overlap.values())
    if matches == 0:
        return 0.0
    precision = matches / len(pred_tokens)
    recall = matches / len(truth_tokens)
    return 2.0 * precision * recall / (precision + recall)


def parse_bool_prediction(text: str) -> bool | None:
    normalized = normalize_answer(text)
    for token in normalized.split():
        if token in BOOL_TRUE:
            return True
        if token in BOOL_FALSE:
            return False
    return None


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        normalized = normalize_answer(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(value)
    return result


def extract_last_number(text: str) -> str | None:
    matches = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text.replace("$", ""))
    if not matches:
        return None
    return matches[-1].replace(",", "")


def stop_strings_for_dataset(dataset_name: str) -> list[str]:
    if dataset_name in SINGLE_LINE_DATASETS:
        return ["\n", "\nQ:", "\nQuestion:"]
    return []


def should_stop_generation(tokenizer, generated_ids: list[int], stop_strings: list[str]) -> bool:
    if not stop_strings or not generated_ids:
        return False
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return any(stop_string in text for stop_string in stop_strings if stop_string)


def clean_prediction_text(dataset_name: str, prediction: str) -> str:
    cleaned = prediction.strip()
    cleaned = re.sub(r"^\s*(?:assistant|answer)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    if dataset_name in SINGLE_LINE_DATASETS:
        cleaned = re.split(r"\n|(?:^|\s)(?:Q:|Question:)", cleaned, maxsplit=1)[0].strip()
        cleaned = re.sub(r"^\s*A:\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned


def encode_prompt(tokenizer, prompt: str, *, use_chat_template: bool, system_prompt: str | None) -> list[int]:
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    text = prompt if not system_prompt else f"{system_prompt}\n\n{prompt}"
    return tokenizer.encode(text, add_special_tokens=False)


def load_boolq_records(max_samples: int, seed: int):
    from datasets import load_dataset

    dataset = load_dataset("super_glue", "boolq", split="validation")
    records = []
    for index in range(len(dataset)):
        row = dataset[index]
        prompt = (
            "Read the passage and answer the question.\n\n"
            f"Passage:\n{row['passage']}\n\n"
            f"Question: {row['question']}\n"
            "Respond with only true or false.\n"
            "Answer: "
        )
        records.append(
            {
                "sample_id": row.get("idx", index),
                "prompt": prompt,
                "answers": ["true" if row["label"] else "false"],
                "dataset": "boolq",
                "metric_name": "accuracy",
            }
        )
    return sample_records(records, max_samples=max_samples, seed=seed)


def load_squad_records(max_samples: int, seed: int):
    from datasets import load_dataset

    dataset = load_dataset("squad", split="validation")
    records = []
    for index in range(len(dataset)):
        row = dataset[index]
        prompt = (
            "Answer the question using the context.\n\n"
            f"Context:\n{row['context']}\n\n"
            f"Question: {row['question']}\n"
            "Answer: "
        )
        records.append(
            {
                "sample_id": row["id"],
                "prompt": prompt,
                "answers": row["answers"]["text"],
                "dataset": "squad",
                "metric_name": "f1",
            }
        )
    return sample_records(records, max_samples=max_samples, seed=seed)


def load_coqa_records(max_samples: int, seed: int):
    from datasets import load_dataset

    dataset = load_dataset("coqa", split="validation")
    records = []
    for story_index in range(len(dataset)):
        row = dataset[story_index]
        story = row["story"]
        history_lines = []
        for turn_index, question in enumerate(row["questions"]):
            answer = row["answers"]["input_text"][turn_index]
            prompt_parts = [
                "Read the story and answer the conversational question.",
                "",
                f"Story:\n{story}",
            ]
            if history_lines:
                prompt_parts.extend(["", "Conversation so far:", "\n".join(history_lines)])
            prompt_parts.extend(["", f"Question: {question}", "Answer: "])
            records.append(
                {
                    "sample_id": f"{story_index}-{turn_index}",
                    "prompt": "\n".join(prompt_parts),
                    "answers": [answer],
                    "dataset": "coqa",
                    "metric_name": "f1",
                }
            )
            history_lines.append(f"Q: {question}\nA: {answer}")
    return sample_records(records, max_samples=max_samples, seed=seed)


def load_truthfulqa_records(max_samples: int, seed: int):
    from datasets import load_dataset

    dataset = load_dataset("truthful_qa", "generation", split="validation")
    records = []
    for index in range(len(dataset)):
        row = dataset[index]
        prompt = (
            "Answer the question truthfully and concisely.\n\n"
            f"Question: {row['question']}\n"
            "Answer: "
        )
        answers = dedupe_preserve_order([row["best_answer"], *row["correct_answers"]])
        records.append(
            {
                "sample_id": index,
                "prompt": prompt,
                "answers": answers,
                "dataset": "truthfulqa",
                "metric_name": "f1",
            }
        )
    return sample_records(records, max_samples=max_samples, seed=seed)


def load_gsm8k_records(max_samples: int, seed: int):
    from datasets import load_dataset

    dataset = load_dataset("gsm8k", "main", split="test")
    records = []
    for index in range(len(dataset)):
        row = dataset[index]
        final_answer = extract_last_number(row["answer"])
        if final_answer is None:
            continue
        prompt = (
            "Solve the math word problem. Respond with only the final numeric answer.\n\n"
            f"Question: {row['question']}\n"
            "Answer: "
        )
        records.append(
            {
                "sample_id": index,
                "prompt": prompt,
                "answers": [final_answer],
                "dataset": "gsm8k",
                "metric_name": "accuracy",
            }
        )
    return sample_records(records, max_samples=max_samples, seed=seed)


DATASET_LOADERS = {
    "boolq": load_boolq_records,
    "squad": load_squad_records,
    "coqa": load_coqa_records,
    "truthfulqa": load_truthfulqa_records,
    "gsm8k": load_gsm8k_records,
}


def score_prediction(dataset_name: str, prediction: str, answers: list[str]) -> dict:
    if dataset_name == "boolq":
        parsed = parse_bool_prediction(prediction)
        truth = answers[0].lower() == "true"
        accuracy = float(parsed is not None and parsed == truth)
        return {
            "primary_score": accuracy,
            "accuracy": accuracy,
        }

    if dataset_name == "squad":
        exact_match = max(exact_match_score(prediction, answer) for answer in answers)
        f1 = max(token_f1_score(prediction, answer) for answer in answers)
        return {
            "primary_score": f1,
            "exact_match": exact_match,
            "f1": f1,
        }

    if dataset_name == "coqa":
        exact_match = max(exact_match_score(prediction, answer) for answer in answers)
        f1 = max(token_f1_score(prediction, answer) for answer in answers)
        return {
            "primary_score": f1,
            "exact_match": exact_match,
            "f1": f1,
        }

    if dataset_name == "truthfulqa":
        exact_match = max(exact_match_score(prediction, answer) for answer in answers)
        f1 = max(token_f1_score(prediction, answer) for answer in answers)
        return {
            "primary_score": f1,
            "exact_match": exact_match,
            "f1": f1,
        }

    if dataset_name == "gsm8k":
        prediction_number = extract_last_number(prediction)
        accuracy = float(prediction_number is not None and prediction_number == answers[0])
        return {
            "primary_score": accuracy,
            "accuracy": accuracy,
            "prediction_number": prediction_number,
        }

    raise ValueError(f"Unsupported dataset: {dataset_name}")


@torch.inference_mode()
def greedy_generate(
    model,
    tokenizer,
    *,
    dataset_name: str,
    prompt_ids: list[int],
    cache_mode: str,
    device: torch.device,
    answer_max_tokens: int,
    residual_length: int,
    q_group_size: int,
):
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    cache = make_cache(cache_mode, model.config, residual_length=residual_length, q_group_size=q_group_size)
    cache_position = torch.arange(input_ids.shape[1], device=device)

    synchronize(device)
    prefill_start = time.perf_counter()
    outputs = model(
        **build_model_inputs(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=cache,
            use_cache=True,
        ),
    )
    synchronize(device)
    prefill_latency_s = time.perf_counter() - prefill_start

    generated_ids: list[int] = []
    past_key_values = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    eos_token_id = tokenizer.eos_token_id
    decode_step_latencies = []
    stop_strings = stop_strings_for_dataset(dataset_name)

    for _ in range(answer_max_tokens):
        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
            dim=-1,
        )
        cache_position = cache_position[-1:] + 1
        synchronize(device)
        decode_start = time.perf_counter()
        outputs = model(
            **build_model_inputs(
                model,
                input_ids=next_token,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                use_cache=True,
            ),
        )
        synchronize(device)
        decode_step_latencies.append(time.perf_counter() - decode_start)
        token_id = int(next_token.item())
        generated_ids.append(token_id)
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        if eos_token_id is not None and token_id == eos_token_id:
            break
        if should_stop_generation(tokenizer, generated_ids, stop_strings):
            break

    return {
        "prediction": clean_prediction_text(dataset_name, tokenizer.decode(generated_ids, skip_special_tokens=True)),
        "prefill_latency_s": prefill_latency_s,
        "decode_latency_s": sum(decode_step_latencies),
        "decode_step_mean_ms": statistics.mean(decode_step_latencies) * 1000.0 if decode_step_latencies else 0.0,
        "generated_tokens": len(generated_ids),
    }


def summarize_prediction(text: str) -> str:
    return " ".join(text.strip().split())[:200]


def print_summary(rows: list[dict]) -> None:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["cache_mode"])].append(row)

    header = "dataset".ljust(14) + "cache".ljust(14) + "samples".ljust(10) + "primary".ljust(12) + "extra".ljust(12)
    print(header)
    print("-" * len(header))
    for (dataset_name, cache_mode), group_rows in sorted(grouped.items()):
        samples = len(group_rows)
        if dataset_name in {"boolq", "gsm8k"}:
            accuracy = 100.0 * sum(row["accuracy"] for row in group_rows) / samples
            extra = f"acc={accuracy:.2f}"
            primary = accuracy
        else:
            f1 = 100.0 * sum(row["f1"] for row in group_rows) / samples
            exact_match = 100.0 * sum(row["exact_match"] for row in group_rows) / samples
            primary = f1
            extra = f"em={exact_match:.2f}"

        print(
            dataset_name.ljust(14)
            + cache_mode.ljust(14)
            + str(samples).ljust(10)
            + f"{primary:.2f}".ljust(12)
            + extra.ljust(12)
        )


def write_results(output_dir: Path, rows: list[dict]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = output_dir / f"qa_eval_{timestamp}.jsonl"
    with result_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return result_path


def ensure_runtime_feasibility(device: torch.device) -> None:
    if device.type == "cpu":
        print(
            "Warning: running on CPU. QA evaluation will be slow; prefer a smaller Qwen2 model for quick smoke tests.",
            file=sys.stderr,
        )


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = parse_device(args.device)
    torch_dtype_arg = resolve_torch_dtype(args.torch_dtype)
    ensure_runtime_feasibility(device)
    sanitized_model_id = sanitize_model_id(args.model_id)
    model_dir = Path(args.model_dir) if args.model_dir else REPO_ROOT / "artifacts" / "models" / sanitized_model_id
    manifest_path = (
        Path(args.asset_manifest) if args.asset_manifest else REPO_ROOT / "artifacts" / "manifests" / f"{sanitized_model_id}.json"
    )

    asset_paths = ensure_model_assets(
        model_id=args.model_id,
        model_dir=model_dir,
        vendor_code_dir=Path(args.vendor_code_dir) if args.vendor_code_dir else None,
        manifest_path=manifest_path,
        revision=args.revision,
        token=args.token,
        download_if_missing=not args.no_download,
        export_code_if_missing=True,
    )
    model, tokenizer = load_model(str(asset_paths.model_dir), device, torch_dtype_arg, args.cache)
    use_chat_template = args.use_chat_template or bool(getattr(tokenizer, "chat_template", None))

    rows = []
    for dataset_name in args.datasets:
        records = DATASET_LOADERS[dataset_name](args.max_samples_per_dataset, args.seed)
        for sample_index, record in enumerate(records):
            prompt_ids = encode_prompt(
                tokenizer,
                record["prompt"],
                use_chat_template=use_chat_template,
                system_prompt=args.system_prompt,
            )
            for cache_mode in args.cache:
                generation = greedy_generate(
                    model,
                    tokenizer,
                    dataset_name=dataset_name,
                    prompt_ids=prompt_ids,
                    cache_mode=cache_mode,
                    device=device,
                    answer_max_tokens=args.max_answer_tokens,
                    residual_length=args.residual_length,
                    q_group_size=args.q_group_size,
                )
                prediction = generation["prediction"]
                metrics = score_prediction(dataset_name, prediction, record["answers"])
                rows.append(
                    {
                        "dataset": dataset_name,
                        "sample_index": sample_index,
                        "sample_id": record["sample_id"],
                        "model_id": args.model_id,
                        "model_path": str(asset_paths.model_dir),
                        "device": str(device),
                        "torch_dtype": dtype_name(torch_dtype_arg),
                        "cache_mode": cache_mode,
                        "answers": record["answers"],
                        "prediction": summarize_prediction(prediction),
                        "metric_name": record["metric_name"],
                        **metrics,
                        "prefill_latency_s": generation["prefill_latency_s"],
                        "decode_latency_s": generation["decode_latency_s"],
                        "decode_step_mean_ms": generation["decode_step_mean_ms"],
                        "generated_tokens": generation["generated_tokens"],
                    }
                )

    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "results"
    result_path = write_results(output_dir=output_dir, rows=rows)
    print_summary(rows)
    print(f"\nresults written to {result_path}")


if __name__ == "__main__":
    main()

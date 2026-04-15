from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from equant.cache_factories import make_cache
from equant.kivi import patch_model_for_kivi
from equant.model_assets import ensure_model_assets
from equant.runtime import build_model_inputs, dtype_name, parse_device, resolve_torch_dtype, synchronize

REPO_ROOT = Path(__file__).resolve().parents[3]

LONG_BENCH_E_PROMPTS = {
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia.\n"
        "Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique "
        "paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there "
        "in total?\n\n"
        "{context}\n\n"
        "Please enter the final count of unique paragraphs after removing duplicates. "
        "The output format should only contain the number, such as 1, 2, 3, and so on.\n\n"
        "The final answer is: "
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract.\n"
        "Please determine which paragraph the abstract is from.\n\n"
        "{context}\n\n"
        "The following is an abstract.\n\n"
        "{input}\n\n"
        "Please enter the number of the paragraph that the abstract is from. "
        "The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\n"
        "The answer is: "
    ),
}

LONG_BENCH_E_MAX_GEN = {
    "passage_count": 32,
    "passage_retrieval_en": 32,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a minimal official LongBench-E subset with custom KV cache modes.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-14B")
    parser.add_argument(
        "--model-dir",
        default=str(REPO_ROOT / "artifacts" / "models" / "Qwen2.5-14B"),
    )
    parser.add_argument(
        "--vendor-code-dir",
        default=str(REPO_ROOT / "artifacts" / "model_code" / "transformers_qwen2"),
    )
    parser.add_argument(
        "--asset-manifest",
        default=str(REPO_ROOT / "artifacts" / "manifests" / "qwen2_14b_assets.json"),
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--cache", nargs="+", default=["dynamic", "quanto-int4", "quanto-int2"])
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["passage_count", "passage_retrieval_en"],
        choices=sorted(LONG_BENCH_E_PROMPTS),
    )
    parser.add_argument(
        "--max-samples-per-dataset",
        type=int,
        default=20,
        help="Use -1 to run the full selected LongBench-E dataset(s).",
    )
    parser.add_argument("--max-context-tokens", type=int, default=None)
    parser.add_argument("--residual-length", type=int, default=128)
    parser.add_argument("--q-group-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
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


def resolve_context_limit(model, tokenizer, explicit_limit: int | None) -> int | None:
    if explicit_limit is not None:
        return explicit_limit

    candidates = [
        getattr(model.config, "max_position_embeddings", None),
        getattr(model.config, "sliding_window", None),
        getattr(tokenizer, "model_max_length", None),
    ]
    normalized = [value for value in candidates if isinstance(value, int) and 0 < value < 10_000_000]
    return min(normalized) if normalized else None


def truncate_from_middle(token_ids: list[int], max_prompt_tokens: int) -> list[int]:
    if len(token_ids) <= max_prompt_tokens:
        return token_ids
    left = max_prompt_tokens // 2
    right = max_prompt_tokens - left
    return token_ids[:left] + token_ids[-right:]


def count_score(prediction: str, ground_truth: str) -> float:
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    right_num = sum(1 for number in numbers if str(number) == str(ground_truth))
    return float(right_num / len(numbers))


def retrieval_score(prediction: str, ground_truth: str) -> float:
    matches = re.findall(r"Paragraph (\d+)", ground_truth)
    if not matches:
        return 0.0
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    right_num = sum(1 for number in numbers if str(number) == str(ground_truth_id))
    return float(right_num / len(numbers))


DATASET_TO_SCORER = {
    "passage_count": count_score,
    "passage_retrieval_en": retrieval_score,
}


def score_prediction(dataset_name: str, prediction: str, answers: list[str]) -> float:
    scorer = DATASET_TO_SCORER[dataset_name]
    return max(scorer(prediction, ground_truth) for ground_truth in answers)


def sample_records(records, max_samples: int, seed: int):
    if max_samples < 0 or len(records) <= max_samples:
        return list(records)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(records)), max_samples))
    return [records[index] for index in indices]


def load_longbench_e_records(dataset_name: str, max_samples: int, seed: int):
    from datasets import load_dataset

    dataset = load_dataset("THUDM/LongBench", f"{dataset_name}_e", split="test")
    records = [dataset[index] for index in range(len(dataset))]
    return sample_records(records, max_samples=max_samples, seed=seed)


@torch.inference_mode()
def greedy_generate(
    model,
    tokenizer,
    *,
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

    return {
        "prediction": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "prefill_latency_s": prefill_latency_s,
        "decode_latency_s": sum(decode_step_latencies),
        "decode_step_mean_ms": statistics.mean(decode_step_latencies) * 1000.0 if decode_step_latencies else 0.0,
        "generated_tokens": len(generated_ids),
    }


def summarize_prediction(text: str) -> str:
    return " ".join(text.strip().split())[:160]


def length_bucket(length: int) -> str:
    if length < 4000:
        return "0-4k"
    if length < 8000:
        return "4-8k"
    return "8k+"


def print_summary(rows: list[dict]) -> None:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    bucketed: dict[tuple[str, str, str], list[float]] = defaultdict(list)

    for row in rows:
        grouped[(row["dataset"], row["cache_mode"])].append(row["score"])
        bucketed[(row["dataset"], row["cache_mode"], row["length_bucket"])].append(row["score"])

    header = "dataset".ljust(24) + "cache".ljust(14) + "samples".ljust(10) + "score".ljust(10)
    print(header)
    print("-" * len(header))
    for (dataset_name, cache_mode), values in sorted(grouped.items()):
        print(dataset_name.ljust(24) + cache_mode.ljust(14) + str(len(values)).ljust(10) + f"{100.0 * sum(values) / len(values):.2f}")

    print("\nlength buckets")
    print("dataset".ljust(24) + "cache".ljust(14) + "bucket".ljust(10) + "score".ljust(10))
    print("-" * 58)
    for (dataset_name, cache_mode, bucket), values in sorted(bucketed.items()):
        print(dataset_name.ljust(24) + cache_mode.ljust(14) + bucket.ljust(10) + f"{100.0 * sum(values) / len(values):.2f}")


def write_results(output_dir: Path, rows: list[dict]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = output_dir / f"longbench_e_{timestamp}.jsonl"
    with result_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return result_path


def ensure_runtime_feasibility(device: torch.device) -> None:
    if device.type == "cpu":
        print(
            "Warning: running on CPU. LongBench-E evaluation for Qwen2.5-14B will be very slow.",
            file=sys.stderr,
        )


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = parse_device(args.device)
    torch_dtype_arg = resolve_torch_dtype(args.torch_dtype)
    ensure_runtime_feasibility(device)

    asset_paths = ensure_model_assets(
        model_id=args.model_id,
        model_dir=Path(args.model_dir),
        vendor_code_dir=Path(args.vendor_code_dir),
        manifest_path=Path(args.asset_manifest),
        revision=args.revision,
        token=args.token,
        download_if_missing=not args.no_download,
        export_code_if_missing=True,
    )
    model, tokenizer = load_model(str(asset_paths.model_dir), device, torch_dtype_arg, args.cache)
    context_limit = resolve_context_limit(model, tokenizer, args.max_context_tokens)

    rows = []
    for dataset_name in args.datasets:
        records = load_longbench_e_records(
            dataset_name,
            max_samples=args.max_samples_per_dataset,
            seed=args.seed,
        )
        prompt_format = LONG_BENCH_E_PROMPTS[dataset_name]
        answer_max_tokens = LONG_BENCH_E_MAX_GEN[dataset_name]

        for sample_index, record in enumerate(records):
            prompt = prompt_format.format(**record)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if context_limit is not None:
                prompt_ids = truncate_from_middle(prompt_ids, max(1, context_limit - answer_max_tokens))

            for cache_mode in args.cache:
                generation = greedy_generate(
                    model,
                    tokenizer,
                    prompt_ids=prompt_ids,
                    cache_mode=cache_mode,
                    device=device,
                    answer_max_tokens=answer_max_tokens,
                    residual_length=args.residual_length,
                    q_group_size=args.q_group_size,
                )
                prediction = generation["prediction"]
                score = score_prediction(dataset_name, prediction, record["answers"])
                rows.append(
                    {
                        "dataset": dataset_name,
                        "sample_index": sample_index,
                        "sample_id": record["_id"],
                        "length": record["length"],
                        "length_bucket": length_bucket(record["length"]),
                        "model_id": args.model_id,
                        "model_path": str(asset_paths.model_dir),
                        "device": str(device),
                        "torch_dtype": dtype_name(torch_dtype_arg),
                        "cache_mode": cache_mode,
                        "score": score,
                        "prediction": summarize_prediction(prediction),
                        "answers": record["answers"],
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

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from equant.cache_factories import make_cache
from equant.kivi import patch_model_for_kivi
from equant.model_assets import ensure_model_assets
from equant.runtime import build_model_inputs, dtype_name, parse_device, resolve_torch_dtype, synchronize

REPO_ROOT = Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark KV cache latency for Qwen models.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-14B")
    parser.add_argument(
        "--model-dir",
        "--model-path",
        dest="model_dir",
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
    parser.add_argument("--prompt-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--residual-length", type=int, default=128)
    parser.add_argument("--q-group-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--prompt-template", default="请解释 KV cache 量化对大模型推理时延和显存占用的影响。")
    parser.add_argument("--output-dir", default=None)
    return parser


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def build_input_ids(tokenizer, prompt_length: int, batch_size: int, prompt_template: str, device: torch.device):
    seed_ids = tokenizer.encode(prompt_template, add_special_tokens=False)
    if not seed_ids:
        raise ValueError("Prompt template tokenized to an empty sequence.")
    repeats = (prompt_length + len(seed_ids) - 1) // len(seed_ids)
    prompt_ids = (seed_ids * repeats)[:prompt_length]
    input_ids = torch.tensor([prompt_ids for _ in range(batch_size)], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    return input_ids, attention_mask


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


@torch.inference_mode()
def run_single_trial(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cache_mode: str,
    max_new_tokens: int,
    residual_length: int,
    q_group_size: int,
):
    device = input_ids.device
    cache = make_cache(cache_mode, model.config, residual_length=residual_length, q_group_size=q_group_size)
    running_attention_mask = attention_mask
    cache_position = torch.arange(input_ids.shape[1], device=device)

    synchronize(device)
    prefill_start = time.perf_counter()
    outputs = model(
        **build_model_inputs(
            model,
            input_ids=input_ids,
            attention_mask=running_attention_mask,
            cache_position=cache_position,
            past_key_values=cache,
            use_cache=True,
        ),
    )
    synchronize(device)
    prefill_latency_s = time.perf_counter() - prefill_start

    past_key_values = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    decode_step_latencies = []
    generated_tokens = 0

    for _ in range(max_new_tokens):
        running_attention_mask = torch.cat(
            [
                running_attention_mask,
                running_attention_mask.new_ones((running_attention_mask.shape[0], 1)),
            ],
            dim=-1,
        )
        cache_position = cache_position[-1:] + 1
        synchronize(device)
        decode_step_start = time.perf_counter()
        outputs = model(
            **build_model_inputs(
                model,
                input_ids=next_token,
                attention_mask=running_attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                use_cache=True,
            ),
        )
        synchronize(device)
        decode_step_latencies.append(time.perf_counter() - decode_step_start)
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_tokens += 1

    decode_latency_s = sum(decode_step_latencies)
    return {
        "prefill_latency_s": prefill_latency_s,
        "decode_latency_s": decode_latency_s,
        "decode_step_mean_ms": statistics.mean(decode_step_latencies) * 1000.0,
        "decode_step_p50_ms": percentile(decode_step_latencies, 0.50) * 1000.0,
        "decode_step_p95_ms": percentile(decode_step_latencies, 0.95) * 1000.0,
        "generated_tokens": generated_tokens,
    }


def write_results(output_dir: Path, rows: list[dict]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = output_dir / f"kv_latency_{timestamp}.jsonl"
    with result_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return result_path


def print_summary(rows: list[dict]) -> None:
    header = (
        "cache".ljust(14)
        + "trial".ljust(8)
        + "prefill(s)".ljust(14)
        + "decode(s)".ljust(14)
        + "mean_step(ms)".ljust(18)
        + "tok/s".ljust(12)
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            row["cache_mode"].ljust(14)
            + str(row["trial"]).ljust(8)
            + f'{row["prefill_latency_s"]:.4f}'.ljust(14)
            + f'{row["decode_latency_s"]:.4f}'.ljust(14)
            + f'{row["decode_step_mean_ms"]:.3f}'.ljust(18)
            + f'{row["decode_tokens_per_s"]:.3f}'.ljust(12)
        )


def ensure_runtime_feasibility(device: torch.device, prompt_length: int, max_new_tokens: int) -> None:
    if device.type == "cpu":
        print(
            "Warning: running on CPU. Qwen2.5-14B latency data on CPU is usually not representative.",
            file=sys.stderr,
        )
    if prompt_length <= 0 or max_new_tokens <= 0:
        raise ValueError("prompt_length and max_new_tokens must be positive.")


def main() -> None:
    global args
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)

    device = parse_device(args.device)
    torch_dtype_arg = resolve_torch_dtype(args.torch_dtype)
    ensure_runtime_feasibility(device, args.prompt_length, args.max_new_tokens)

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
    input_ids, attention_mask = build_input_ids(
        tokenizer=tokenizer,
        prompt_length=args.prompt_length,
        batch_size=args.batch_size,
        prompt_template=args.prompt_template,
        device=device,
    )

    for _ in range(args.warmup):
        _ = run_single_trial(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            cache_mode=args.cache[0],
            max_new_tokens=args.max_new_tokens,
            residual_length=args.residual_length,
            q_group_size=args.q_group_size,
        )

    rows = []
    for cache_mode in args.cache:
        for trial in range(1, args.trials + 1):
            metrics = run_single_trial(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                cache_mode=cache_mode,
                max_new_tokens=args.max_new_tokens,
                residual_length=args.residual_length,
                q_group_size=args.q_group_size,
            )
            row = {
                "model_path": str(asset_paths.model_dir),
                "model_id": args.model_id,
                "device": str(device),
                "torch_dtype": dtype_name(torch_dtype_arg),
                "cache_mode": cache_mode,
                "trial": trial,
                "batch_size": args.batch_size,
                "prompt_length": args.prompt_length,
                "max_new_tokens": args.max_new_tokens,
                "residual_length": args.residual_length,
                "q_group_size": args.q_group_size,
                "decode_tokens_per_s": metrics["generated_tokens"] / max(metrics["decode_latency_s"], 1e-12),
                "prompt_tokens_per_s": (args.batch_size * args.prompt_length) / max(metrics["prefill_latency_s"], 1e-12),
                **metrics,
            }
            rows.append(row)

    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "results"
    result_path = write_results(output_dir=output_dir, rows=rows)
    print_summary(rows)
    print(f"\nresults written to {result_path}")


if __name__ == "__main__":
    main()

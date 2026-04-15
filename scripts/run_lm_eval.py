#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run EleutherAI LM Evaluation Harness against a locally prepared model snapshot, "
            "with optional equant cache modes such as KIVI."
        )
    )
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-14B")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--asset-manifest", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Allow downloading model assets if the local snapshot is missing.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["truthfulqa_gen", "gsm8k"],
        help="LM-Eval task names. Use '--list-tasks' after installation to inspect availability.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--cache-mode", default="dynamic")
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--num-fewshot", type=int, default=None)
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--use-cache", default=None)
    parser.add_argument("--log-samples", action="store_true")
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--gen-kwargs", default=None)
    parser.add_argument("--residual-length", type=int, default=128)
    parser.add_argument("--q-group-size", type=int, default=64)
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Run 'python -m lm_eval ls tasks' and exit.",
    )
    return parser


def ensure_lm_eval_installed() -> None:
    if importlib.util.find_spec("lm_eval") is None:
        raise ModuleNotFoundError(
            "lm_eval is not installed. Install it with '.venv/bin/pip install -r requirements.lm_eval.txt'."
        )


def resolve_output_path(output_path: str | None) -> Path | None:
    if output_path is None:
        return None

    path = Path(output_path)
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path / f"lm_eval_{timestamp}.json"


def write_results(results, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")


def main() -> None:
    args = build_parser().parse_args()
    if args.list_tasks:
        ensure_lm_eval_installed()
        subprocess.run([sys.executable, "-m", "lm_eval", "ls", "tasks"], check=True)
        return

    from equant.cache_factories import parse_cache_descriptor
    from equant.evals.qa_eval import load_model, sanitize_model_id
    from equant.model_assets import ensure_model_assets
    from equant.runtime import parse_device, resolve_torch_dtype

    parse_cache_descriptor(args.cache_mode)
    ensure_lm_eval_installed()

    from lm_eval.evaluator import simple_evaluate
    from lm_eval.utils import make_table

    from equant.evals.lm_eval_backend import EquantHFLM

    sanitized_model_id = sanitize_model_id(args.model_id)
    model_dir = Path(args.model_dir) if args.model_dir else REPO_ROOT / "artifacts" / "models" / sanitized_model_id
    manifest_path = (
        Path(args.asset_manifest) if args.asset_manifest else REPO_ROOT / "artifacts" / "manifests" / f"{sanitized_model_id}.json"
    )
    ensure_model_assets(
        model_id=args.model_id,
        model_dir=model_dir,
        manifest_path=manifest_path,
        revision=args.revision,
        token=args.token,
        download_if_missing=args.download,
        export_code_if_missing=False,
    )

    device = parse_device(args.device)
    torch_dtype = resolve_torch_dtype(args.torch_dtype)
    model, tokenizer = load_model(str(model_dir.resolve()), device, torch_dtype, [args.cache_mode])
    lm = EquantHFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        device=str(device),
        cache_mode=args.cache_mode,
        residual_length=args.residual_length,
        q_group_size=args.q_group_size,
    )

    results = simple_evaluate(
        model=lm,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=str(device),
        use_cache=args.use_cache,
        limit=args.limit,
        log_samples=args.log_samples,
        apply_chat_template=args.apply_chat_template,
        gen_kwargs=args.gen_kwargs,
    )
    if results is None:
        return

    results.setdefault("config", {})
    results["config"].update(
        {
            "model_id": args.model_id,
            "cache_mode": args.cache_mode,
            "torch_dtype": args.torch_dtype,
            "residual_length": args.residual_length,
            "q_group_size": args.q_group_size,
        }
    )

    print(make_table(results))
    output_path = resolve_output_path(args.output_path)
    if output_path is not None:
        write_results(results, output_path)
        print(f"results_path={output_path}")


if __name__ == "__main__":
    main()

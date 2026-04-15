#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from equant.evals.qa_eval import sanitize_model_id
from equant.model_assets import ensure_model_assets
from equant.runtime import parse_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run EleutherAI LM Evaluation Harness with the official HuggingFace backend "
            "against a locally prepared model snapshot."
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
        help="LM-Eval task names. Use 'python -m lm_eval --help' or 'python -m lm_eval ls tasks' after installation to inspect availability.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--num-fewshot", type=int, default=None)
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--use-cache", default=None)
    parser.add_argument("--log-samples", action="store_true")
    parser.add_argument(
        "--model-args-extra",
        default=None,
        help="Extra lm-eval --model_args entries, appended verbatim, e.g. 'parallelize=True,max_memory_per_gpu=70GiB'.",
    )
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


def main() -> None:
    args = build_parser().parse_args()
    ensure_lm_eval_installed()

    if args.list_tasks:
        subprocess.run([sys.executable, "-m", "lm_eval", "ls", "tasks"], check=True)
        return

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

    resolved_device = str(parse_device(args.device))
    model_args = [
        f"pretrained={model_dir.resolve()}",
        "trust_remote_code=False",
    ]
    if args.revision is not None:
        model_args.append(f"revision={args.revision}")
    if args.model_args_extra:
        model_args.append(args.model_args_extra)

    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        ",".join(model_args),
        "--tasks",
        ",".join(args.tasks),
        "--device",
        resolved_device,
        "--batch_size",
        str(args.batch_size),
    ]
    if args.num_fewshot is not None:
        cmd.extend(["--num_fewshot", str(args.num_fewshot)])
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.output_path is not None:
        cmd.extend(["--output_path", args.output_path])
    if args.use_cache is not None:
        cmd.extend(["--use_cache", args.use_cache])
    if args.log_samples:
        cmd.append("--log_samples")

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

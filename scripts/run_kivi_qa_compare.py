#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from equant.evals import qa_eval


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Qwen BF16 baseline vs KIVI-2 on CoQA, TruthfulQA, and GSM8K. "
            "Defaults follow the paper-style KIVI configuration: int2, group_size=32, residual_length=128."
        )
    )
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-14B")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--vendor-code-dir", default=None)
    parser.add_argument("--asset-manifest", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Allow downloading model assets if they are missing. By default this script assumes the model is already prepared locally.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-dtype", default="bf16")
    parser.add_argument("--max-samples-per-dataset", type=int, default=32)
    parser.add_argument("--max-answer-tokens", type=int, default=64)
    parser.add_argument(
        "--residual-length",
        type=int,
        default=128,
        choices=[32, 128],
        help="KIVI residual full-precision window. 128 is the main table setting; 32 matches the appendix supplement.",
    )
    parser.add_argument("--q-group-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--system-prompt", default=qa_eval.DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--output-dir", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    forwarded_argv = [
        "run_qa_eval.py",
        "--model-id",
        args.model_id,
        "--torch-dtype",
        args.torch_dtype,
        "--cache",
        "dynamic",
        "kivi-int2",
        "--datasets",
        "coqa",
        "truthfulqa",
        "gsm8k",
        "--max-samples-per-dataset",
        str(args.max_samples_per_dataset),
        "--max-answer-tokens",
        str(args.max_answer_tokens),
        "--residual-length",
        str(args.residual_length),
        "--q-group-size",
        str(args.q_group_size),
        "--seed",
        str(args.seed),
        "--system-prompt",
        args.system_prompt,
    ]

    optional_args = {
        "--model-dir": args.model_dir,
        "--vendor-code-dir": args.vendor_code_dir,
        "--asset-manifest": args.asset_manifest,
        "--revision": args.revision,
        "--token": args.token,
        "--device": args.device,
        "--output-dir": args.output_dir,
    }
    for flag, value in optional_args.items():
        if value is not None:
            forwarded_argv.extend([flag, value])

    if not args.download:
        forwarded_argv.append("--no-download")
    if args.use_chat_template:
        forwarded_argv.append("--use-chat-template")

    original_argv = sys.argv
    try:
        sys.argv = forwarded_argv
        qa_eval.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()

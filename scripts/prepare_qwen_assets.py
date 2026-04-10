#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from equant.model_assets import ensure_qwen_assets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ensure Qwen weights and local Qwen model source files are available."
    )
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
        "--manifest-path",
        default=str(REPO_ROOT / "artifacts" / "manifests" / "qwen2_14b_assets.json"),
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--no-download", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    asset_paths = ensure_qwen_assets(
        model_id=args.model_id,
        model_dir=Path(args.model_dir),
        vendor_code_dir=Path(args.vendor_code_dir),
        manifest_path=Path(args.manifest_path),
        revision=args.revision,
        token=args.token,
        download_if_missing=not args.no_download,
        export_code_if_missing=True,
    )
    print(f"model_dir={asset_paths.model_dir}")
    if asset_paths.vendor_code_dir is not None:
        print(f"vendor_code_dir={asset_paths.vendor_code_dir}")
    if asset_paths.manifest_path is not None:
        print(f"manifest_path={asset_paths.manifest_path}")


if __name__ == "__main__":
    main()

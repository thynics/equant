#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from equant.model_assets import ensure_model_assets


def default_model_dir(model_id: str) -> Path:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", model_id)
    return REPO_ROOT / "artifacts" / "models" / sanitized


def default_manifest_path(model_id: str) -> Path:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", model_id)
    return REPO_ROOT / "artifacts" / "manifests" / f"{sanitized}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download or resume an open-source model snapshot for local evaluation.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--vendor-code-dir", default=None)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--no-download", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_dir = Path(args.model_dir) if args.model_dir else default_model_dir(args.model_id)
    manifest_path = Path(args.manifest_path) if args.manifest_path else default_manifest_path(args.model_id)
    vendor_code_dir = Path(args.vendor_code_dir) if args.vendor_code_dir else None
    asset_paths = ensure_model_assets(
        model_id=args.model_id,
        model_dir=model_dir,
        vendor_code_dir=vendor_code_dir,
        manifest_path=manifest_path,
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

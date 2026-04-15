from __future__ import annotations

import inspect
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import huggingface_hub
import transformers
from huggingface_hub import snapshot_download


@dataclass(frozen=True)
class AssetPaths:
    model_dir: Path
    vendor_code_dir: Path | None
    manifest_path: Path | None


WEIGHT_INDEX_FILES = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "tf_model.h5",
    "flax_model.msgpack",
)
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "spiece.model",
    "vocab.json",
)


def detect_model_type(model_dir: Path) -> str | None:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return None
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle).get("model_type")


def _required_weight_files(model_dir: Path) -> list[str]:
    for index_name in WEIGHT_INDEX_FILES:
        index_path = model_dir / index_name
        if not index_path.exists():
            continue
        with index_path.open("r", encoding="utf-8") as handle:
            index_data = json.load(handle)
        shard_files = sorted(set(index_data.get("weight_map", {}).values()))
        return [index_name, *shard_files]

    for filename in WEIGHT_FILES:
        if (model_dir / filename).exists():
            return [filename]
    return [WEIGHT_FILES[0]]


def model_snapshot_complete(model_dir: Path) -> bool:
    if not model_dir.exists():
        return False
    if not (model_dir / "config.json").exists():
        return False
    if not any((model_dir / filename).exists() for filename in TOKENIZER_FILES):
        return False
    for filename in _required_weight_files(model_dir):
        if not (model_dir / filename).exists():
            return False
    return True


def vendor_snapshot_complete(vendor_code_dir: Path, model_type: str | None = None) -> bool:
    if model_type != "qwen2":
        return vendor_code_dir.exists()
    required = [
        vendor_code_dir / "__init__.py",
        vendor_code_dir / "configuration_qwen2.py",
        vendor_code_dir / "modeling_qwen2.py",
    ]
    return all(path.exists() for path in required)


def export_qwen2_sources(target_dir: Path) -> list[str]:
    import transformers.models.qwen2 as qwen2_module

    source_dir = Path(inspect.getfile(qwen2_module)).resolve().parent
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)
    return sorted(str(path.relative_to(target_dir)) for path in target_dir.rglob("*") if path.is_file())


def ensure_model_assets(
    *,
    model_id: str,
    model_dir: Path,
    vendor_code_dir: Path | None = None,
    manifest_path: Path | None = None,
    revision: str | None = None,
    token: str | None = None,
    download_if_missing: bool = True,
    export_code_if_missing: bool = True,
) -> AssetPaths:
    model_dir = model_dir.resolve()
    if vendor_code_dir is not None:
        vendor_code_dir = vendor_code_dir.resolve()
    if manifest_path is not None:
        manifest_path = manifest_path.resolve()

    model_dir.parent.mkdir(parents=True, exist_ok=True)
    if vendor_code_dir is not None:
        vendor_code_dir.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_snapshot_complete(model_dir):
        if not download_if_missing:
            raise FileNotFoundError(
                f"Model snapshot is incomplete at {model_dir}. "
                "Re-run without --no-download to fetch or resume the weights."
            )
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            token=token,
            local_dir=str(model_dir),
        )

    model_type = detect_model_type(model_dir)
    exported_files: list[str] = []
    if vendor_code_dir is not None and export_code_if_missing and model_type == "qwen2":
        if not vendor_snapshot_complete(vendor_code_dir, model_type=model_type):
            exported_files = export_qwen2_sources(vendor_code_dir)

    if manifest_path is not None:
        manifest = {
            "model_id": model_id,
            "revision": revision,
            "model_type": model_type,
            "model_dir": str(model_dir),
            "vendor_code_dir": str(vendor_code_dir) if vendor_code_dir is not None else None,
            "transformers_version": transformers.__version__,
            "huggingface_hub_version": huggingface_hub.__version__,
            "model_snapshot_complete": model_snapshot_complete(model_dir),
            "vendor_snapshot_complete": (
                vendor_snapshot_complete(vendor_code_dir, model_type=model_type) if vendor_code_dir is not None else None
            ),
            "vendor_export_supported": model_type == "qwen2",
            "exported_files": exported_files,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return AssetPaths(
        model_dir=model_dir,
        vendor_code_dir=vendor_code_dir,
        manifest_path=manifest_path,
    )


def ensure_qwen_assets(**kwargs) -> AssetPaths:
    return ensure_model_assets(**kwargs)

from __future__ import annotations

from typing import Optional

import torch


def auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_device(device_name: Optional[str]) -> torch.device:
    return torch.device(device_name) if device_name else auto_device()


def resolve_torch_dtype(dtype_name: str):
    normalized = dtype_name.lower()
    if normalized == "auto":
        return "auto"
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[normalized]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def dtype_name(dtype_or_auto) -> str:
    if isinstance(dtype_or_auto, str):
        return dtype_or_auto
    return str(dtype_or_auto).replace("torch.", "")

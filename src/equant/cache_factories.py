from __future__ import annotations

from dataclasses import dataclass

from transformers.cache_utils import DynamicCache, QuantizedCache


@dataclass(frozen=True)
class CacheDescriptor:
    name: str
    backend: str | None = None
    nbits: int | None = None
    axis_key: int = 0
    axis_value: int = 0
    q_group_size: int = 64
    residual_length: int = 128


def parse_cache_descriptor(cache_name: str) -> CacheDescriptor:
    normalized = cache_name.lower()
    if normalized == "dynamic":
        return CacheDescriptor(name="dynamic")

    if normalized.startswith("quanto-int"):
        nbits = int(normalized.removeprefix("quanto-int"))
        return CacheDescriptor(name=normalized, backend="quanto", nbits=nbits)

    if normalized.startswith("hqq-int"):
        nbits = int(normalized.removeprefix("hqq-int"))
        return CacheDescriptor(name=normalized, backend="hqq", nbits=nbits, axis_key=1, axis_value=1)

    raise ValueError(f"Unsupported cache mode: {cache_name}")


def make_cache(cache_name: str, model_config, residual_length: int, q_group_size: int):
    descriptor = parse_cache_descriptor(cache_name)
    if descriptor.backend is None:
        return DynamicCache(config=model_config)

    return QuantizedCache(
        backend=descriptor.backend,
        config=model_config,
        nbits=descriptor.nbits,
        axis_key=descriptor.axis_key,
        axis_value=descriptor.axis_value,
        q_group_size=q_group_size,
        residual_length=residual_length,
    )

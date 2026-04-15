from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, apply_rotary_pos_emb, repeat_kv


def _safe_scale(mx: torch.Tensor, mn: torch.Tensor, bits: int) -> torch.Tensor:
    max_int = float(2**bits - 1)
    scale = (mx - mn) / max_int
    return torch.where(scale > 0, scale, torch.ones_like(scale))


def pack_tensor(data: torch.Tensor, bits: int, pack_dim: int) -> torch.Tensor:
    feat_per_int = 32 // bits
    shape = data.shape
    if shape[pack_dim] % feat_per_int != 0:
        raise ValueError(f"Dimension {shape[pack_dim]} must be divisible by {feat_per_int} for {bits}-bit packing.")

    code_shape = shape[:pack_dim] + (shape[pack_dim] // feat_per_int,) + shape[pack_dim + 1 :]
    code = torch.zeros(code_shape, dtype=torch.int32, device=data.device)
    unpacked_indices = [slice(None)] * len(shape)
    packed_indices = [slice(None)] * len(shape)

    unpacked_offset = 0
    packed_offset = 0
    while packed_offset < code.shape[pack_dim]:
        packed_indices[pack_dim] = packed_offset
        for inner in range(unpacked_offset, unpacked_offset + feat_per_int):
            unpacked_indices[pack_dim] = inner
            code[tuple(packed_indices)] |= data[tuple(unpacked_indices)] << (bits * (inner - unpacked_offset))
        unpacked_offset += feat_per_int
        packed_offset += 1
    return code


def unpack_tensor(code: torch.Tensor, bits: int, pack_dim: int) -> torch.Tensor:
    feat_per_int = 32 // bits
    shape = code.shape
    unpacked_shape = shape[:pack_dim] + (shape[pack_dim] * feat_per_int,) + shape[pack_dim + 1 :]
    unpacked = torch.zeros(unpacked_shape, dtype=torch.int16, device=code.device)
    packed_indices = [slice(None)] * len(unpacked_shape)
    packed_indices[pack_dim] = torch.arange(unpacked_shape[pack_dim], device=code.device) // feat_per_int
    shifts = (torch.arange(unpacked_shape[pack_dim], device=code.device) % feat_per_int) * bits
    mask = (1 << bits) - 1

    if pack_dim == 2:
        unpacked = (code[tuple(packed_indices)] >> shifts[None, None, :, None]) & mask
    elif pack_dim == 3:
        unpacked = (code[tuple(packed_indices)] >> shifts) & mask
    else:
        raise NotImplementedError(f"Unsupported pack_dim={pack_dim}")
    return unpacked


def quantize_key_cache(keys_transposed: torch.Tensor, group_size: int, bits: int):
    if keys_transposed.ndim != 4:
        raise ValueError("Expected transposed keys to be rank-4 [B, H, D, T].")
    batch, num_heads, head_dim, seq_len = keys_transposed.shape
    if seq_len % group_size != 0:
        raise ValueError(f"Key cache length {seq_len} must be divisible by group_size={group_size}.")

    max_int = 2**bits - 1
    reshaped = keys_transposed.view(batch, num_heads, head_dim, seq_len // group_size, group_size)
    mn = reshaped.amin(dim=-1, keepdim=True)
    mx = reshaped.amax(dim=-1, keepdim=True)
    scale = _safe_scale(mx, mn, bits)
    quantized = torch.round(((reshaped - mn) / scale).clamp(0, max_int)).to(torch.int32)
    quantized = quantized.view(batch, num_heads, head_dim, seq_len)
    code = pack_tensor(quantized, bits=bits, pack_dim=3)
    return code, scale.squeeze(-1), mn.squeeze(-1)


def dequantize_key_cache(code: torch.Tensor, scale: torch.Tensor, mn: torch.Tensor, group_size: int, bits: int) -> torch.Tensor:
    data = unpack_tensor(code, bits=bits, pack_dim=3).to(scale.dtype)
    batch, num_heads, head_dim, seq_len = data.shape
    reshaped = data.view(batch, num_heads, head_dim, seq_len // group_size, group_size)
    dequantized = reshaped * scale.unsqueeze(-1) + mn.unsqueeze(-1)
    return dequantized.view(batch, num_heads, head_dim, seq_len)


def quantize_value_cache(values: torch.Tensor, group_size: int, bits: int):
    if values.ndim != 4:
        raise ValueError("Expected values to be rank-4 [B, H, T, D].")
    if values.shape[-1] % group_size != 0:
        raise ValueError(f"Value head_dim {values.shape[-1]} must be divisible by group_size={group_size}.")

    max_int = 2**bits - 1
    reshaped = values.view(*values.shape[:-1], values.shape[-1] // group_size, group_size)
    mn = reshaped.amin(dim=-1, keepdim=True)
    mx = reshaped.amax(dim=-1, keepdim=True)
    scale = _safe_scale(mx, mn, bits)
    quantized = torch.round(((reshaped - mn) / scale).clamp(0, max_int)).to(torch.int32)
    quantized = quantized.view(values.shape)
    code = pack_tensor(quantized, bits=bits, pack_dim=3)
    return code, scale.squeeze(-1), mn.squeeze(-1)


def dequantize_value_cache(code: torch.Tensor, scale: torch.Tensor, mn: torch.Tensor, group_size: int, bits: int) -> torch.Tensor:
    data = unpack_tensor(code, bits=bits, pack_dim=3).to(scale.dtype)
    reshaped = data.view(*data.shape[:-1], data.shape[-1] // group_size, group_size)
    dequantized = reshaped * scale.unsqueeze(-1) + mn.unsqueeze(-1)
    return dequantized.view(data.shape)


@dataclass
class KIVILayerState:
    seq_length: int = 0
    key_states_quant_trans: Optional[torch.Tensor] = None
    key_states_full: Optional[torch.Tensor] = None
    key_scale_trans: Optional[torch.Tensor] = None
    key_mn_trans: Optional[torch.Tensor] = None
    value_states_quant: Optional[torch.Tensor] = None
    value_states_full: Optional[torch.Tensor] = None
    value_scale: Optional[torch.Tensor] = None
    value_mn: Optional[torch.Tensor] = None

    def get_seq_length(self, cache_position=None) -> int:
        return self.seq_length

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        return self.seq_length + cache_position.numel(), 0


class KIVICache:
    def __init__(self, *, num_hidden_layers: int, k_bits: int, v_bits: int, group_size: int, residual_length: int):
        if residual_length % group_size != 0:
            raise ValueError(
                f"KIVI requires residual_length ({residual_length}) to be divisible by group_size ({group_size})."
            )
        self.layers = [KIVILayerState() for _ in range(num_hidden_layers)]
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.group_size = group_size
        self.residual_length = residual_length
        self.is_compileable = False

    def get_seq_length(self, layer_idx: int = 0, cache_position=None) -> int:
        return self.layers[layer_idx].get_seq_length(cache_position=cache_position)

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        return self.layers[layer_idx].get_mask_sizes(cache_position)


class Qwen2AttentionKIVI(nn.Module):
    def __init__(self, base_attn: Qwen2Attention):
        super().__init__()
        self.base_attn = base_attn
        self.config = base_attn.config
        self.layer_idx = base_attn.layer_idx
        self.head_dim = base_attn.head_dim
        self.num_key_value_groups = base_attn.num_key_value_groups
        self.scaling = base_attn.scaling
        self.attention_dropout = base_attn.attention_dropout
        self.is_causal = base_attn.is_causal
        self.sliding_window = getattr(base_attn, "sliding_window", None)

    def _append_quantized_chunk(self, current: Optional[torch.Tensor], new_chunk: torch.Tensor, dim: int) -> torch.Tensor:
        return new_chunk if current is None else torch.cat([current, new_chunk], dim=dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value=None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if not isinstance(past_key_value, KIVICache):
            return self.base_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                cache_position=cache_position,
                **kwargs,
            )

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.base_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.base_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.base_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        layer_state = past_key_value.layers[self.layer_idx]
        if layer_state.seq_length == 0:
            attn_weights = torch.matmul(query_states, repeat_kv(key_states, self.num_key_value_groups).transpose(2, 3))
            attn_weights = attn_weights * self.scaling
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = F.dropout(attn_weights, p=0.0 if not self.training else self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, repeat_kv(value_states, self.num_key_value_groups))

            if key_states.shape[-2] < past_key_value.residual_length:
                key_states_quant_trans = None
                key_scale_trans = None
                key_mn_trans = None
                key_states_full = key_states
            elif key_states.shape[-2] % past_key_value.residual_length != 0:
                split = key_states.shape[-2] % past_key_value.residual_length
                quant_chunk = key_states[:, :, :-split, :].contiguous()
                key_states_full = key_states[:, :, -split:, :].contiguous()
                key_states_quant_trans, key_scale_trans, key_mn_trans = quantize_key_cache(
                    quant_chunk.transpose(2, 3).contiguous(),
                    group_size=past_key_value.group_size,
                    bits=past_key_value.k_bits,
                )
            else:
                key_states_quant_trans, key_scale_trans, key_mn_trans = quantize_key_cache(
                    key_states.transpose(2, 3).contiguous(),
                    group_size=past_key_value.group_size,
                    bits=past_key_value.k_bits,
                )
                key_states_full = None

            if value_states.shape[-2] <= past_key_value.residual_length:
                value_states_quant = None
                value_scale = None
                value_mn = None
                value_states_full = value_states
            else:
                quant_value_chunk = value_states[:, :, :-past_key_value.residual_length, :].contiguous()
                value_states_full = value_states[:, :, -past_key_value.residual_length :, :].contiguous()
                value_states_quant, value_scale, value_mn = quantize_value_cache(
                    quant_value_chunk,
                    group_size=past_key_value.group_size,
                    bits=past_key_value.v_bits,
                )

            layer_state.key_states_quant_trans = key_states_quant_trans
            layer_state.key_states_full = key_states_full
            layer_state.key_scale_trans = key_scale_trans
            layer_state.key_mn_trans = key_mn_trans
            layer_state.value_states_quant = value_states_quant
            layer_state.value_states_full = value_states_full
            layer_state.value_scale = value_scale
            layer_state.value_mn = value_mn
            layer_state.seq_length = key_states.shape[-2]
        else:
            quant_key_states = None
            if layer_state.key_states_quant_trans is not None:
                quant_key_states = dequantize_key_cache(
                    layer_state.key_states_quant_trans,
                    layer_state.key_scale_trans,
                    layer_state.key_mn_trans,
                    group_size=past_key_value.group_size,
                    bits=past_key_value.k_bits,
                ).transpose(2, 3).contiguous()

            key_states_full = key_states if layer_state.key_states_full is None else torch.cat([layer_state.key_states_full, key_states], dim=2)

            if quant_key_states is not None:
                quant_attn = torch.matmul(query_states, repeat_kv(quant_key_states, self.num_key_value_groups).transpose(2, 3))
                full_attn = torch.matmul(query_states, repeat_kv(key_states_full, self.num_key_value_groups).transpose(2, 3))
                attn_weights = torch.cat([quant_attn, full_attn], dim=-1)
            else:
                attn_weights = torch.matmul(query_states, repeat_kv(key_states_full, self.num_key_value_groups).transpose(2, 3))

            attn_weights = attn_weights * self.scaling
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask[:, :, :, : attn_weights.shape[-1]]
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = F.dropout(attn_weights, p=0.0 if not self.training else self.attention_dropout, training=self.training)

            value_states_full = value_states if layer_state.value_states_full is None else torch.cat([layer_state.value_states_full, value_states], dim=2)
            if layer_state.value_states_quant is not None:
                quant_value_states = dequantize_value_cache(
                    layer_state.value_states_quant,
                    layer_state.value_scale,
                    layer_state.value_mn,
                    group_size=past_key_value.group_size,
                    bits=past_key_value.v_bits,
                )
                full_len = value_states_full.shape[-2]
                attn_output = torch.matmul(
                    attn_weights[:, :, :, :-full_len],
                    repeat_kv(quant_value_states, self.num_key_value_groups),
                )
                attn_output = attn_output + torch.matmul(
                    attn_weights[:, :, :, -full_len:],
                    repeat_kv(value_states_full, self.num_key_value_groups),
                )
            else:
                attn_output = torch.matmul(attn_weights, repeat_kv(value_states_full, self.num_key_value_groups))

            if key_states_full is not None and key_states_full.shape[-2] == past_key_value.residual_length:
                quant_chunk, scale_chunk, mn_chunk = quantize_key_cache(
                    key_states_full.transpose(2, 3).contiguous(),
                    group_size=past_key_value.group_size,
                    bits=past_key_value.k_bits,
                )
                layer_state.key_states_quant_trans = self._append_quantized_chunk(layer_state.key_states_quant_trans, quant_chunk, dim=3)
                layer_state.key_scale_trans = self._append_quantized_chunk(layer_state.key_scale_trans, scale_chunk, dim=3)
                layer_state.key_mn_trans = self._append_quantized_chunk(layer_state.key_mn_trans, mn_chunk, dim=3)
                key_states_full = None

            overflow = 0 if value_states_full is None else max(0, value_states_full.shape[-2] - past_key_value.residual_length)
            if overflow > 0:
                quant_value_chunk, scale_chunk, mn_chunk = quantize_value_cache(
                    value_states_full[:, :, :overflow, :].contiguous(),
                    group_size=past_key_value.group_size,
                    bits=past_key_value.v_bits,
                )
                layer_state.value_states_quant = self._append_quantized_chunk(layer_state.value_states_quant, quant_value_chunk, dim=2)
                layer_state.value_scale = self._append_quantized_chunk(layer_state.value_scale, scale_chunk, dim=2)
                layer_state.value_mn = self._append_quantized_chunk(layer_state.value_mn, mn_chunk, dim=2)
                value_states_full = value_states_full[:, :, overflow:, :].contiguous()

            layer_state.key_states_full = key_states_full
            layer_state.value_states_full = value_states_full
            layer_state.seq_length += key_states.shape[-2]

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.base_attn.o_proj(attn_output)
        return attn_output, None


def patch_model_for_kivi(model) -> None:
    if getattr(model, "_equant_kivi_patched", False):
        return

    decoder = model.get_decoder() if hasattr(model, "get_decoder") else getattr(model, "model", None)
    if decoder is None or not hasattr(decoder, "layers"):
        raise TypeError("KIVI patching currently supports decoder-only Qwen2 style models.")

    for layer in decoder.layers:
        if not isinstance(layer.self_attn, Qwen2AttentionKIVI):
            layer.self_attn = Qwen2AttentionKIVI(layer.self_attn)
    model._equant_kivi_patched = True

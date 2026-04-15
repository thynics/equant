from __future__ import annotations

import copy
import unittest

import torch
from transformers import DynamicCache, Qwen2Config, Qwen2ForCausalLM

from equant.cache_factories import make_cache
from equant.kivi import (
    KIVICache,
    dequantize_key_cache,
    dequantize_value_cache,
    patch_model_for_kivi,
    quantize_key_cache,
    quantize_value_cache,
)
from equant.runtime import build_model_inputs


class KIVIQuantizationTests(unittest.TestCase):
    def test_key_quantization_round_trip_is_shape_preserving(self) -> None:
        torch.manual_seed(0)
        keys = torch.randn(2, 3, 8, 16, dtype=torch.float32)
        code, scale, mn = quantize_key_cache(keys, group_size=4, bits=2)
        restored = dequantize_key_cache(code, scale, mn, group_size=4, bits=2)

        self.assertEqual(restored.shape, keys.shape)
        self.assertLessEqual((restored - keys).abs().max().item(), scale.max().item() + 1e-6)

    def test_value_quantization_round_trip_is_shape_preserving(self) -> None:
        torch.manual_seed(0)
        values = torch.randn(2, 3, 16, 16, dtype=torch.float32)
        code, scale, mn = quantize_value_cache(values, group_size=4, bits=2)
        restored = dequantize_value_cache(code, scale, mn, group_size=4, bits=2)

        self.assertEqual(restored.shape, values.shape)
        self.assertLessEqual((restored - values).abs().max().item(), scale.max().item() + 1e-6)

    def test_kivi_cache_requires_residual_multiple_of_group_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            KIVICache(num_hidden_layers=1, k_bits=2, v_bits=2, group_size=32, residual_length=48)


class KIVIParityTests(unittest.TestCase):
    def test_kivi_matches_dynamic_before_quantization_triggers(self) -> None:
        torch.manual_seed(0)
        config = Qwen2Config(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            tie_word_embeddings=False,
        )
        dynamic_model = Qwen2ForCausalLM(config).eval()
        kivi_model = copy.deepcopy(dynamic_model).eval()
        patch_model_for_kivi(kivi_model)

        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        cache_position = torch.arange(input_ids.shape[1])
        dynamic_cache = DynamicCache()
        kivi_cache = make_cache("kivi-int2", kivi_model.config, residual_length=32, q_group_size=4)

        with torch.no_grad():
            dynamic_outputs = dynamic_model(
                **build_model_inputs(
                    dynamic_model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=dynamic_cache,
                    use_cache=True,
                    cache_position=cache_position,
                ),
            )
            kivi_outputs = kivi_model(
                **build_model_inputs(
                    kivi_model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=kivi_cache,
                    use_cache=True,
                    cache_position=cache_position,
                ),
            )

        self.assertTrue(torch.allclose(dynamic_outputs.logits, kivi_outputs.logits, atol=1e-5, rtol=1e-5))

        next_token = dynamic_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        dynamic_past = dynamic_outputs.past_key_values
        kivi_past = kivi_outputs.past_key_values

        for _ in range(3):
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((1, 1))], dim=-1)
            cache_position = cache_position[-1:] + 1
            with torch.no_grad():
                dynamic_outputs = dynamic_model(
                    **build_model_inputs(
                        dynamic_model,
                        input_ids=next_token,
                        attention_mask=attention_mask,
                        past_key_values=dynamic_past,
                        use_cache=True,
                        cache_position=cache_position,
                    ),
                )
                kivi_outputs = kivi_model(
                    **build_model_inputs(
                        kivi_model,
                        input_ids=next_token,
                        attention_mask=attention_mask,
                        past_key_values=kivi_past,
                        use_cache=True,
                        cache_position=cache_position,
                    ),
                )

            self.assertTrue(torch.allclose(dynamic_outputs.logits, kivi_outputs.logits, atol=1e-5, rtol=1e-5))
            next_token = dynamic_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            dynamic_past = dynamic_outputs.past_key_values
            kivi_past = kivi_outputs.past_key_values


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from lm_eval.models.huggingface import HFLM

from equant.cache_factories import make_cache
from equant.kivi import patch_model_for_kivi
from equant.runtime import build_model_inputs


class EquantHFLM(HFLM):
    def __init__(
        self,
        *,
        pretrained,
        tokenizer=None,
        cache_mode: str = "dynamic",
        residual_length: int = 128,
        q_group_size: int = 64,
        **kwargs,
    ) -> None:
        self.cache_mode = cache_mode.lower()
        self.residual_length = residual_length
        self.q_group_size = q_group_size
        super().__init__(pretrained=pretrained, tokenizer=tokenizer, **kwargs)
        if self.cache_mode.startswith("kivi-int"):
            patch_model_for_kivi(self.model)

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        if self.cache_mode != "dynamic":
            raise NotImplementedError(
                "LM-Eval integration with non-dynamic cache modes currently supports generate_until tasks only. "
                "Use generation tasks such as truthfulqa_gen or gsm8k, or switch back to --cache-mode dynamic."
            )
        return super().loglikelihood(requests, disable_tqdm=disable_tqdm)

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        if self.cache_mode != "dynamic":
            raise NotImplementedError(
                "LM-Eval integration with non-dynamic cache modes currently supports generate_until tasks only. "
                "Use generation tasks such as truthfulqa_gen or gsm8k, or switch back to --cache-mode dynamic."
            )
        return super().loglikelihood_rolling(requests, disable_tqdm=disable_tqdm)

    def _model_generate(
        self,
        context,
        max_length: int,
        stop: list[str],
        attention_mask: torch.Tensor | None = None,
        **generation_kwargs,
    ) -> torch.Tensor:
        if self.cache_mode == "dynamic":
            if attention_mask is not None:
                generation_kwargs["attention_mask"] = attention_mask
            return super()._model_generate(context=context, max_length=max_length, stop=stop, **generation_kwargs)

        return self._custom_cache_generate(
            context=context,
            attention_mask=attention_mask,
            max_length=max_length,
            stop=stop,
            **generation_kwargs,
        )

    def _custom_cache_generate(
        self,
        *,
        context: torch.Tensor,
        attention_mask: torch.Tensor | None,
        max_length: int,
        stop: list[str],
        **generation_kwargs,
    ) -> torch.Tensor:
        if self.backend != "causal":
            raise NotImplementedError("Custom cache LM-Eval integration currently supports decoder-only causal models only.")

        if generation_kwargs.get("do_sample") not in (None, False):
            raise NotImplementedError("Custom cache LM-Eval integration currently supports greedy decoding only.")

        if int(generation_kwargs.get("num_beams", 1)) != 1:
            raise NotImplementedError("Custom cache LM-Eval integration currently supports num_beams=1 only.")

        max_new_tokens = max(0, max_length - context.shape[1])
        mask = attention_mask if attention_mask is not None else torch.ones_like(context, device=context.device)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0

        full_sequences: list[torch.Tensor] = []
        for row_idx in range(context.shape[0]):
            prompt_ids = context[row_idx][mask[row_idx].bool()].to(self.device)
            generated_ids = self._generate_one(
                prompt_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                stop=stop,
            )
            full_sequence = torch.cat(
                [
                    context[row_idx],
                    torch.tensor(generated_ids, dtype=torch.long, device=context.device),
                ],
                dim=0,
            )
            full_sequences.append(full_sequence)

        max_total_length = max(sequence.shape[0] for sequence in full_sequences)
        result = torch.full(
            (len(full_sequences), max_total_length),
            pad_token_id,
            dtype=torch.long,
            device=context.device,
        )
        for row_idx, sequence in enumerate(full_sequences):
            result[row_idx, : sequence.shape[0]] = sequence
        return result

    def _generate_one(self, *, prompt_ids: torch.Tensor, max_new_tokens: int, stop: list[str]) -> list[int]:
        if max_new_tokens <= 0:
            return []

        input_ids = prompt_ids.unsqueeze(0)
        attention_mask = torch.ones_like(input_ids, device=self.device)
        cache = make_cache(
            self.cache_mode,
            self.model.config,
            residual_length=self.residual_length,
            q_group_size=self.q_group_size,
        )
        cache_position = torch.arange(input_ids.shape[1], device=self.device)
        generated_ids: list[int] = []
        eos_token_id = self.tokenizer.eos_token_id
        autocast_context = (
            torch.autocast(
                device_type=self.device.type,
                dtype=self.mixed_precision_dtype,
                enabled=True,
            )
            if self.mixed_precision_dtype is not None and self.device.type in {"cpu", "cuda"}
            else nullcontext()
        )

        with torch.no_grad(), autocast_context:
            outputs = self.model(
                **build_model_inputs(
                    self.model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=cache_position,
                ),
            )
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            for _ in range(max_new_tokens):
                token_id = int(next_token.item())
                generated_ids.append(token_id)
                if eos_token_id is not None and token_id == eos_token_id:
                    break
                if self._contains_stop_string(generated_ids, stop):
                    break

                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                    dim=-1,
                )
                cache_position = cache_position[-1:] + 1
                outputs = self.model(
                    **build_model_inputs(
                        self.model,
                        input_ids=next_token,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        cache_position=cache_position,
                    ),
                )
                past_key_values = outputs.past_key_values
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        return generated_ids

    def _contains_stop_string(self, generated_ids: list[int], stop: list[str]) -> bool:
        stop_strings = [value for value in stop if value]
        if not stop_strings:
            return False

        text = self.tok_decode(generated_ids)
        return any(stop_string in text for stop_string in stop_strings)

    def get_model_info(self) -> dict[str, Any]:
        info = super().get_model_info()
        info.update(
            {
                "cache_mode": self.cache_mode,
                "residual_length": self.residual_length,
                "q_group_size": self.q_group_size,
            }
        )
        return info

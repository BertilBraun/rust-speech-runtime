import os
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch
from torch import Tensor
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from voice_worker.cache import continuation_mask, continuation_positions, join_caches
from voice_worker.config import WorkerConfig
from voice_worker.model import ASSISTANT_SUFFIX, USER_PREFIX, SpeechModel


@pytest.fixture(scope="module")
def configured_model() -> SpeechModel:
    configuration_path = os.environ.get("VOICE_WORKER_CONFIG")
    if configuration_path is None or not torch.cuda.is_available():
        pytest.skip("Set VOICE_WORKER_CONFIG on a CUDA node with the trained projector")
    configuration = WorkerConfig.model_validate_json(
        Path(configuration_path).read_text(encoding="utf-8")
    )
    model = SpeechModel(configuration)
    model.warmup()
    return model


def logits(model: SpeechModel, prompts: Sequence[Tensor], caches: Sequence[DynamicCache]) -> Tensor:
    lengths = [cache.get_seq_length() for cache in caches]
    appended = prompts[0].shape[0]
    with torch.inference_mode():
        output: CausalLMOutputWithPast = model.language_model(
            inputs_embeds=torch.stack(tuple(prompts)),
            past_key_values=join_caches(caches, model.text_config),
            attention_mask=continuation_mask(lengths, appended, model.device),
            position_ids=continuation_positions(lengths, appended, model.device),
            use_cache=True,
            logits_to_keep=1,
        )
    return output.logits[:, -1].float().cpu()


def native_continuation_logits(model: SpeechModel, prefix: Tensor, continuation: Tensor) -> Tensor:
    cache = DynamicCache(config=model.text_config)
    model.language_model(
        inputs_embeds=prefix.unsqueeze(0), past_key_values=cache, use_cache=True, logits_to_keep=1
    )
    output: CausalLMOutputWithPast = model.language_model(
        inputs_embeds=continuation.unsqueeze(0),
        past_key_values=cache,
        attention_mask=continuation_mask([prefix.shape[0]], continuation.shape[0], model.device),
        position_ids=continuation_positions([prefix.shape[0]], continuation.shape[0], model.device),
        use_cache=True,
        logits_to_keep=1,
    )
    return output.logits[:, -1].float().cpu()


@pytest.mark.integration
def test_real_speech_multiturn_cache_matches_full_replay(configured_model: SpeechModel) -> None:
    model = configured_model
    with torch.inference_mode():
        speech, _ = model.encode_audio((bytes(32000), bytes(64000)))
        prefix = model.embed(USER_PREFIX)
        suffix = model.embed(ASSISTANT_SUFFIX)
        prompts = tuple(torch.cat((prefix, audio, suffix)) for audio in speech)
        first = model.forward_batch((prompts[0],), (DynamicCache(config=model.text_config),))
        accepted = model.embed((first.token_ids[0],))
        next_prompt = torch.cat((accepted, model.embed((248046, 198)), prompts[1]))
        cached = logits(model, (next_prompt,), first.caches)
        replay = logits(
            model,
            (torch.cat((prompts[0], next_prompt)),),
            (DynamicCache(config=model.text_config),),
        )
        native_logits = native_continuation_logits(model, prompts[0], next_prompt)
        torch.testing.assert_close(cached, native_logits, atol=0, rtol=0)
        # Native BF16 cached/replayed kernels differed by 0.198 on the shared 3090.
        torch.testing.assert_close(cached, replay, atol=0.25, rtol=0.03)
        torch.testing.assert_close(cached.argmax(-1), replay.argmax(-1), atol=0, rtol=0)


@pytest.mark.integration
def test_real_ragged_decode_batch_matches_serial(configured_model: SpeechModel) -> None:
    model = configured_model
    with torch.inference_mode():
        speech, _ = model.encode_audio((bytes(32000), bytes(64000)))
        prompts = tuple(
            torch.cat((model.embed(USER_PREFIX), audio, model.embed(ASSISTANT_SUFFIX)))
            for audio in speech
        )
        first = tuple(
            model.forward_batch((prompt,), (DynamicCache(config=model.text_config),))
            for prompt in prompts
        )
        tokens = tuple(model.embed((batch.token_ids[0],)) for batch in first)
        caches = tuple(batch.caches[0] for batch in first)
        serial = torch.cat(
            tuple(
                logits(model, (token,), (cache,))
                for token, cache in zip(tokens, caches, strict=True)
            )
        )
        batched = logits(model, tokens, caches)
        # Native BF16 batching on the 3090 also changes logits by up to 0.203.
        torch.testing.assert_close(batched, serial, atol=0.25, rtol=0.03)
        torch.testing.assert_close(batched.argmax(-1), serial.argmax(-1), atol=0, rtol=0)
        decoded = model.forward_batch(tokens, caches)
        assert tuple(cache.get_seq_length() for cache in decoded.caches) == tuple(
            cache.get_seq_length() + 1 for cache in caches
        )
        continued = tuple(model.embed((token_id,)) for token_id in serial.argmax(-1).tolist())
        serial_next = tuple(
            model.forward_batch((token,), (cache,))
            for token, cache in zip(tokens, caches, strict=True)
        )
        next_serial = torch.cat(
            tuple(
                logits(model, (token,), batch.caches)
                for token, batch in zip(continued, serial_next, strict=True)
            )
        )
        next_batched = logits(model, continued, decoded.caches)
        torch.testing.assert_close(next_batched, next_serial, atol=0.25, rtol=0.03)
        torch.testing.assert_close(next_batched.argmax(-1), next_serial.argmax(-1), atol=0, rtol=0)

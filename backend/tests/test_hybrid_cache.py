from collections.abc import Sequence

import pytest
import torch
from torch import Tensor
from transformers import Qwen3_5ForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from voice_worker.cache import (
    attention_layer,
    batch_workspace_bytes,
    cache_bytes,
    continuation_mask,
    continuation_positions,
    join_caches,
    recurrent_layer,
    reservation_bytes,
    split_cache,
)


@pytest.fixture(scope="module")
def tiny_model() -> Qwen3_5ForCausalLM:
    torch.set_num_threads(1)
    torch.manual_seed(11)
    configuration = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [2, 3, 3],
        },
        pad_token_id=0,
        eos_token_id=63,
    )
    model = Qwen3_5ForCausalLM(configuration).eval()
    model.config._attn_implementation = "sdpa"
    return model


def forward(
    model: Qwen3_5ForCausalLM, tokens: Sequence[Sequence[int]], caches: Sequence[DynamicCache]
) -> tuple[Tensor, tuple[DynamicCache, ...]]:
    lengths = [cache.get_seq_length() for cache in caches]
    appended = len(tokens[0])
    joined = join_caches(caches, model.config)
    with torch.inference_mode():
        output: CausalLMOutputWithPast = model(
            input_ids=torch.tensor(tokens),
            attention_mask=continuation_mask(lengths, appended, torch.device("cpu")),
            position_ids=continuation_positions(lengths, appended, torch.device("cpu")),
            past_key_values=joined,
            use_cache=True,
        )
    return output.logits.detach(), split_cache(joined, lengths, appended, model.config)


def fresh(model: Qwen3_5ForCausalLM, tokens: Sequence[int]) -> tuple[Tensor, DynamicCache]:
    logits, caches = forward(model, (tokens,), (DynamicCache(config=model.config),))
    return logits, caches[0]


def assert_cache_equal(first: DynamicCache, second: DynamicCache) -> None:
    assert first.get_seq_length() == second.get_seq_length()
    torch.testing.assert_close(
        attention_layer(first, 1).keys, attention_layer(second, 1).keys, atol=2e-6, rtol=2e-5
    )
    torch.testing.assert_close(
        attention_layer(first, 1).values, attention_layer(second, 1).values, atol=2e-6, rtol=2e-5
    )
    torch.testing.assert_close(
        recurrent_layer(first, 0).conv_states,
        recurrent_layer(second, 0).conv_states,
        atol=2e-6,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        recurrent_layer(first, 0).recurrent_states,
        recurrent_layer(second, 0).recurrent_states,
        atol=2e-6,
        rtol=2e-5,
    )


def test_ragged_decode_matches_independent_sessions(tiny_model: Qwen3_5ForCausalLM) -> None:
    _, first = fresh(tiny_model, (1, 2, 3, 4, 5))
    _, second = fresh(tiny_model, (6, 7, 8, 9, 10, 11, 12, 13))
    first_before = attention_layer(first, 1).keys.clone()
    linear_before = recurrent_layer(first, 0).conv_states.clone()
    expected_a, cache_a = forward(tiny_model, ((14,),), (first,))
    expected_b, cache_b = forward(tiny_model, ((15,),), (second,))
    actual, caches = forward(tiny_model, ((14,), (15,)), (first, second))
    torch.testing.assert_close(actual[0], expected_a[0], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual[1], expected_b[0], atol=2e-6, rtol=2e-5)
    assert_cache_equal(caches[0], cache_a[0])
    assert_cache_equal(caches[1], cache_b[0])
    torch.testing.assert_close(attention_layer(first, 1).keys, first_before)
    torch.testing.assert_close(recurrent_layer(first, 0).conv_states, linear_before)
    assert cache_bytes(caches[0], tiny_model.config) > 0


def test_multiturn_chunk_continuation_matches_full_replay(tiny_model: Qwen3_5ForCausalLM) -> None:
    prefix_a = (1, 2, 3, 4, 5)
    prefix_b = (6, 7, 8, 9, 10, 11, 12, 13)
    _, cache_a = fresh(tiny_model, prefix_a)
    _, cache_b = fresh(tiny_model, prefix_b)
    continuation_a = (20, 21, 22, 23, 24)
    continuation_b = (25, 26, 27, 28, 29)
    actual, caches = forward(tiny_model, (continuation_a, continuation_b), (cache_a, cache_b))
    expected_a, replay_a = fresh(tiny_model, prefix_a + continuation_a)
    expected_b, replay_b = fresh(tiny_model, prefix_b + continuation_b)
    torch.testing.assert_close(actual[0], expected_a[0, -5:], atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(actual[1], expected_b[0, -5:], atol=3e-6, rtol=3e-5)
    assert_cache_equal(caches[0], replay_a)
    assert_cache_equal(caches[1], replay_b)


def test_batch_membership_can_change_and_states_do_not_alias(
    tiny_model: Qwen3_5ForCausalLM,
) -> None:
    _, first = fresh(tiny_model, (1, 2, 3, 4, 5))
    _, second = fresh(tiny_model, (6, 7, 8, 9, 10))
    _, caches = forward(tiny_model, ((11,), (12,)), (first, second))
    _, newcomer = fresh(tiny_model, (20, 21, 22, 23))
    expected, _ = forward(tiny_model, ((13,),), (caches[0],))
    actual, next_caches = forward(tiny_model, ((13,), (24,)), (caches[0], newcomer))
    torch.testing.assert_close(actual[0], expected[0], atol=2e-6, rtol=2e-5)
    other_state = recurrent_layer(next_caches[1], 0).conv_states.clone()
    recurrent_layer(next_caches[0], 0).conv_states.zero_()
    torch.testing.assert_close(recurrent_layer(next_caches[1], 0).conv_states, other_state)


def test_workspace_accounts_for_ragged_attention_padding_and_appended_tokens(
    tiny_model: Qwen3_5ForCausalLM,
) -> None:
    configuration = tiny_model.config
    workspace = batch_workspace_bytes(configuration, (1000, 1), 5)
    assert workspace == 4 * reservation_bytes(configuration, 1005)
    assert workspace > 2 * sum(reservation_bytes(configuration, length) for length in (1000, 1))

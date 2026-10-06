"""Join and split every state of Qwen's pinned hybrid DynamicCache."""

from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor
from torch.nn import functional
from transformers.cache_utils import DynamicCache, DynamicLayer, LinearAttentionLayer
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig


def attention_layer(cache: DynamicCache, index: int) -> DynamicLayer:
    return cast(DynamicLayer, cache.layers[index])


def recurrent_layer(cache: DynamicCache, index: int) -> LinearAttentionLayer:
    return cast(LinearAttentionLayer, cache.layers[index])


def join_caches(caches: Sequence[DynamicCache], config: Qwen3_5TextConfig) -> DynamicCache:
    """Left-pad only attention KV; recurrent/convolution state has no padding positions."""
    if not caches:
        raise ValueError("Cannot join an empty cache batch")
    joined = DynamicCache(config=config)
    lengths = [cache.get_seq_length() for cache in caches]
    if max(lengths) == 0:
        return joined
    if min(lengths) == 0:
        raise ValueError("Fresh and populated caches require separate prefill groups")
    maximum = max(lengths)
    for index, layer_type in enumerate(config.layer_types):
        match layer_type:
            case "full_attention":
                keys = []
                values = []
                for cache, length in zip(caches, lengths, strict=True):
                    layer = attention_layer(cache, index)
                    assert layer.keys is not None and layer.values is not None
                    keys.append(functional.pad(layer.keys, (0, 0, maximum - length, 0)))
                    values.append(functional.pad(layer.values, (0, 0, maximum - length, 0)))
                joined.update(torch.cat(keys), torch.cat(values), index)
            case "linear_attention":
                convolution = []
                recurrent = []
                for cache in caches:
                    layer = recurrent_layer(cache, index)
                    assert layer.conv_states is not None and layer.recurrent_states is not None
                    convolution.append(layer.conv_states)
                    recurrent.append(layer.recurrent_states)
                joined.update_conv_state(torch.cat(convolution), index)
                joined.update_recurrent_state(torch.cat(recurrent), index)
            case _:
                raise ValueError(f"Unsupported Qwen cache layer: {layer_type}")
    return joined


def split_cache(
    batched: DynamicCache,
    lengths: Sequence[int],
    appended_tokens: int,
    config: Qwen3_5TextConfig,
) -> tuple[DynamicCache, ...]:
    maximum = max(lengths)
    separated = tuple(DynamicCache(config=config) for _ in lengths)
    for index, layer_type in enumerate(config.layer_types):
        match layer_type:
            case "full_attention":
                layer = attention_layer(batched, index)
                assert layer.keys is not None and layer.values is not None
                for row, (cache, length) in enumerate(zip(separated, lengths, strict=True)):
                    start = maximum - length
                    stop = maximum + appended_tokens
                    cache.update(
                        layer.keys[row : row + 1, :, start:stop].clone(),
                        layer.values[row : row + 1, :, start:stop].clone(),
                        index,
                    )
            case "linear_attention":
                layer = recurrent_layer(batched, index)
                assert layer.conv_states is not None and layer.recurrent_states is not None
                for row, cache in enumerate(separated):
                    cache.update_conv_state(layer.conv_states[row : row + 1].clone(), index)
                    cache.update_recurrent_state(
                        layer.recurrent_states[row : row + 1].clone(), index
                    )
            case _:
                raise ValueError(f"Unsupported Qwen cache layer: {layer_type}")
    return separated


def continuation_mask(lengths: Sequence[int], appended_tokens: int, device: torch.device) -> Tensor:
    maximum = max(lengths)
    positions = torch.arange(maximum + appended_tokens, device=device).unsqueeze(0)
    starts = maximum - torch.tensor(lengths, device=device).unsqueeze(1)
    return (positions >= starts).to(torch.long)


def continuation_positions(
    lengths: Sequence[int], appended_tokens: int, device: torch.device
) -> Tensor:
    return torch.tensor(lengths, device=device).unsqueeze(1) + torch.arange(
        appended_tokens, device=device
    ).unsqueeze(0)


def cache_bytes(cache: DynamicCache, config: Qwen3_5TextConfig) -> int:
    total = 0
    for index, layer_type in enumerate(config.layer_types):
        match layer_type:
            case "full_attention":
                layer = attention_layer(cache, index)
                tensors = (layer.keys, layer.values)
            case "linear_attention":
                layer = recurrent_layer(cache, index)
                tensors = (layer.conv_states, layer.recurrent_states)
            case _:
                raise ValueError(f"Unsupported Qwen cache layer: {layer_type}")
        total += sum(
            tensor.numel() * tensor.element_size() for tensor in tensors if tensor is not None
        )
    return total


def reservation_bytes(config: Qwen3_5TextConfig, context_tokens: int) -> int:
    attention = config.layer_types.count("full_attention")
    linear = config.layer_types.count("linear_attention")
    key_value = attention * 2 * config.num_key_value_heads * config.head_dim * 2 * context_tokens
    convolution = (
        linear
        * (
            2 * config.linear_num_key_heads * config.linear_key_head_dim
            + config.linear_num_value_heads * config.linear_value_head_dim
        )
        * config.linear_conv_kernel_dim
        * 2
    )
    recurrent = (
        linear
        * config.linear_num_value_heads
        * config.linear_key_head_dim
        * config.linear_value_head_dim
        * 4
    )
    return key_value + convolution + recurrent

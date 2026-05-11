# Copyright 2025 The HuggingFace Team. All rights reserved.
# Modified for agent-factory-3: extracted from mxfp4_bf16_dequant.py into its
# own module so the quantizer (always-on) and the routing-replay activator
# (optional) live in clearly separated files.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Routing-replay activator for gpt-oss MoE.

Routing replay reuses the expert indices captured during rollout (by sglang)
when re-running the model in training, so the train-time MoE selects exactly
the same experts the inference-time MoE did. Without this, expert selection
drifts every weight update and the imitation gradient is biased.

This module's `enable_routing_replay()` is the public switch. The MLP-side
machinery that actually consumes per-token expert indices (the patched
`mlp_forward` and the `routing_replay()` helper) is installed by the mxfp4
quantizer in :mod:`agent_factory_3.models.gpt_oss.quantization`; this
function only needs to wire `routing_indices` through `GptOssModel.forward`
and `GptOssDecoderLayer.forward` so they reach the MLP.
"""
import torch
from transformers.utils import logging

logger = logging.get_logger(__name__)


def enable_routing_replay(model):
    """Patch GptOssModel and GptOssDecoderLayer to support routing_indices.

    After calling this, model(input_ids, routing_indices=[B,S,L,K]) will:
      1. GptOssModel.forward pops routing_indices, slices per-layer [B,S,K]
      2. GptOssDecoderLayer.forward passes router_indices to MLP
      3. mlp_forward uses routing_replay() instead of top-k

    Call AFTER model loading (since mlp_forward is already MethodType-bound).
    """
    from transformers.models.gpt_oss.modeling_gpt_oss import (
        GptOssModel, GptOssDecoderLayer,
    )

    if getattr(GptOssModel, "_routing_replay_enabled", False):
        logger.info("Routing replay already enabled")
        return

    # --- Patch DecoderLayer.forward to accept router_indices ---
    _orig_decoder_forward = GptOssDecoderLayer.forward

    def _decoder_forward_with_replay(
        self, hidden_states, attention_mask=None, position_ids=None,
        past_key_values=None, use_cache=False, cache_position=None,
        position_embeddings=None, router_indices=None, **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            use_cache=use_cache, cache_position=cache_position,
            position_embeddings=position_embeddings, **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, _ = self.mlp(hidden_states, router_indices=router_indices)
        hidden_states = residual + hidden_states
        return hidden_states

    GptOssDecoderLayer.forward = _decoder_forward_with_replay

    # --- Patch GptOssModel.forward to accept routing_indices [B,S,L,K] ---
    _orig_model_forward = GptOssModel.forward

    from transformers.cache_utils import DynamicCache
    from transformers.modeling_outputs import MoeModelOutputWithPast

    def _model_forward_with_replay(
        self, input_ids=None, attention_mask=None, position_ids=None,
        past_key_values=None, inputs_embeds=None, use_cache=None,
        cache_position=None, **kwargs,
    ):
        routing_indices = kwargs.pop("routing_indices", None)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if routing_indices is not None:
            B, S, L, K = inputs_embeds.shape[0], inputs_embeds.shape[1], len(self.layers), self.config.num_experts_per_tok
            assert routing_indices.shape == (B, S, L, K), \
                f"routing_indices shape {routing_indices.shape} != expected ({B},{S},{L},{K})"

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
            mask_kwargs = dict(
                config=self.config, input_embeds=inputs_embeds, attention_mask=attention_mask,
                cache_position=cache_position, past_key_values=past_key_values, position_ids=position_ids,
            )
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_ri = routing_indices[:, :, layer_idx, :] if routing_indices is not None else None
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids, past_key_values=past_key_values,
                use_cache=use_cache, cache_position=cache_position,
                position_embeddings=position_embeddings,
                router_indices=layer_ri, **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)

    GptOssModel.forward = _model_forward_with_replay

    GptOssModel._routing_replay_enabled = True
    logger.info("Routing replay enabled for GptOssModel + GptOssDecoderLayer")

"""RoLAForCausalLM — HF causal LM with RoLA sequence mixers. Mirrors
fla.models.gla.modeling_gla (same backbone: RMSNorm + mixer + GatedMLP blocks, fused
losses, FLA cache/generation), with the mixer swapped for RoLA. Training + loglikelihood
eval (no var-len/cu_seqlens or recurrent-cache decode yet — not needed for the standard
zero-shot table, which is loglikelihood-based)."""
from __future__ import annotations

import math
import os
import sys
import warnings
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.deprecation import deprecate_kwarg

from fla.models.modeling_layers import GradientCheckpointingLayer
from fla.models.utils import Cache, FLAGenerationMixin
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss, RMSNorm
from fla.modules import GatedMLP
from fla.modules.l2warp import l2_warp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rola import RoLA, rola_instance                       # noqa: E402
from rola_hf.configuration_rola import RoLAConfig           # noqa: E402

if TYPE_CHECKING:
    from typing import Unpack


class RoLAttention(nn.Module):
    """FLA-layer-style wrapper around rola.RoLA: forward returns the FLA 3-tuple
    (output, attentions=None, past_key_values). Training/loglikelihood only."""

    def __init__(self, config: RoLAConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        kw = rola_instance(config.rola_instance, d_qk=config.d_qk, d_v=config.d_v,
                           num_chunks=config.num_states, n_heads=config.num_heads)
        self.rola = RoLA(d_model=config.hidden_size, **kw)

    def forward(self, hidden_states, attention_mask=None, past_key_values=None,
                use_cache=False, output_attentions=False, **kwargs):
        # attention_mask ignored: causal linear attn + right-padding makes it safe for
        # loglikelihood; flame packs full sequences. No recurrent cache (train/ll only).
        return self.rola(hidden_states), None, past_key_values


class RoLABlock(GradientCheckpointingLayer):

    def __init__(self, config: RoLAConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.attn = RoLAttention(config, layer_idx)
        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

    def forward(self, hidden_states, attention_mask=None, past_key_values=None,
                use_cache=False, output_attentions=False, **kwargs):
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states, attention_mask=attention_mask,
            past_key_values=past_key_values, use_cache=use_cache,
            output_attentions=output_attentions, **kwargs)
        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states
        return (hidden_states, attentions, past_key_values)


class RoLAPreTrainedModel(PreTrainedModel):

    config_class = RoLAConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['RoLABlock']
    _supports_cache_class = True

    def _init_weights(self, module, prenorm_residual_strategy=None, num_residuals_per_layer=2):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()


class RoLAModel(RoLAPreTrainedModel):

    def __init__(self, config: RoLAConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([RoLABlock(config, i) for i in range(config.num_hidden_layers)])
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None,
                past_key_values=None, use_cache=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, **kwargs):
        output_attentions = False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify exactly one of input_ids / inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states, _, past_key_values = layer(
                hidden_states, attention_mask=attention_mask, past_key_values=past_key_values,
                use_cache=use_cache, output_attentions=False, **kwargs)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if not return_dict:
            return tuple(i for i in [hidden_states, past_key_values, all_hidden_states] if i is not None)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states,
                                       past_key_values=past_key_values,
                                       hidden_states=all_hidden_states)


class RoLAForCausalLM(RoLAPreTrainedModel, FLAGenerationMixin):

    # Untied (tie_word_embeddings=False) -> no tied weights. Empty DICT, not a list:
    # transformers 5.x iterates this as a mapping (FLA's legacy list form breaks save).
    _tied_weights_keys = {}

    def __init__(self, config):
        super().__init__(config)
        self.model = RoLAModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model

    def set_decoder(self, decoder):
        self.model = decoder

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None,
                past_key_values=None, labels=None, use_cache=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, logits_to_keep=0, **kwargs):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             inputs_embeds=inputs_embeds, past_key_values=past_key_values,
                             use_cache=use_cache, output_hidden_states=output_hidden_states,
                             return_dict=return_dict, **kwargs)
        hidden_states = outputs[0]
        loss, logits = None, None
        if not self.config.fuse_linear_cross_entropy or labels is None:
            logits = self.lm_head(hidden_states if logits_to_keep is None else hidden_states[:, -logits_to_keep:])
        if labels is not None:
            if getattr(self, 'criterion', None) is None:
                if self.config.fuse_linear_cross_entropy:
                    criterion = FusedLinearCrossEntropyLoss(use_l2warp=self.config.use_l2warp)
                elif self.config.fuse_cross_entropy:
                    criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion
            labels = labels.to(hidden_states.device)
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), 1)
            if self.config.fuse_linear_cross_entropy:
                loss = criterion(hidden_states, labels, self.lm_head.weight, self.lm_head.bias)
            else:
                loss = criterion(logits.view(labels.numel(), -1), labels.view(-1))
                loss = l2_warp(loss, logits) if self.config.use_l2warp else loss
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithPast(loss=loss, logits=logits,
                                      past_key_values=outputs.past_key_values,
                                      hidden_states=outputs.hidden_states)

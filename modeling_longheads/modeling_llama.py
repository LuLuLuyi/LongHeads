# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput, BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast, dataclass
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_llama import LlamaConfig
# from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func, flash_attn_with_kvcache
from flash_attn.modules.mha import FlashSelfAttention

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


def _pad_to_chunk_size(tensor, window_size, past_key_value_length=0, dim=-2, pad_value=0):
    assert dim < 0  # only accept ``dim'' index in a reverse manner
    seqlen = int(tensor.shape[dim])
    seqlen += past_key_value_length
    m = seqlen / window_size
    if m.is_integer():
        return tensor
    remainder = math.ceil(m) * window_size - seqlen  # get padding size
    pad_offset = (0,) * (-1 - dim) * 2  #
    padded_res = F.pad(tensor, (*pad_offset, 0, remainder), value=pad_value)
    return padded_res


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return (self.weight * hidden_states).to(input_dtype)

@dataclass
class BaseModelOutputWithPast(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_queries: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_global_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        # print("max_position_embeddings", max_position_embeddings)
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(x.dtype), persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(x.dtype), persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin) # q: [bs, nh, seq_len, dim]
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def apply_rotary_pos_emb_for_relative_keys(k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return k_embed

def apply_rotary_pos_emb_for_relative_query(q, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin) # q: [bs, nh, seq_len, dim]
    return q_embed


class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

        # longheads config
        self.cpu_offload = config.cpu_offload
        self.batch_encoding = config.batch_encoding
        self.encoding_batch_size = config.encoding_batch_size
        self.atten_length = config.atten_length
        self.begin_selective_length = config.begin_selective_length
        self.window_size = config.window_size
        self.attn_scale = self.head_dim ** -0.5
        self.causal_attn = FlashSelfAttention(causal=True, softmax_scale=self.head_dim ** -0.5)
        self.attn = FlashSelfAttention(causal=False, softmax_scale=self.head_dim ** -0.5)
        

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _nonoverlap_window_1d_partition(self, x):
        # cut the chunk along the seq_len dimension
        return rearrange(x, '... (g w) d -> ... g w d', w=self.window_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,  # [bsz, 1, tgt_seq_len, src_seq_len]
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        past_query: Optional[List[torch.FloatTensor]] = None,
        past_global_key_value: Optional[List[torch.FloatTensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # check valid hyper param
        assert attention_mask is not None
        assert self.atten_length % 256 == 0
        past_key_value_length = 0
        if past_key_value is not None:
            past_key_value_length = past_key_value[0].shape[-2]
        sequence_length = position_ids[0,-1] + 1
        ori_len = hidden_states.shape[-2]
        
        atten_length = self.atten_length
        begin_selective_length = self.begin_selective_length
        if ori_len > 1:
            atten_chunk_num = atten_length // self.window_size
            
            hidden_states = _pad_to_chunk_size(hidden_states, self.window_size,
                                            dim=-2, past_key_value_length=past_key_value_length)
            attention_mask = _pad_to_chunk_size(attention_mask, self.window_size,
                                            dim=-1, past_key_value_length=past_key_value_length)
            local_position_ids = _pad_to_chunk_size(position_ids, self.window_size,
                                            dim=-1, past_key_value_length=past_key_value_length)
            local_position_ids = local_position_ids % self.window_size  # apply rope within chunk
            bsz, q_len, _ = hidden_states.size()
            attention_mask = 1 - attention_mask

            query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                kv_seq_len += past_key_value_length
            # decide where to add the rope, inside or outside chunk
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states_ori = query_states
            key_states_ori = key_states
            if sequence_length > begin_selective_length:
                normal_query_states, normal_key_states = apply_rotary_pos_emb(query_states[:,:,:begin_selective_length], key_states[:,:,:begin_selective_length], cos, sin, position_ids[:,:begin_selective_length])
                normal_value_states = value_states[:,:,:begin_selective_length]
            else:
                normal_query_states, normal_key_states = apply_rotary_pos_emb(query_states[:,:,:sequence_length], key_states[:,:,:sequence_length], cos, sin, position_ids[:,:sequence_length])
                normal_value_states = value_states[:,:,:sequence_length]
            local_query_states, local_key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, local_position_ids)
            # [bsz, nh, t, hd]

            if past_key_value is not None:
                # reuse k, v, self_attention
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
            chunk_num = key_states.shape[-2] / self.window_size
            assert chunk_num.is_integer()
            
            if self.cpu_offload:
                past_key_value = (key_states[:,:,:ori_len].to('cpu'), value_states[:,:,:ori_len].to('cpu')) if use_cache else None
                past_query = (query_states[:,:,:ori_len].to('cpu'),) if use_cache else None
            else:
                past_key_value = (key_states[:,:,:ori_len], value_states[:,:,:ori_len]) if use_cache else None
                past_query = (query_states[:,:,:ori_len],) if use_cache else None

            # partition to get global chunk
            # [bsz, nh, t, hd] = > [bsz, nh, g, w, hd]
            rf_w_q = self._nonoverlap_window_1d_partition(local_query_states)
            rf_w_k = self._nonoverlap_window_1d_partition(local_key_states)
            rf_w_v = self._nonoverlap_window_1d_partition(value_states)

            rf_w_k_ori = self._nonoverlap_window_1d_partition(key_states_ori)

            # compute global feature
            global_chunk_mask = attention_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool)  # [bsz, 1, t, 1]
            rf_w_mask = self._nonoverlap_window_1d_partition(global_chunk_mask)  # [bsz, 1, g, w, 1]
            rf_w_q = rf_w_q.masked_fill(rf_w_mask, 0.)
            rf_w_k = rf_w_k.masked_fill(rf_w_mask, 0.)
            rf_w_v = rf_w_v.masked_fill(rf_w_mask, 0.)
            rf_w_k_ori = rf_w_k_ori.masked_fill(rf_w_mask, 0.)

            
            b, h, g, w, d = rf_w_q.shape
            attn_rf_w_q = rearrange(rf_w_q, "b h g w d -> b (h g) w d") 
            attn_rf_w_k = rearrange(rf_w_k, "b h g w d -> b (h g) w d") 
            attn_rf_w_v = rearrange(rf_w_v, "b h g w d -> b (h g) w d") 
            qkv = torch.stack([attn_rf_w_q, attn_rf_w_k, attn_rf_w_v], dim=2)
            
            chunk_features = self.attn(qkv.transpose(1, 3))
            chunk_features = chunk_features.transpose(1, 2)
            chunk_features = rearrange(chunk_features, "b (h g) w d -> b h g w d", h=h)  
            chunk_features_value = chunk_features.mean(dim=-2)  # [bsz, nh, g, hd]

            chunk_features_key = F.scaled_dot_product_attention(query=chunk_features_value.unsqueeze(-2),
                                                            key=rf_w_k,
                                                            value=rf_w_k).squeeze(-2) # [bsz, nh, g, 1, hd]
            if self.cpu_offload:
                past_global_key_value = (chunk_features_key[:,:,:-1].to('cpu'), chunk_features_value[:,:,:-1].to('cpu')) if use_cache else None
            else:
                past_global_key_value = (chunk_features_key[:,:,:-1], chunk_features_value[:,:,:-1]) if use_cache else None
            past_cache = (past_key_value, past_query, past_global_key_value)
            if sequence_length >= begin_selective_length:
                selective_query_ori = query_states_ori[:,:,begin_selective_length:]
                selective_query_position_ids = torch.arange(0, selective_query_ori.size(-2), dtype=position_ids.dtype).unsqueeze(0) % self.window_size + (atten_chunk_num - 1) * self.window_size
                selective_query = apply_rotary_pos_emb_for_relative_query(selective_query_ori, cos, sin, selective_query_position_ids)
                
                global_qk = torch.einsum('bhsd,bhgd->bhsg', selective_query_ori, chunk_features_key)
                global_qk = rearrange(global_qk, "b h (g w) d -> b h g w d", h=h, w=self.window_size)
                global_qk = global_qk.transpose(-2, -3)
                global_qk_mask = torch.ones_like(global_qk, dtype=query_states.dtype).bool().triu(begin_selective_length//self.window_size+1)
                global_qk_select = torch.zeros_like(global_qk, dtype=query_states.dtype).bool()
                global_qk_select[:,:,:,:,0] = True
                for idx in range(global_qk.size(-2)):
                    global_qk_select[:,:,:,idx, idx + begin_selective_length//self.window_size-1: idx + begin_selective_length//self.window_size+1] = True
                mask_value = torch.finfo(global_qk.dtype).min
                select_value = torch.finfo(global_qk.dtype).max
                global_qk = global_qk.masked_fill(global_qk_mask, mask_value)
                global_qk = global_qk.masked_fill(global_qk_select, select_value)
                global_qk = global_qk.transpose(-2, -3)
                global_qk = rearrange(global_qk, "b h g w d -> b h (g w) d", h=h, w=self.window_size)
                
                atten_chunks = torch.topk(global_qk, k=atten_chunk_num, dim=-1, largest=True)[1]
                sorted_atten_chunks, indices = torch.sort(atten_chunks, dim=-1)
                
                selective_query = selective_query[:,:, :ori_len - begin_selective_length]
                # allocate space for atten output
                select_atten_output = torch.zeros_like(selective_query, dtype=query_states.dtype)
                
                if self.batch_encoding == True:
                    # batch encoding implementation
                    # allocate space for selected kv list
                    selective_sequence_length = selective_query.size(2)
    
                    encoding_batch_size = self.encoding_batch_size
                    batch_num = selective_sequence_length // encoding_batch_size
                    last_batch_size = selective_sequence_length % encoding_batch_size
                    
                    
                    for batch_idx in range(batch_num):
                        st = batch_idx * encoding_batch_size
                        ed = (batch_idx + 1) * encoding_batch_size
                        
                        # init selected kv cache for every query in batch, position information is prepared in selected qkv cache
                        selected_keys_cache = torch.zeros([encoding_batch_size, self.num_heads, atten_length, self.head_dim], dtype=query_states.dtype, device=query_states.device)
                        selected_values_cache = torch.zeros([encoding_batch_size, self.num_heads, atten_length, self.head_dim], dtype=query_states.dtype, device=query_states.device)
                        cache_seqlens = torch.zeros([encoding_batch_size], dtype=torch.int32, device=query_states.device)
                        for relative_idx, query_idx in enumerate(range(st, ed)):
                            relative_position =  (atten_chunk_num - 1) * self.window_size + query_idx % self.window_size
                            cache_seqlens[relative_idx] = relative_position
                            relative_position_ids = torch.arange(0, relative_position, dtype=position_ids.dtype).unsqueeze(0)
                            selected_keys = torch.gather(rf_w_k_ori.transpose(-1,-3),-1, sorted_atten_chunks[:,:,query_idx].unsqueeze(-2).unsqueeze(-2).repeat(1, 1,self.head_dim,self.window_size,1)).transpose(-1,-3)
                            selected_values = torch.gather(rf_w_v.transpose(-1,-3),-1, sorted_atten_chunks[:,:,query_idx].unsqueeze(-2).unsqueeze(-2).repeat(1, 1,self.head_dim,self.window_size,1)).transpose(-1,-3)
                            selected_keys = rearrange(selected_keys, "b h g w d -> b h (g w) d")
                            selected_values = rearrange(selected_values, "b h g w d -> b h (g w) d")
                            # add rope to selected keys
                            selected_keys[:,:,:relative_position] = apply_rotary_pos_emb_for_relative_keys(selected_keys[:,:,:relative_position], cos, sin, relative_position_ids)
                            selected_keys_cache[relative_idx,:,:relative_position] = selected_keys[:,:,:relative_position]
                            selected_values_cache[relative_idx,:,:relative_position] = selected_values[:,:,:relative_position]
                        batch_queries = selective_query[:, :, st : ed].transpose(0, 2).transpose(1, 2)
                        batch_keys_list = selected_keys_cache.transpose(1, 2)
                        batch_values_list = selected_values_cache.transpose(1, 2)
                        batch_atten_output = flash_attn_with_kvcache(batch_queries, k_cache=batch_keys_list, v_cache=batch_values_list, causal=False, cache_seqlens=cache_seqlens)
                        select_atten_output[:, :, st : ed] = batch_atten_output.transpose(1, 2).transpose(0, 2)
                    
                    if last_batch_size != 0:
                        st = batch_num * encoding_batch_size
                        ed = st + last_batch_size
                        assert ed==selective_sequence_length
                        
                        selected_keys_cache = torch.zeros([last_batch_size, self.num_heads, atten_length, self.head_dim], dtype=query_states.dtype, device=query_states.device)
                        selected_values_cache = torch.zeros([last_batch_size, self.num_heads, atten_length, self.head_dim], dtype=query_states.dtype, device=query_states.device)
                        cache_seqlens = torch.zeros([last_batch_size], dtype=torch.int32, device=query_states.device)
                        for relative_idx, query_idx in enumerate(range(st, ed)):
                            relative_position =  (atten_chunk_num - 1) * self.window_size + query_idx % self.window_size
                            relative_position_ids = torch.arange(0, relative_position, dtype=position_ids.dtype).unsqueeze(0)
                            cache_seqlens[relative_idx] = relative_position
                            selected_keys = torch.gather(rf_w_k_ori.transpose(-1,-3),-1, sorted_atten_chunks[:,:,query_idx].unsqueeze(-2).unsqueeze(-2).repeat(1, 1,self.head_dim,self.window_size,1)).transpose(-1,-3)
                            selected_values = torch.gather(rf_w_v.transpose(-1,-3),-1, sorted_atten_chunks[:,:,query_idx].unsqueeze(-2).unsqueeze(-2).repeat(1, 1,self.head_dim,self.window_size,1)).transpose(-1,-3)
                            selected_keys = rearrange(selected_keys, "b h g w d -> b h (g w) d")
                            selected_values = rearrange(selected_values, "b h g w d -> b h (g w) d")
                            # add rope to selected keys
                            selected_keys[:,:,:relative_position] = apply_rotary_pos_emb_for_relative_keys(selected_keys[:,:,:relative_position], cos, sin, relative_position_ids)
                            selected_keys_cache[relative_idx,:,:relative_position] = selected_keys[:,:,:relative_position]
                            selected_values_cache[relative_idx,:,:relative_position] = selected_values[:,:,:relative_position]
                        batch_queries = selective_query[:, :, st : ed].transpose(0, 2).transpose(1, 2)
                        batch_keys_list = selected_keys_cache.transpose(1, 2)
                        batch_values_list = selected_values_cache.transpose(1, 2)
                        batch_atten_output = flash_attn_with_kvcache(batch_queries, k_cache=batch_keys_list, v_cache=batch_values_list, causal=False, cache_seqlens=cache_seqlens)
                        select_atten_output[:, :, st : ed] = batch_atten_output.transpose(1, 2).transpose(0, 2)
                else:
                    # sequencial encoding implementation
                    for query_idx in range(selective_query.size(2)):
                        current_query = selective_query[:,:,query_idx]
                        relative_position =  (atten_chunk_num - 1) * self.window_size + query_idx % self.window_size
                        relative_position_ids = torch.arange(0, relative_position, dtype=position_ids.dtype).unsqueeze(0)
                        # TODO: use mask for every query and kv to speed up
                        selected_keys = torch.gather(rf_w_k_ori.transpose(-1,-3),-1, sorted_atten_chunks[:,:,query_idx].unsqueeze(-2).unsqueeze(-2).repeat(1, 1,self.head_dim,self.window_size,1)).transpose(-1,-3)
                        selected_values = torch.gather(rf_w_v.transpose(-1,-3),-1, sorted_atten_chunks[:,:,query_idx].unsqueeze(-2).unsqueeze(-2).repeat(1, 1,self.head_dim,self.window_size,1)).transpose(-1,-3)
                        selected_keys = rearrange(selected_keys, "b h g w d -> b h (g w) d")
                        selected_values = rearrange(selected_values, "b h g w d -> b h (g w) d")
                        selected_keys = selected_keys[:,:,:relative_position]
                        selected_values = selected_values[:,:,:relative_position]
                        # add rope to selected keys
                        selected_keys = apply_rotary_pos_emb_for_relative_keys(selected_keys, cos, sin, relative_position_ids)
                        # implemenation with flash-attention 2
                        current_query = current_query.unsqueeze(1)
                        selected_keys = selected_keys.transpose(1, 2)
                        selected_values = selected_values.transpose(1, 2)
                        current_attn_output = flash_attn_with_kvcache(current_query, k_cache=selected_keys, v_cache=selected_values, softmax_scale=self.attn_scale, causal=False)
                        current_attn_output = current_attn_output.squeeze(1)
                        
                        select_atten_output[:,:,query_idx] = current_attn_output
            
            
            # calculate previous feature with flash-attention 2
            normal_qkv = torch.stack([normal_query_states, normal_key_states, normal_value_states], dim=2)
            normal_qkv = normal_qkv.transpose(1, 3)
            normal_attn_output = flash_attn_qkvpacked_func(normal_qkv, softmax_scale=self.attn_scale, causal=True)
            normal_attn_output = normal_attn_output.transpose(1, 2)
            
            if sequence_length >= begin_selective_length:
                attn_output = torch.cat([normal_attn_output, select_atten_output], dim=-2)
            else:
                attn_output = normal_attn_output
            attn_output = attn_output[:,:, past_key_value_length:past_key_value_length+ori_len].transpose(1, 2).reshape(bsz, ori_len, -1)
            attn_output = self.o_proj(attn_output)
            if not output_attentions:
                attn_weights = None
            return attn_output, attn_weights, past_cache
        elif sequence_length >= begin_selective_length:
            atten_chunk_num = atten_length // self.window_size
            bsz, q_len, _ = hidden_states.size()
            query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                kv_seq_len += past_key_value_length
                # decide where to add the rope, inside or outside chunk
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            if past_key_value is not None:
                # reuse k, v, self_attention
                if self.cpu_offload:
                    query_states = query_states.to('cpu')
                    key_states = key_states.to('cpu')
                    value_states = value_states.to('cpu')
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
            # update past qkv
            past_key_value = (key_states, value_states) if use_cache else None
            past_query = (torch.cat([past_query[0], query_states], dim=2),)
            
            selection_length = sequence_length - sequence_length % self.window_size
            last_length = sequence_length % self.window_size
            last_keys = key_states[:,:,-last_length:] if last_length > 0 else None
            last_values = value_states[:,:,-last_length:] if last_length > 0 else None
            
            rf_w_k = self._nonoverlap_window_1d_partition(key_states[:,:, :selection_length])
            rf_w_v = self._nonoverlap_window_1d_partition(value_states[:,:, :selection_length])
            b, h, g, w, d = rf_w_k.shape
            
            relative_position =  (atten_chunk_num - 1) * self.window_size + sequence_length % self.window_size
            relative_position_ids = torch.arange(0, relative_position, dtype=position_ids.dtype).unsqueeze(0)
            current_query_position_ids = relative_position_ids[:, -1].unsqueeze(0)
            
            chunk_features_key = past_global_key_value[0] if past_global_key_value is not None else None
            current_query_ori = query_states[:,:,ori_len-1]
            
            if chunk_features_key is None or chunk_features_key.size()[-2] < 1:
                position_ids_for_local_atten = torch.arange(0, selection_length, dtype=position_ids.dtype).unsqueeze(0)
                position_ids_for_local_atten = position_ids_for_local_atten % self.window_size
                all_query_states_for_local_atten = past_query[0][:,:, :selection_length]
                all_key_states_for_local_atten = key_states[:,:, :selection_length]
                if self.cpu_offload:
                    all_query_states_for_local_atten = all_query_states_for_local_atten.to(hidden_states.device)
                    all_key_states_for_local_atten = all_key_states_for_local_atten.to(hidden_states.device)
                    
                all_query_states_for_local_atten, all_key_states_for_local_atten = apply_rotary_pos_emb(all_query_states_for_local_atten, all_key_states_for_local_atten, cos, sin, position_ids_for_local_atten)
                rf_w_q_rope = self._nonoverlap_window_1d_partition(all_query_states_for_local_atten)
                rf_w_k_rope = self._nonoverlap_window_1d_partition(all_key_states_for_local_atten)
                
                attn_rf_w_q = rearrange(rf_w_q_rope, "b h g w d -> b (h g) w d") 
                attn_rf_w_k = rearrange(rf_w_k_rope, "b h g w d -> b (h g) w d") 
                if self.cpu_offload:
                    attn_rf_w_v = rearrange(rf_w_v, "b h g w d -> b (h g) w d")
                    attn_rf_w_v = attn_rf_w_v.to(hidden_states.device)
                else:
                    attn_rf_w_v = rearrange(rf_w_v, "b h g w d -> b (h g) w d") 
                qkv = torch.stack([attn_rf_w_q, attn_rf_w_k, attn_rf_w_v], dim=2)
                
                chunk_features = self.attn(qkv.transpose(1, 3))
                chunk_features = chunk_features.transpose(1, 2)
                chunk_features = rearrange(chunk_features, "b (h g) w d -> b h g w d", h=h)  
                chunk_features_value = chunk_features.mean(dim=-2)  # [bsz, nh, g, hd]

                chunk_features_key = F.scaled_dot_product_attention(query=chunk_features_value.unsqueeze(-2), key=rf_w_k_rope, value=rf_w_k_rope).squeeze(-2) # [bsz, nh, g, 1, hd]
                
                past_global_key_value = (chunk_features_key, chunk_features_value) if use_cache else None
                
            elif g > chunk_features_key.size()[-2]:
                position_ids_for_local_atten = torch.arange(0, self.window_size, dtype=position_ids.dtype).unsqueeze(0)
                all_query_states_for_local_atten = past_query[0][:,:, selection_length - self.window_size:selection_length]
                all_key_states_for_local_atten = key_states[:,:, selection_length - self.window_size:selection_length]
                if self.cpu_offload:
                    all_query_states_for_local_atten = all_query_states_for_local_atten.to(hidden_states.device)
                    all_key_states_for_local_atten = all_key_states_for_local_atten.to(hidden_states.device)
                    
                all_query_states_for_local_atten, all_key_states_for_local_atten = apply_rotary_pos_emb(all_query_states_for_local_atten, all_key_states_for_local_atten, cos, sin, position_ids_for_local_atten)
                rf_w_q_rope = self._nonoverlap_window_1d_partition(all_query_states_for_local_atten)
                rf_w_k_rope = self._nonoverlap_window_1d_partition(all_key_states_for_local_atten)
                
                attn_rf_w_q = rearrange(rf_w_q_rope, "b h g w d -> b (h g) w d") 
                attn_rf_w_k = rearrange(rf_w_k_rope, "b h g w d -> b (h g) w d")
                if self.cpu_offload:
                    attn_rf_w_v = rearrange(rf_w_v[:,:,-1:], "b h g w d -> b (h g) w d")
                    attn_rf_w_v = attn_rf_w_v.to(hidden_states.device)
                else:    
                    attn_rf_w_v = rearrange(rf_w_v[:,:,-1:], "b h g w d -> b (h g) w d") 
                qkv = torch.stack([attn_rf_w_q, attn_rf_w_k, attn_rf_w_v], dim=2)
                
                chunk_features = self.attn(qkv.transpose(1, 3))
                chunk_features = chunk_features.transpose(1, 2)
                chunk_features = rearrange(chunk_features, "b (h g) w d -> b h g w d", h=h)  
                new_chunk_features_value = chunk_features.mean(dim=-2)  # [bsz, nh, g, hd]

                new_chunk_features_key = F.scaled_dot_product_attention(query=new_chunk_features_value.unsqueeze(-2), key=rf_w_k_rope, value=rf_w_k_rope).squeeze(-2) # [bsz, nh, g, 1, hd]
                
                if self.cpu_offload:
                    chunk_features_key = torch.cat([past_global_key_value[0], new_chunk_features_key.to('cpu')], dim=2)
                    chunk_features_value = torch.cat([past_global_key_value[1], new_chunk_features_value.to('cpu')], dim=2)
                else:
                    chunk_features_key = torch.cat([past_global_key_value[0], new_chunk_features_key], dim=2)
                    chunk_features_value = torch.cat([past_global_key_value[1], new_chunk_features_value], dim=2)
                past_global_key_value = (chunk_features_key, chunk_features_value) if use_cache else None
                
                
            if chunk_features_key.size()[-2] != g:
                raise ValueError(
                    f"`chunk_features_key` should be have {g} chunks, but only have"
                    f" {chunk_features_key.size()[-2]} chunks"
                )
            if self.cpu_offload:
                global_qk = torch.einsum('bhd,bhgd->bhg', current_query_ori.to(hidden_states.device), chunk_features_key[:,:,1:-1].to(hidden_states.device))
            else:
                global_qk = torch.einsum('bhd,bhgd->bhg', current_query_ori, chunk_features_key[:,:,1:-1])

            atten_chunks = torch.topk(global_qk, k=atten_chunk_num - 3, dim=-1, largest=True)[1] + 1
            sorted_atten_chunks, indices = torch.sort(atten_chunks, dim=-1)
            # chunk N-1 is the last chunk, and the last part of seq is denote as last key and last value
            first_chunk = torch.full(sorted_atten_chunks[...,-1:].size(), 0, dtype=sorted_atten_chunks.dtype, device=sorted_atten_chunks.device)
            last_chunk = torch.full(sorted_atten_chunks[...,-1:].size(), g-1, dtype=sorted_atten_chunks.dtype, device=sorted_atten_chunks.device)
            if self.cpu_offload:
                sorted_atten_chunks = torch.cat([first_chunk, sorted_atten_chunks, last_chunk], dim=-1).to('cpu')
            else:
                sorted_atten_chunks = torch.cat([first_chunk, sorted_atten_chunks, last_chunk], dim=-1)
            
            selected_keys = torch.gather(rf_w_k.transpose(-1,-3),-1, sorted_atten_chunks.unsqueeze(-2).unsqueeze(-2).repeat(1, 1, self.head_dim,self.window_size, 1)).transpose(-1,-3)
            selected_values = torch.gather(rf_w_v.transpose(-1,-3),-1, sorted_atten_chunks.unsqueeze(-2).unsqueeze(-2).repeat(1, 1, self.head_dim,self.window_size, 1)).transpose(-1,-3)
            selected_keys = rearrange(selected_keys, "b h g w d -> b h (g w) d")
            selected_values = rearrange(selected_values, "b h g w d -> b h (g w) d")
            atten_keys = torch.cat([selected_keys, last_keys], dim=-2) if last_keys is not None else selected_keys
            atten_values = torch.cat([selected_values, last_values], dim=-2) if last_values is not None else selected_values
            if self.cpu_offload:
                atten_keys = atten_keys.to(hidden_states.device)
                atten_values = atten_values.to(hidden_states.device)
                query_states = query_states.to(hidden_states.device)
            # apply rope
            query_states = apply_rotary_pos_emb_for_relative_query(query_states, cos, sin, current_query_position_ids)
            atten_keys = apply_rotary_pos_emb_for_relative_keys(atten_keys, cos, sin, relative_position_ids)

            # implementation with flash-attention 2
            query_states = query_states.transpose(1, 2)
            atten_keys = atten_keys.transpose(1, 2)
            atten_values = atten_values.transpose(1, 2)
            attn_output = flash_attn_with_kvcache(query_states, k_cache=atten_keys, v_cache=atten_values, softmax_scale=self.attn_scale, causal=False)
            attn_output = attn_output.reshape(bsz, ori_len, -1)
            attn_output = self.o_proj(attn_output)
            
            past_cache = (past_key_value, past_query, past_global_key_value)
            if not output_attentions:
                attn_weights = None
            return attn_output, attn_weights, past_cache
        else:
            bsz, q_len, _ = hidden_states.size()
            query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                kv_seq_len += past_key_value_length
                # decide where to add the rope, inside or outside chunk
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            if past_key_value is not None:
                # reuse k, v, self_attention
                if self.cpu_offload:
                    key_states = torch.cat([past_key_value[0].to(hidden_states.device), key_states], dim=2)
                    value_states = torch.cat([past_key_value[1].to(hidden_states.device), value_states], dim=2)
                else:
                    key_states = torch.cat([past_key_value[0], key_states], dim=2)
                    value_states = torch.cat([past_key_value[1], value_states], dim=2)
                
            # update past qkv
            if self.cpu_offload:
                past_key_value = (key_states.to('cpu'), value_states.to('cpu')) if use_cache else None
                past_query = (torch.cat([past_query[0], query_states.to('cpu')], dim=2),) if past_query is not None else (query_states.to('cpu'),)
            else:
                past_key_value = (key_states, value_states) if use_cache else None
                past_query = (torch.cat([past_query[0], query_states], dim=2),) if past_query is not None else (query_states,)
            # add rope
            key_position_ids = torch.arange(0, sequence_length, dtype=position_ids.dtype).unsqueeze(0)
            query_states = apply_rotary_pos_emb_for_relative_query(query_states, cos, sin, position_ids)
            key_states = apply_rotary_pos_emb_for_relative_keys(key_states, cos, sin, key_position_ids)
            
            # implementation with flash-attention 2
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
            attn_output = flash_attn_with_kvcache(query_states, k_cache=key_states, v_cache=value_states, softmax_scale=self.attn_scale, causal=False)
            
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)
            
            past_cache = (past_key_value, past_query, past_global_key_value)
            if not output_attentions:
                attn_weights = None

            return attn_output, attn_weights, past_cache
       


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        past_query: Optional[List[torch.FloatTensor]] = None,
        past_global_key_value: Optional[List[torch.FloatTensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_qkv_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            past_query=past_query,
            past_global_key_value=past_global_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_qkv_cache,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlamaModel):
            module.gradient_checkpointing = value


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

        self.window_size = config.window_size

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        past_queries: Optional[List[torch.FloatTensor]] = None,
        past_global_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        # attention_mask = self._prepare_decoder_attention_mask(
        #     attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        # )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        next_past_key_value_cache = () if use_cache else None
        next_past_query_cache = () if use_cache else None
        next_past_global_key_value_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            past_query = past_queries[idx] if past_queries is not None else None
            past_global_key_value = past_global_key_values[idx] if past_global_key_values is not None else None
            
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    past_query=past_query,
                    past_global_key_value=past_global_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
                next_past_key_value_cache += (layer_outputs[2 if output_attentions else 1][0],)
                next_past_query_cache += (layer_outputs[2 if output_attentions else 1][1],)
                next_past_global_key_value_cache += (layer_outputs[2 if output_attentions else 1][2],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_past_key_value_cache,
            past_queries=next_past_query_cache,
            past_global_key_values=next_past_global_key_value_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # get past_queries and past_global_key_values from past_key_values
        past_queries = None
        past_global_key_values = None
        if past_key_values is not None:
            past_queries = past_key_values[1] 
            past_global_key_values = past_key_values[2] 
            past_key_values = past_key_values[0]
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            past_queries=past_queries,
            past_global_key_values=past_global_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        past_cache = (outputs.past_key_values, outputs.past_queries, outputs.past_global_key_values) if use_cache else None
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_cache,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

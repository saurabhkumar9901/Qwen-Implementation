import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers import Qwen3RMSNorm, apply_rotary_pos_emb

class Qwen3Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_norm = Qwen3RMSNorm(self.num_heads * self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.num_kv_heads * self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, cos=None, sin=None, use_cache=False):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
            
        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat kv heads for GQA so that SDPA can broadcast correctly if needed, or rely on SDPA's built in GQA.
        # PyTorch SDPA does not natively support different number of Q and KV heads via broadcasting 
        # unless we explicitly expand them, BUT wait, since PyTorch 2.2 SDPA supports GQA natively by passing them as is?
        # Actually, in PyTorch 2.1+, SDPA natively supports GQA if you pass enable_gqa=True, but the standard way 
        # to use SDPA with GQA without native support is using repeat_interleave which is more efficient than repeat.
        # Let's use repeat_interleave so it works universally with SDPA.
        key_states = torch.repeat_interleave(key_states, dim=1, repeats=self.num_key_value_groups)
        value_states = torch.repeat_interleave(value_states, dim=1, repeats=self.num_key_value_groups)

        # FlashAttention / SDPA
        # SDPA handles causal mask natively if is_causal=True. But we might have a custom attention_mask for padding.
        if attention_mask is not None:
            # We assume attention_mask is a boolean mask or additive mask.
            # SDPA expects an additive mask of shape (B, 1, Q, K)
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=0.0,
            )
        else:
            # If generating (q_len == 1) and no mask, causal mask is false
            is_causal = True if q_len > 1 else False
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                is_causal=is_causal,
                dropout_p=0.0,
            )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value
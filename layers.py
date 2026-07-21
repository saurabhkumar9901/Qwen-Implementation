import torch
import torch.nn as nn
import torch.nn.functional as F

class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps).to(hidden_states.dtype)
        return self.weight * hidden_states

class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=8192, base=1000000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Build cache once
        t = torch.arange(max_position_embeddings).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    def _update_cache(self, max_seq_len):
        self.max_position_embeddings = max_seq_len
        t = torch.arange(max_seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    def forward(self, position_ids):
        # position_ids: (batch, seq_len)
        max_seq_len = position_ids.max().item() + 1
        if max_seq_len > self.max_position_embeddings:
            self._update_cache(max_seq_len)
            
        cos = self.cos_cache[position_ids]
        sin = self.sin_cache[position_ids]
        return cos, sin

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings to query and key tensors.
    
    Args:
        q: (batch, num_heads, seq_len, head_dim)
        k: (batch, num_kv_heads, seq_len, head_dim)
        cos: (batch, seq_len, head_dim)
        sin: (batch, seq_len, head_dim)
    """
    # Unsqueeze for head dimension broadcasting: (B, 1, S, D)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # SwiGLU activation mechanism[cite: 1]
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from config import Qwen3_0_6B_Config
from layers import Qwen3RMSNorm, Qwen3MLP, Qwen3RotaryEmbedding
from attention import Qwen3Attention

class Qwen3Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen3Attention(config)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen3MLP(config)

    def forward(self, hidden_states, attention_mask, position_ids, rot_emb):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            rot_emb=rot_emb
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class Qwen3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        self.rotary_emb = Qwen3RotaryEmbedding(config.head_dim, max_position_embeddings=config.max_position_embeddings, base=config.rope_theta)
        self.layers = nn.ModuleList([Qwen3Block(config) for _ in range(config.num_hidden_layers)])
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.gradient_checkpointing = False
    
    def enable_gradient_checkpointing(self, enable=True):
        self.gradient_checkpointing = enable

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        batch_size, seq_len = input_ids.shape
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
        
        hidden_states = self.embed_tokens(input_ids)
        
        # Create attention mask
        if attention_mask is None:
            causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=input_ids.device))
            attention_mask = (~causal_mask).unsqueeze(0).unsqueeze(0).float() * -1e9
        else:
            # Text-only: combine causal mask with padding mask
            causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=input_ids.device))
            # Padding mask on key side: (B, 1, seq_len)
            padding_mask = attention_mask.unsqueeze(1).bool()
            # Combined: attend only to non-padded keys that are at or before the query position
            combined = causal_mask.unsqueeze(0) & padding_mask  # (B, seq_len, seq_len)
            attention_mask = (~combined).unsqueeze(1).float() * -1e9  # (B, 1, S, S)
        
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    self.rotary_emb,
                    use_reentrant=False
                )
            else:
                hidden_states = layer(hidden_states, attention_mask, position_ids, self.rotary_emb)
        
        hidden_states = self.norm(hidden_states)
        return hidden_states

class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        hidden_states = self.model(input_ids, attention_mask, position_ids)
        logits = self.lm_head(hidden_states)
        return logits
    
    def generate(self, input_ids, max_new_tokens=100, temperature=0.7, top_k=20):
        """Simple generation method"""
        device = input_ids.device
        
        for _ in range(max_new_tokens):
            logits = self(input_ids)
            next_token_logits = logits[:, -1, :]
            
            if temperature > 0.0:
                next_token_logits = next_token_logits / temperature
            
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        return input_ids

if __name__ == "__main__":
    config = Qwen3_0_6B_Config()
    model = Qwen3ForCausalLM(config)
    params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Instantiated Qwen3 with {params:.2f}B parameters.")

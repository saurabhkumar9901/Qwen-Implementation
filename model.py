import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math
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

    def forward(self, hidden_states, attention_mask, position_ids, past_key_value=None, cos=None, sin=None, use_cache=False):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, past_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            cos=cos,
            sin=sin,
            use_cache=use_cache
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, past_key_value

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

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, use_cache=False):
        batch_size, seq_len = input_ids.shape
        
        if past_key_values is None:
            past_key_values_length = 0
        else:
            past_key_values_length = past_key_values[0][0].shape[2]
            
        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length, seq_len + past_key_values_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
        
        hidden_states = self.embed_tokens(input_ids)
        
        cos, sin = self.rotary_emb(position_ids)
        
        # Create attention mask
        if attention_mask is not None:
            # Text-only: combine causal mask with padding mask
            if seq_len > 1:
                causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=input_ids.device))
                if past_key_values_length > 0:
                    past_mask = torch.ones((seq_len, past_key_values_length), dtype=torch.bool, device=input_ids.device)
                    causal_mask = torch.cat([past_mask, causal_mask], dim=-1)
                
                # Padding mask on key side: (B, 1, seq_len + past_key_values_length)
                padding_mask = attention_mask.unsqueeze(1).bool()
                # Combined: attend only to non-padded keys that are at or before the query position
                combined = causal_mask.unsqueeze(0) & padding_mask  # (B, seq_len, seq_len + past)
                min_val = torch.finfo(hidden_states.dtype).min
                attention_mask = (~combined).unsqueeze(1).to(hidden_states.dtype) * min_val  # (B, 1, S, S+past)
            else:
                padding_mask = attention_mask.unsqueeze(1).unsqueeze(2).bool()
                min_val = torch.finfo(hidden_states.dtype).min
                attention_mask = (~padding_mask).to(hidden_states.dtype) * min_val
        
        next_decoder_cache = () if use_cache else None
        
        for idx, layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            if self.gradient_checkpointing and self.training:
                hidden_states, _ = checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                    cos,
                    sin,
                    False,
                    use_reentrant=False
                )
            else:
                hidden_states, past_key_value = layer(
                    hidden_states, 
                    attention_mask, 
                    position_ids, 
                    past_key_value=past_key_value, 
                    cos=cos, 
                    sin=sin, 
                    use_cache=use_cache
                )
                
            if use_cache:
                next_decoder_cache += (past_key_value,)
        
        hidden_states = self.norm(hidden_states)
        return hidden_states, next_decoder_cache

class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
            

        # Apply special scaling for residual projections ONCE after all modules are initialized
        std = 0.02
        for pn, p in self.named_parameters():
            if pn.endswith('o_proj.weight') or pn.endswith('down_proj.weight'):
                p.data.normal_(mean=0.0, std=(std / math.sqrt(2 * self.config.num_hidden_layers)))

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, use_cache=False):
        hidden_states, next_cache = self.model(
            input_ids, 
            attention_mask=attention_mask, 
            position_ids=position_ids, 
            past_key_values=past_key_values,
            use_cache=use_cache
        )
        logits = self.lm_head(hidden_states)
        if use_cache:
            return logits, next_cache
        return logits
    
    def generate(self, input_ids, attention_mask=None, max_new_tokens=100, temperature=0.7, top_k=20):
        """KV-cached generation method"""
        device = input_ids.device
        past_key_values = None
        
        # Generate position ids if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        
        for _ in range(max_new_tokens):
            if past_key_values is not None:
                model_inputs = input_ids[:, -1:]
                pos_ids = position_ids[:, -1:]
            else:
                model_inputs = input_ids
                pos_ids = position_ids
                
            outputs = self(
                model_inputs, 
                attention_mask=attention_mask,
                position_ids=pos_ids,
                past_key_values=past_key_values, 
                use_cache=True
            )
            logits, past_key_values = outputs
            
            next_token_logits = logits[:, -1, :]
            
            if temperature > 0.0:
                next_token_logits = next_token_logits / temperature
                
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding for T=0
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Update attention mask and position ids for the new token
            attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), device=device)], dim=-1)
            next_pos = position_ids[:, -1:] + 1
            position_ids = torch.cat([position_ids, next_pos], dim=-1)
        
        return input_ids

if __name__ == "__main__":
    config = Qwen3_0_6B_Config()
    model = Qwen3ForCausalLM(config)
    params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Instantiated Qwen3 with {params:.2f}B parameters.")

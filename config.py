from dataclasses import dataclass

@dataclass
class Qwen3_0_6B_Config:
    # Core Qwen3 Specifications
    vocab_size: int = 151680                 
    num_hidden_layers: int = 28              
    num_attention_heads: int = 16            
    num_key_value_heads: int = 8             
    max_position_embeddings: int = 512       # Reduced from 2048 to fit in 16GB T4 GPU
    tie_word_embeddings: bool = True         
    rope_theta: float = 1000000.0            
    
    # Inferred dimensions for ~0.6B scale
    hidden_size: int = 1536                  
    intermediate_size: int = 4096            
    rms_norm_eps: float = 1e-6
    gradient_checkpointing: bool = False
    
    # Training Parameters
    batch_size: int = 1
    accumulation_steps: int = 16
    epochs: int = 3
    lr: float = 3e-4
    num_samples: int = 10000
    weight_decay: float = 0.1

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads


@dataclass
class Qwen3_CPU_Config(Qwen3_0_6B_Config):
    """Tiny config for CPU training/debugging (~30M parameters).
    
    Use this when training on CPU to get reasonable iteration speed.
    The full 0.6B config is too large for practical CPU training.
    """
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    hidden_size: int = 512
    intermediate_size: int = 1024
    max_position_embeddings: int = 256

    # Training overrides for CPU
    batch_size: int = 1
    accumulation_steps: int = 4
    epochs: int = 100
    lr: float = 1e-4
    num_samples: int = 100
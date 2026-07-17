import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from config import Qwen3_0_6B_Config
from model import Qwen3ForCausalLM

import argparse
import os
from checkpoint_manager import CheckpointManager

@torch.no_grad()
def generate(
    model, 
    tokenizer, 
    prompt, 
    max_new_tokens=100, 
    temperature=0.7, 
    top_k=20, 
    device="cpu"
):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    
    model.eval()
    
    print(f"Prompt: {prompt}")
    print("Generating...", end="", flush=True)

    for _ in range(max_new_tokens):
        logits = model(input_ids)
        next_token_logits = logits[:, -1, :]
        
        if temperature > 0.0:
            next_token_logits = next_token_logits / temperature
            
        if top_k > 0:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = -float('Inf')
            
        probs = F.softmax(next_token_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        
        if next_token_id.item() == tokenizer.eos_token_id:
            break

    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="Run inference with Qwen3")
    parser.add_argument('--prompt', type=str, default="The future of artificial intelligence is", help='Text prompt to start generation')
    parser.add_argument('--max_new_tokens', type=int, default=50, help='Maximum tokens to generate')
    parser.add_argument('--checkpoint_dir', type=str, default="./checkpoints_real", help='Directory containing checkpoints')
    parser.add_argument('--checkpoint_file', type=str, default=None, help='Specific checkpoint file to load (optional)')
    parser.add_argument('--cpu_config', action='store_true', help='Use CPU-friendly config (~30M params) if model was trained with it')
    parser.add_argument('--temperature', type=float, default=0.7, help='Generation temperature')
    
    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the official Qwen Tokenizer
    tokenizer_id = "Qwen/Qwen2.5-0.5B" 
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    # 2. Initialize your custom Qwen3 model
    if args.cpu_config:
        from config import Qwen3_CPU_Config
        config = Qwen3_CPU_Config()
        print("Initialized with CPU config (~30M params)")
    else:
        config = Qwen3_0_6B_Config()
    
    model = Qwen3ForCausalLM(config).to(device)
    
    # 3. Load Checkpoint if available
    manager = CheckpointManager(args.checkpoint_dir)
    try:
        manager.load_checkpoint(model, filename=args.checkpoint_file)
    except FileNotFoundError:
        print(f"[WARNING] No checkpoint found. Generating with randomly initialized weights!")

    # 4. Run Inference
    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=20,
        device=device
    )
    
    print(f"\n\nOutput:\n{output}")

if __name__ == "__main__":
    main()
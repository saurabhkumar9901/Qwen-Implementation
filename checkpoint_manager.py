import torch
import os
import re
import json
from typing import Optional, Dict, Any
from config import Qwen3_0_6B_Config
from model import Qwen3ForCausalLM


class CheckpointManager:
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(
        self,
        model: Qwen3ForCausalLM,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        step: int,
        loss: float,
        filename: Optional[str] = None,
        save_optimizer: bool = True
    ):
        if filename is None:
            filename = f"checkpoint_epoch{epoch}_step{step}.pt"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'loss': loss,
            'model_state_dict': model.state_dict(),
            'config': vars(model.config),
        }
        
        if save_optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        print(f"[CheckpointManager] Saved checkpoint: {checkpoint_path}")
        
        config_save_path = os.path.join(self.checkpoint_dir, "config.json")
        with open(config_save_path, 'w') as f:
            json.dump(vars(model.config), f, indent=2)
    
    def load_checkpoint(
        self,
        model: Qwen3ForCausalLM,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        filename: Optional[str] = None,
        load_optimizer: bool = True
    ) -> Dict:
        if filename is None:
            checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pt')]
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir}")
            # Filter to only step-based checkpoints and sort by step number
            step_checkpoints = [f for f in checkpoints if re.search(r'step(\d+)', f)]
            if not step_checkpoints:
                # Fall back to any checkpoint (e.g., best_model.pt)
                filename = sorted(checkpoints)[-1]
            else:
                step_checkpoints.sort(key=lambda x: int(re.search(r'step(\d+)', x).group(1)))
                filename = step_checkpoints[-1]
        
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        print(f"[CheckpointManager] Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"[CheckpointManager] Loaded checkpoint from epoch {checkpoint['epoch']}, step {checkpoint['step']}")
        print(f"  Loss: {checkpoint['loss']:.4f}")
        
        return checkpoint
    
    def save_best_model(
        self,
        model: Qwen3ForCausalLM,
        metric: float,
        best_metric: float,
        metric_name: str = "loss"
    ) -> bool:
        is_best = metric < best_metric
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': vars(model.config),
                metric_name: metric
            }, best_path)
            print(f"[CheckpointManager] New best model saved! {metric_name}: {metric:.4f}")
        
        return is_best
    
    def load_best_model(self, device: str = "cpu") -> Qwen3ForCausalLM:
        best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        
        if not os.path.exists(best_path):
            raise FileNotFoundError(f"No best model found at {best_path}")
        
        checkpoint = torch.load(best_path, map_location=device)
        
        config_dict = checkpoint['config']
        config = Qwen3_0_6B_Config(**config_dict)
        
        model = Qwen3ForCausalLM(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        print(f"[CheckpointManager] Loaded best model")
        if 'loss' in checkpoint:
            print(f"  Loss: {checkpoint['loss']:.4f}")
        
        return model
    
    def list_checkpoints(self) -> list:
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pt')]
        checkpoints.sort()
        
        print(f"\n[CheckpointManager] Available checkpoints in {self.checkpoint_dir}:")
        for ckpt in checkpoints:
            print(f"  - {ckpt}")
        
        return checkpoints


def save_model_weights(model: Qwen3ForCausalLM, output_path: str):
    torch.save(model.state_dict(), output_path)
    print(f"Model weights saved to: {output_path}")


def load_model_weights(model: Qwen3ForCausalLM, weights_path: str, device: str = "cpu"):
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Model weights loaded from: {weights_path}")
    return model


if __name__ == "__main__":
    print("Testing CheckpointManager...")
    
    config = Qwen3_0_6B_Config()
    model = Qwen3ForCausalLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
    
    manager = CheckpointManager("./test_checkpoints")
    
    model.train()
    input_ids = torch.randint(0, config.vocab_size, (1, 128))
    logits = model(input_ids)
    loss = logits.mean()
    loss.backward()
    optimizer.step()
    
    manager.save_checkpoint(model, optimizer, scheduler, epoch=1, step=100, loss=loss.item())
    
    model_loaded = Qwen3ForCausalLM(config)
    checkpoint = manager.load_checkpoint(model_loaded, optimizer, scheduler)
    
    manager.list_checkpoints()
    
    print("\nCheckpointManager test complete!")

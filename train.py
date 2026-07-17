"""
Training script with real coding + OCR datasets

Usage:
    python train.py --mode code                    # Code only
    python train.py --mode instruction             # Code instruction tuning
    python train.py --mode code --cpu_config       # Use small model for CPU
"""

import torch
import torch.nn as nn
import argparse
import contextlib
import os
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from model import Qwen3ForCausalLM
from config import Qwen3_0_6B_Config, Qwen3_CPU_Config
from checkpoint_manager import CheckpointManager
from data_loader import (
    CodeDataset,
    CodeInstructionDataset,
)

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False


def train_step(model, batch, optimizer, scheduler, scaler, device, accumulation_steps, step_idx, use_amp=False):
    model.train()
    
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    
    if 'labels' in batch:
        labels = batch['labels'].to(device)
    else:
        labels = input_ids.clone()
    # Ensure padding tokens are always masked in labels
    labels[attention_mask == 0] = -100
    
    # Use AMP only on CUDA
    if use_amp and device.type == "cuda":
        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        amp_ctx = contextlib.nullcontext()
    
    with amp_ctx:
        logits = model(input_ids, attention_mask=attention_mask)
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        valid_tokens = max((shift_labels != -100).sum().item(), 1)
        loss = loss / (valid_tokens * accumulation_steps)
    
    if scaler is not None and scaler.is_enabled():
        scaler.scale(loss).backward()
    else:
        loss.backward()
    
    if (step_idx + 1) % accumulation_steps == 0:
        if scaler is not None and scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        if scaler is not None and scaler.is_enabled():
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            if scale_before <= scale_after:
                scheduler.step()
        else:
            optimizer.step()
            scheduler.step()
        optimizer.zero_grad()
    
    return loss.item() * accumulation_steps


def create_optimizer(model, config, device):
    """Create optimizer — 8-bit AdamW on CUDA if available, regular AdamW otherwise."""
    if HAS_BNB and device.type == "cuda":
        return bnb.optim.AdamW8bit(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        return torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)


def create_scaler(device, use_amp):
    """Create GradScaler only for CUDA AMP. Returns None on CPU."""
    if use_amp and device.type == "cuda":
        return torch.amp.GradScaler("cuda", enabled=True)
    else:
        return None  # No scaler on CPU — avoids crash on older PyTorch


def resume_from_checkpoint(args, model, optimizer, scheduler):
    """Load checkpoint and return (start_epoch, start_step). Returns (0, 0) if no resume."""
    if getattr(args, 'resume_checkpoint', None) and os.path.exists(args.resume_checkpoint):
        checkpoint_dir = os.path.dirname(args.resume_checkpoint) or args.checkpoint_dir
        filename = os.path.basename(args.resume_checkpoint)
        manager = CheckpointManager(checkpoint_dir)
        checkpoint = manager.load_checkpoint(model, optimizer, scheduler, filename=filename)
        return checkpoint['epoch'], checkpoint['step'] + 1
    elif getattr(args, 'auto_resume', False):
        manager = CheckpointManager(args.checkpoint_dir)
        try:
            checkpoint = manager.load_checkpoint(model, optimizer, scheduler)
            return checkpoint['epoch'], checkpoint['step'] + 1
        except (FileNotFoundError, IndexError):
            print(f"No checkpoint found in {args.checkpoint_dir} to auto-resume from. Starting from scratch.")
    return 0, 0


def train_code(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Training: Code Model")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    if args.cpu_config and device.type == "cpu":
        config = Qwen3_CPU_Config(gradient_checkpointing=args.gradient_checkpointing)
        print("Using CPU-optimized config (~30M parameters)")
    else:
        config = Qwen3_0_6B_Config(gradient_checkpointing=args.gradient_checkpointing)
        config.max_position_embeddings = args.max_length
    
    model = Qwen3ForCausalLM(config).to(device)
    
    if config.gradient_checkpointing:
        model.model.enable_gradient_checkpointing(True)
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {total_params:.2f}M")
    
    print(f"\nLoading code dataset...")
    dataset = CodeDataset(
        dataset_name=args.code_dataset,
        language=args.language,
        max_length=config.max_position_embeddings,
        num_samples=config.num_samples,
        streaming=args.streaming
    )
    
    accumulation_steps = config.accumulation_steps
    num_epochs = config.epochs
    total_steps = (len(dataset) // config.batch_size // accumulation_steps) * num_epochs
    
    optimizer = create_optimizer(model, config, device)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.05),
        num_training_steps=total_steps
    )
    
    use_amp = device.type == "cuda"
    scaler = create_scaler(device, use_amp)
    
    checkpoint_manager = CheckpointManager(args.checkpoint_dir) if args.save_checkpoints else None
    
    # Resume from checkpoint if specified
    start_epoch, start_step = resume_from_checkpoint(args, model, optimizer, scheduler)
    
    optimizer.zero_grad()
    
    print(f"\nStarting training...")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Gradient accumulation: {accumulation_steps}")
    print(f"  Effective batch size: {config.batch_size * accumulation_steps}")
    print(f"  Learning rate: {config.lr}")
    print(f"  Total steps: {total_steps}")
    if start_epoch > 0 or start_step > 0:
        print(f"  Resuming from epoch {start_epoch}, step {start_step}")
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        # Deterministic shuffle per epoch for safe resuming
        if getattr(dataset, "streaming", False) or isinstance(dataset, torch.utils.data.IterableDataset):
            dataloader = DataLoader(
                dataset,
                batch_size=config.batch_size,
                num_workers=args.num_workers if device.type == "cuda" else 0
            )
        else:
            generator = torch.Generator()
            generator.manual_seed(42 + epoch)
            sampler = torch.utils.data.RandomSampler(dataset, generator=generator)
            dataloader = DataLoader(
                dataset,
                batch_size=config.batch_size,
                sampler=sampler,
                num_workers=args.num_workers if device.type == "cuda" else 0
            )
        
        for step_idx, batch in enumerate(dataloader):
            if epoch == start_epoch and step_idx < start_step:
                continue  # Skip already-completed steps when resuming
            
            loss = train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                accumulation_steps=accumulation_steps,
                step_idx=step_idx,
                use_amp=use_amp
            )
            
            if (step_idx + 1) % accumulation_steps == 0:
                step_num = step_idx + 1
                lr = scheduler.get_last_lr()[0]
                print(f"Step {step_num}/{len(dataloader)} | Loss: {loss:.4f} | LR: {lr:.2e}")
                
                if checkpoint_manager and step_num % (accumulation_steps * 10) == 0:
                    checkpoint_manager.save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        step=step_idx,
                        loss=loss,
                        filename=f"code_checkpoint_epoch{epoch}_step{step_idx}.pt"
                    )
        
        # Apply remaining gradients at the end of the epoch
        if len(dataloader) % accumulation_steps != 0:
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            if scaler is not None and scaler.is_enabled():
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                if scale_before <= scale_after:
                    scheduler.step()
            else:
                optimizer.step()
                scheduler.step()
            optimizer.zero_grad()





def train_code_instruction(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Training: Code Instruction Tuning")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    if args.cpu_config and device.type == "cpu":
        config = Qwen3_CPU_Config(gradient_checkpointing=args.gradient_checkpointing)
        print("Using CPU-optimized config (~30M parameters)")
    else:
        config = Qwen3_0_6B_Config(gradient_checkpointing=args.gradient_checkpointing)
        config.max_position_embeddings = args.max_length
    
    model = Qwen3ForCausalLM(config).to(device)
    
    if config.gradient_checkpointing:
        model.model.enable_gradient_checkpointing(True)
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {total_params:.2f}M")
    
    print(f"\nLoading code instruction dataset...")
    dataset = CodeInstructionDataset(
        max_length=config.max_position_embeddings,
        num_samples=config.num_samples
    )
    
    accumulation_steps = config.accumulation_steps
    num_epochs = config.epochs
    total_steps = (len(dataset) // config.batch_size // accumulation_steps) * num_epochs
    
    optimizer = create_optimizer(model, config, device)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.05),
        num_training_steps=total_steps
    )
    
    use_amp = device.type == "cuda"
    scaler = create_scaler(device, use_amp)
    
    checkpoint_manager = CheckpointManager(args.checkpoint_dir) if args.save_checkpoints else None
    
    start_epoch, start_step = resume_from_checkpoint(args, model, optimizer, scheduler)
    
    optimizer.zero_grad()
    
    print(f"\nStarting instruction tuning...")
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        # Deterministic shuffle per epoch for safe resuming
        if getattr(dataset, "streaming", False) or isinstance(dataset, torch.utils.data.IterableDataset):
            dataloader = DataLoader(
                dataset,
                batch_size=config.batch_size,
                num_workers=args.num_workers if device.type == "cuda" else 0
            )
        else:
            generator = torch.Generator()
            generator.manual_seed(42 + epoch)
            sampler = torch.utils.data.RandomSampler(dataset, generator=generator)
            dataloader = DataLoader(
                dataset,
                batch_size=config.batch_size,
                sampler=sampler,
                num_workers=args.num_workers if device.type == "cuda" else 0
            )
        
        for step_idx, batch in enumerate(dataloader):
            if epoch == start_epoch and step_idx < start_step:
                continue
            
            loss = train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                accumulation_steps=accumulation_steps,
                step_idx=step_idx,
                use_amp=use_amp
            )
            
            if (step_idx + 1) % accumulation_steps == 0:
                step_num = step_idx + 1
                lr = scheduler.get_last_lr()[0]
                print(f"Step {step_num}/{len(dataloader)} | Loss: {loss:.4f} | LR: {lr:.2e}")
                
                if checkpoint_manager and step_num % (accumulation_steps * 10) == 0:
                    checkpoint_manager.save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        step=step_idx,
                        loss=loss,
                        filename=f"instruction_checkpoint_epoch{epoch}_step{step_idx}.pt"
                    )

        # Apply remaining gradients at the end of the epoch
        if len(dataloader) % accumulation_steps != 0:
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            if scaler is not None and scaler.is_enabled():
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                if scale_before <= scale_after:
                    scheduler.step()
            else:
                optimizer.step()
                scheduler.step()
            optimizer.zero_grad()


def main():
    parser = argparse.ArgumentParser(description="Train Qwen3 on Code datasets")
    
    parser.add_argument('--mode', type=str, default='code', choices=['code', 'instruction'],
                        help='Training mode')
    parser.add_argument('--max_length', type=int, default=2048, help='Maximum sequence length')
    parser.add_argument('--language', type=str, default='python', help='Programming language for code dataset')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Enable gradient checkpointing')
    parser.add_argument('--save_checkpoints', action='store_true', help='Save checkpoints during training')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_real', help='Checkpoint directory')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Path to checkpoint file to resume training from')
    parser.add_argument('--auto_resume', action='store_true', help='Automatically resume from the latest checkpoint in checkpoint_dir')
    parser.add_argument('--streaming', action='store_true', help='Use streaming dataset')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--code_dataset', type=str, default='bigcode/starcoderdata', help='Code dataset name')
    parser.add_argument('--cpu_config', action='store_true', help='Use small CPU-friendly model config (~30M params)')
    
    args = parser.parse_args()
    
    if args.mode == 'code':
        train_code(args)
    elif args.mode == 'instruction':
        train_code_instruction(args)


if __name__ == "__main__":
    main()

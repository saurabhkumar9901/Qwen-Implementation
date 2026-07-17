import modal
import subprocess
import os

# Define the Modal App
app = modal.App("qwen3-training")

# Define the environment image with PyTorch and dependencies, and add our local project files
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "pillow",
        "numpy",
        "bitsandbytes"
    )
    .workdir("/workspace")
    .add_local_dir(
        local_path=".", 
        remote_path="/workspace",
        ignore=[".git", "*.pt", "checkpoints*"]
    )
)

# Create a persistent volume to save checkpoints across runs
volume = modal.Volume.from_name("qwen3-checkpoints", create_if_missing=True)

@app.function(
    image=image,
    gpu="A10G", # Upgraded to 24GB A10G because 16GB T4 is too small for a 1.15B multimodal backward pass
    volumes={"/workspace/checkpoints_real": volume}, # Mount the volume to the checkpoint dir
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])], # Authenticate with HuggingFace
    timeout=86400 # Allow up to 24 hours of training
)
def run_training(mode: str = "code", auto_resume: bool = True):
    """
    Run the training script inside the Modal container.
    """
    import subprocess
    import os
    
    os.chdir("/workspace")
    
    cmd = [
        "python", "train.py",
        "--mode", mode,
        "--gradient_checkpointing",
        "--save_checkpoints",
    ]
    if auto_resume:
        cmd.append("--auto_resume")
        
    print(f"Running command: {' '.join(cmd)}")
    print("Checkpoints will be saved to the persistent volume.")
    subprocess.run(cmd, check=True)

@app.local_entrypoint()
def main(mode: str = "code", resume: bool = True):
    print(f"Submitting training job to Modal (Mode: {mode})...")
    print("If you used the --detach flag, you can safely close this terminal.")
    
    # Use .remote() and rely on the --detach CLI flag to keep the app alive
    run_training.remote(mode=mode, auto_resume=resume)
    
    print(f"\n✅ Training completed!")

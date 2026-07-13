#!/usr/bin/env python
"""
Run sweep for MViT HOG fine-tuning with regression + classification
"""

import os
import sys
import subprocess
import wandb
import torch
import yaml
from pathlib import Path

def check_gpu_availability():
    """Check if CUDA is available and print GPU info"""
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This training requires GPU.")
        sys.exit(1)

    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    print()

def check_checkpoint_exists():
    """Verify that the pretrained checkpoint exists"""
    checkpoint_path = "/media/data1/models/DeepCoro_HOG_MVIT/mvit_v2_s_24_16_2_AdamW_new_20250915-154813_9rc217lv/best.pt"
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    print(f"✓ Found checkpoint: {checkpoint_path}")
    return checkpoint_path

def train():
    """Training function for wandb sweep"""
    # Initialize wandb run
    run = wandb.init()

    # Get sweep parameters
    config = wandb.config

    # Build command with sweep parameters
    cmd = [
        "python", "orion/utils/video_training_and_eval.py",
        f"--config_path={config.config_path}",
    ]

    # Add all sweep parameters as command line arguments
    for key, value in config.items():
        if key != "config_path":
            # Handle nested dictionaries (like loss_structure)
            if isinstance(value, dict):
                import json
                value_str = json.dumps(value)
                cmd.append(f"--{key}={value_str}")
            else:
                cmd.append(f"--{key}={value}")

    # Add sweep-specific flags
    cmd.extend([
        "--wandb_sweep",  # Enable sweep mode
        f"--wandb_run_id={run.id}",
        f"--wandb_run_name={run.name}",
    ])

    print(f"Running command: {' '.join(cmd)}")

    # Run training
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print("Training completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        wandb.finish(exit_code=1)
        return

    wandb.finish()

def main():
    """Main function to set up and run the sweep"""

    # Check environment
    print("=" * 80)
    print("MViT HOG Fine-tuning Sweep Runner")
    print("=" * 80)

    # Check GPU availability
    check_gpu_availability()

    # Check checkpoint exists
    checkpoint_path = check_checkpoint_exists()

    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("\nWARNING: Virtual environment may not be activated!")
        print("Continuing with sweep execution...")

    # Load sweep configuration
    sweep_config_path = "notebooks/config/sweep_mvit_hog_finetune.yaml"
    if not os.path.exists(sweep_config_path):
        print(f"ERROR: Sweep config not found at {sweep_config_path}")
        sys.exit(1)

    with open(sweep_config_path, 'r') as f:
        sweep_config = yaml.safe_load(f)

    print(f"\n✓ Loaded sweep config from: {sweep_config_path}")
    print(f"  Method: {sweep_config['method']}")
    print(f"  Metric: {sweep_config['metric']['name']} ({sweep_config['metric']['goal']})")
    print(f"  Total runs: {sweep_config.get('count', 'unlimited')}")

    # Initialize wandb sweep
    sweep_id = wandb.sweep(
        sweep_config,
        project=sweep_config['parameters']['project']['value'],
        entity=sweep_config['parameters']['entity']['value']
    )

    print(f"\n✓ Created sweep with ID: {sweep_id}")
    print(f"  View at: https://wandb.ai/{sweep_config['parameters']['entity']['value']}/{sweep_config['parameters']['project']['value']}/sweeps/{sweep_id}")

    # Get number of GPUs to use
    n_gpus = torch.cuda.device_count()
    n_agents = min(n_gpus, 2)  # Use up to 2 agents

    # Auto-start sweep without prompting
    print(f"\n✓ Auto-starting sweep with {n_agents} agent(s)...")

    print(f"\nStarting sweep with {n_agents} agent(s)...")

    # Run sweep agent(s)
    if n_agents == 1:
        # Single agent
        wandb.agent(sweep_id, function=train, count=sweep_config.get('count'))
    else:
        # Multiple agents in parallel
        import multiprocessing

        def run_agent(gpu_id):
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            wandb.agent(sweep_id, function=train)

        processes = []
        for i in range(n_agents):
            p = multiprocessing.Process(target=run_agent, args=(i,))
            p.start()
            processes.append(p)

        # Wait for all agents to complete
        for p in processes:
            p.join()

    print("\n" + "=" * 80)
    print("Sweep completed!")
    print(f"View results at: https://wandb.ai/{sweep_config['parameters']['entity']['value']}/{sweep_config['parameters']['project']['value']}/sweeps/{sweep_id}")
    print("=" * 80)

if __name__ == "__main__":
    main()
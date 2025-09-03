#!/bin/bash

# This script is specifically for W&B sweeps with multi-GPU support
# Properly handles distributed training to prevent duplicate sweep runs

# Initialize Conda environment
source /root/miniforge3/etc/profile.d/mamba.sh
eval "$(conda shell.bash hook)"  # Ensures Conda commands are available in the script

# Activate the 'pytorch' environment
conda activate pytorch

# Set environment variable to indicate we're in sweep mode
export WANDB_SWEEP_MODE=1

# Set parent PID for consistent file naming across processes
export PARENT_PID=$$

# Get number of GPUs (default to 1 if not specified)
NGPUS=${NGPUS:-1}

# Execute torchrun with multi-GPU for sweep runs
torchrun --standalone --nnodes=1 --nproc-per-node=${NGPUS} "$1" --config_path="$2" "${@:3}"
#"$1" is the path to your Python script.
#"$2" is the path to your config file, which is passed along with the --config_path flag.
#"${@:3}" represents all additional arguments starting from the third one, which will be the hyperparameters provided by wandb.
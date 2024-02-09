#!/bin/bash

# This script activates the Mamba environment and runs torchrun with the Python script and config file as arguments.

# Initialize Conda environment
source /root/miniforge3/etc/profile.d/mamba.sh
eval "$(conda shell.bash hook)"  # Ensures Conda commands are available in the script

# Activate the 'pytorch' environment
conda activate pytorch

# Execute torchrun with the Python script as an argument and pass the config file to the Python script
torchrun --standalone --nnodes=1 --nproc-per-node=2 "$1" --config_file="$2" "${@:3}"
#"$1" is the path to your Python script.
#"$2" is the path to your config file, which is passed along with the --config_file flag.
#"${@:3}" represents all additional arguments starting from the third one, which will be the hyperparameters provided by wandb.
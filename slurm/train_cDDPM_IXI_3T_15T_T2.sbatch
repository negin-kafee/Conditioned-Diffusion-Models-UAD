#!/bin/bash

#SBATCH --job-name=cDDPM_IXI_3T_15T_T2
#SBATCH --output=slurm/logs/cDDPM_IXI_3T_15T_T2_%j.log
#SBATCH --error=slurm/logs/cDDPM_IXI_3T_15T_T2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --time=30-00:00:00
#SBATCH --partition=gpuq

# Activate environment
# Activate your conda/venv environment
# source /path/to/your/env/bin/activate

# Set environment variables
export DATA_DIR=${DATA_DIR:-/path/to/cddpm_prepared}
export LOG_DIR=${LOG_DIR:-./logs}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-./wandb_cache}
export WANDB_MODE=online

# Create log directory
mkdir -p $LOG_DIR

# Change to repo directory
cd "$(dirname "$0")/.."  # Change to repo root

# Run training with single fold (combined 3T + 1.5T T2 dataset)
python -u run.py experiment=cDDPM/cDDPM_IXI_3T_15T_T2

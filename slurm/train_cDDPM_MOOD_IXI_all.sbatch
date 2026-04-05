#!/bin/bash
#SBATCH --job-name=cDDPM_MOOD_IXI_all
#SBATCH --output=slurm/logs/%j_cDDPM_MOOD_IXI_all.out
#SBATCH --error=slurm/logs/%j_cDDPM_MOOD_IXI_all.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

echo "Starting MOOD_IXI_all cDDPM training job on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"

cd "$(dirname "$0")/.."  # Change to repo root

# Activate environment
# Activate your conda/venv environment
# source /path/to/your/env/bin/activate

# Set environment variables
export DATA_DIR=${DATA_DIR:-/path/to/cddpm_prepared}
export LOG_DIR=${LOG_DIR:-./logs}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-./wandb_cache}
export WANDB_MODE=online

echo "Environment activated"
echo "Python path: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Train MOOD_IXI_all model (combines all MOOD + IXI data: 3T+1.5T, T1+T2)
echo "Starting training for MOOD_IXI_all..."
python -u run.py experiment=cDDPM/cDDPM_MOOD_IXI_all

echo "MOOD_IXI_all training completed at $(date)"
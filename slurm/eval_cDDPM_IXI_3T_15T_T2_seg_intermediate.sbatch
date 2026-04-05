#!/bin/bash

#SBATCH --job-name=eval_IXI_3T_15T_T2_seg_int
#SBATCH --output=slurm/logs/eval_IXI_3T_15T_T2_seg_intermediate_%j.log
#SBATCH --error=slurm/logs/eval_IXI_3T_15T_T2_seg_intermediate_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --partition=gpuq

# Activate environment
# Activate your conda/venv environment
# source /path/to/your/env/bin/activate

# Set environment variables
export DATA_DIR=${DATA_DIR:-/path/to/cddpm_prepared}
export LOG_DIR=${LOG_DIR:-./logs}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-./wandb_cache}
export WANDB_MODE=online

# Change to repo directory
cd "$(dirname "$0")/.."  # Change to repo root

# Find latest checkpoint - use the actual run directory
CKPT_PATH="${LOG_DIR:-./logs}"
BEST_CKPT=$(ls ${CKPT_PATH}/epoch-*.ckpt 2>/dev/null | grep -v last | sort -t'-' -k2 -n | tail -1)

echo "Using checkpoint: ${BEST_CKPT}"

# Run evaluation only (skip training)
python -u run.py experiment=cDDPM/cDDPM_IXI_3T_15T_T2_seg \
    trainer.max_epochs=0 \
    "ckpt_path=${BEST_CKPT}" \
    trainer.devices=1

#!/bin/bash

#SBATCH --job-name=eval_MOOD_IXI_all_BraTS_T2
#SBATCH --output=slurm/logs/eval_MOOD_IXI_all_BraTS_T2_%j.log
#SBATCH --error=slurm/logs/eval_MOOD_IXI_all_BraTS_T2_%j.err
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

# Create log directory
mkdir -p $LOG_DIR

# Change to repo directory
cd "$(dirname "$0")/.."  # Change to repo root

# Checkpoint path (parent dir - code will append /checkpoints)
CKPT_DIR="${LOG_DIR:-./logs}"

# Run evaluation only on BraTS_T2
# Override testsets to only evaluate on BraTS_T2
python -u run.py experiment=cDDPM/cDDPM_MOOD_IXI_all \
    onlyEval=True \
    +num_folds=1 \
    load_checkpoint="${CKPT_DIR}" \
    datamodule.cfg.testsets="[Datamodules_eval.BraTS_T2]"

#!/bin/bash
#SBATCH --job-name=eval_MOOD_IXI_3T_15T_T1_seg
#SBATCH --output=${LOG_DIR:-./logs}
#SBATCH --error=${LOG_DIR:-./logs}
#SBATCH --time=06:00:00
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Create log directory
mkdir -p ${LOG_DIR:-./logs}

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

# Run directory path (not checkpoint file)
RUN_PATH="${LOG_DIR:-./logs}"

# Run evaluation only
python run.py \
    experiment=cDDPM/cDDPM_MOOD_IXI_3T_15T_T1_seg \
    onlyEval=True \
    load_checkpoint="${RUN_PATH}" \
    new_wandb_run=True \
    name=eval_MOOD_IXI_3T_15T_T1_seg \
    logger.wandb.name=eval_MOOD_IXI_3T_15T_T1_seg \
    test_after_training=True \
    trainer.devices=1

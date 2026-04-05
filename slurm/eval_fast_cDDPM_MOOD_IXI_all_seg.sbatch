#!/bin/bash
#SBATCH --job-name=eval_fast_T2_all
#SBATCH --output=${LOG_DIR:-./logs}
#SBATCH --error=${LOG_DIR:-./logs}
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Create log directory
mkdir -p ${LOG_DIR:-./logs}

# Activate environment
source "$(dirname "$0")/../pc_environment.env"
conda activate strega-cu118-py311

# Set environment variables
export DATA_DIR=${DATA_DIR:-/path/to/cddpm_prepared}
export LOG_DIR=${LOG_DIR:-./logs}

cd "$(dirname "$0")/.."  # Change to repo root

# Run evaluation with BraTS_T2_FAST only
python run.py \
    experiment=cDDPM/DDPM \
    datamodule=MOOD_IXI_all_seg \
    model=DDPM_2D \
    name=evalf_t2all \
    trainer.devices=1 \
    num_folds=1 \
    onlyEval=True \
    test_after_training=True \
    new_wandb_run=True \
    logger.wandb.name=evalf_t2all \
    datamodule.cfg.testsets=[Datamodules_eval.BraTS_T2_FAST] \
    datamodule.cfg.mode=t2 \
    datamodule.cfg.num_workers=8 \
    hydra.job.name=evalf_t2all \
    hydra.output_subdir=null \
    hydra.run.dir=${LOG_DIR}/logs/runs/evalf_t2all/${now:%Y-%m-%d_%H-%M-%S} \
    load_checkpoint=${LOG_DIR:-./logs}

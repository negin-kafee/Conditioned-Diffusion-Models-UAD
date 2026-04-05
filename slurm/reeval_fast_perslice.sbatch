#!/bin/bash

#SBATCH --job-name=reeval_fast_perslice
#SBATCH --output=slurm/logs/reeval_fast_perslice_%j.log
#SBATCH --error=slurm/logs/reeval_fast_perslice_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --partition=gpuq

# Re-evaluate FAST models to get per-slice DICE metrics

# Activate your conda/venv environment
# source /path/to/your/env/bin/activate

export DATA_DIR=${DATA_DIR:-/path/to/cddpm_prepared}
export LOG_DIR=${LOG_DIR:-./logs}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-./wandb_cache}
export WANDB_MODE=online

cd "$(dirname "$0")/.."  # Change to repo root

echo "=========================================="
echo "Re-evaluating FAST models for per-slice DICE"
echo "=========================================="

# 1. IXI_3T_15T_T2_seg -> BraTS_T2_FAST
echo "[1/3] IXI_3T_15T_T2_seg"
python run.py \
    experiment=cDDPM/DDPM \
    datamodule=IXI_3T_15T_T2_seg \
    model=DDPM_2D \
    name=reeval_fast_t2 \
    trainer.devices=1 \
    num_folds=1 \
    onlyEval=True \
    test_after_training=True \
    new_wandb_run=True \
    datamodule.cfg.testsets=[Datamodules_eval.BraTS_T2_FAST] \
    datamodule.cfg.mode=t2 \
    load_checkpoint=${LOG_DIR:-./logs}

# 2. MOOD_IXI_3T_15T_T1_seg -> BraTS_T1_FAST
echo "[2/3] MOOD_IXI_3T_15T_T1_seg"
python run.py \
    experiment=cDDPM/DDPM \
    datamodule=MOOD_IXI_3T_15T_T1_seg \
    model=DDPM_2D \
    name=reeval_fast_t1 \
    trainer.devices=1 \
    num_folds=1 \
    onlyEval=True \
    test_after_training=True \
    new_wandb_run=True \
    datamodule.cfg.testsets=[Datamodules_eval.BraTS_T1_FAST] \
    datamodule.cfg.mode=t1 \
    load_checkpoint=${LOG_DIR:-./logs}

# 3. MOOD_IXI_all_seg -> BraTS_T2_FAST
echo "[3/3] MOOD_IXI_all_seg"
python run.py \
    experiment=cDDPM/DDPM \
    datamodule=MOOD_IXI_all_seg \
    model=DDPM_2D \
    name=reeval_fast_all \
    trainer.devices=1 \
    num_folds=1 \
    onlyEval=True \
    test_after_training=True \
    new_wandb_run=True \
    datamodule.cfg.testsets=[Datamodules_eval.BraTS_T2_FAST] \
    datamodule.cfg.mode=t1t2 \
    load_checkpoint=${LOG_DIR:-./logs}

echo "=========================================="
echo "All FAST evaluations complete!"
echo "=========================================="

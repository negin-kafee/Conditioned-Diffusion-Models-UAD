#!/bin/bash

#SBATCH --job-name=reeval_all_perslice
#SBATCH --output=slurm/logs/reeval_all_perslice_%j.log
#SBATCH --error=slurm/logs/reeval_all_perslice_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=gpuq

# Re-evaluate ALL baseline models to get per-slice DICE metrics

# Activate your conda/venv environment
# source /path/to/your/env/bin/activate

export DATA_DIR=${DATA_DIR:-/path/to/cddpm_prepared}
export LOG_DIR=${LOG_DIR:-./logs}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-./wandb_cache}
export WANDB_MODE=online

cd "$(dirname "$0")/.."  # Change to repo root

echo "=========================================="
echo "Re-evaluating all models for per-slice DICE"
echo "=========================================="

# 1. MOOD_3T_T1 -> BraTS_T1
echo "[1/8] MOOD_3T_T1"
python -u run.py experiment=cDDPM/cDDPM_MOOD_3T_T1 \
    onlyEval=True num_folds=1 \
    load_checkpoint="${LOG_DIR:-./logs}"

# 2. MOOD_IXI_3T_T1 -> BraTS_T1
echo "[2/8] MOOD_IXI_3T_T1"
python -u run.py experiment=cDDPM/cDDPM_MOOD_IXI_3T_T1 \
    onlyEval=True num_folds=1 \
    load_checkpoint="${LOG_DIR:-./logs}"

# 3. MOOD_IXI_3T_15T_T1 -> BraTS_T1
echo "[3/8] MOOD_IXI_3T_15T_T1"
python -u run.py experiment=cDDPM/cDDPM_MOOD_IXI_3T_15T_T1 \
    onlyEval=True num_folds=1 \
    load_checkpoint="${LOG_DIR:-./logs}"

# 4. IXI_3T_T2 -> BraTS_T2 (5-fold)
echo "[4/8] IXI_3T_T2"
python -u run.py experiment=cDDPM/cDDPM_IXI_3T_T2 \
    onlyEval=True num_folds=5 \
    load_checkpoint="${LOG_DIR:-./logs}"

# 5. IXI_15T_T2 -> BraTS_T2 (5-fold)
echo "[5/8] IXI_15T_T2"
python -u run.py experiment=cDDPM/cDDPM_IXI_15T_T2 \
    onlyEval=True num_folds=5 \
    load_checkpoint="${LOG_DIR:-./logs}"

# 6. IXI_3T_15T_T2 -> BraTS_T2
echo "[6/8] IXI_3T_15T_T2"
python -u run.py experiment=cDDPM/cDDPM_IXI_3T_15T_T2 \
    onlyEval=True num_folds=1 \
    load_checkpoint="${LOG_DIR:-./logs}"

# 7. MOOD_IXI_3T_T1T2 -> BraTS_T1 and BraTS_T2
echo "[7/8] MOOD_IXI_3T_T1T2"
python -u run.py experiment=cDDPM/cDDPM_MOOD_IXI_3T_T1T2 \
    onlyEval=True num_folds=1 \
    load_checkpoint="${LOG_DIR:-./logs}"

# 8. MOOD_IXI_all -> BraTS_T1 and BraTS_T2
echo "[8/8] MOOD_IXI_all"
python -u run.py experiment=cDDPM/cDDPM_MOOD_IXI_all \
    onlyEval=True +num_folds=1 \
    load_checkpoint="${LOG_DIR:-./logs}"

echo "=========================================="
echo "All evaluations complete!"
echo "=========================================="

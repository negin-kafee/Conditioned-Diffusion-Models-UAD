# Dataset Setup Checklist

Use this checklist to track your progress in setting up the datasets for training.

## Pre-Setup

- [ ] Conda environment activated: `conda activate cddpm-uad`
- [ ] All packages installed (from `environment.yml` and `requirements.txt`)
- [ ] `pc_environment.env` configured with DATA_DIR and LOG_DIR paths

## Data Organization

### IXI Dataset (Training Data)
- [ ] IXI T1 images placed in `$DATA_DIR/Data/Train/ixi/t1/`
- [ ] IXI T2 images placed in `$DATA_DIR/Data/Train/ixi/t2/`
- [ ] Brain masks placed in `$DATA_DIR/Data/Train/ixi/mask/`
- [ ] File naming follows convention: `*_t1.nii.gz`, `*_t2.nii.gz`, `*_mask.nii.gz`
- [ ] All images are in NIfTI format (`.nii.gz`)

### BraTS Dataset (Test Data)
- [ ] BraTS images placed in `$DATA_DIR/Data/Test/Brats21/t1/` or `t2/`
- [ ] BraTS masks placed in `$DATA_DIR/Data/Test/Brats21/mask/`
- [ ] BraTS segmentations placed in `$DATA_DIR/Data/Test/Brats21/seg/`
- [ ] File naming follows convention: `*_t1.nii.gz`, `*_mask.nii.gz`, `*_seg.nii.gz`

### MOOD Dataset (Test Data)
- [ ] MOOD images placed in `$DATA_DIR/Data/Test/MOOD/t1/` or `t2/`
- [ ] MOOD masks placed in `$DATA_DIR/Data/Test/MOOD/mask/`
- [ ] MOOD segmentations placed in `$DATA_DIR/Data/Test/MOOD/seg/` (if available)
- [ ] File naming follows convention

## CSV Generation

### IXI Splits
- [ ] Generated IXI training splits for all folds (0-4)
  ```bash
  for fold in {0..4}; do
    python scripts/generate_csv_splits.py --data_dir $DATA_DIR --dataset ixi --modality t1 --split train --fold $fold
  done
  ```
- [ ] Generated IXI validation splits for all folds (0-4)
  ```bash
  for fold in {0..4}; do
    python scripts/generate_csv_splits.py --data_dir $DATA_DIR --dataset ixi --modality t1 --split val --fold $fold
  done
  ```
- [ ] Generated IXI test split
  ```bash
  python scripts/generate_csv_splits.py --data_dir $DATA_DIR --dataset ixi --modality t1 --split test
  ```

### BraTS Splits
- [ ] Generated BraTS validation split
  ```bash
  python scripts/generate_csv_splits.py --data_dir $DATA_DIR --dataset brats --modality t2 --split val
  ```
- [ ] Generated BraTS test split
  ```bash
  python scripts/generate_csv_splits.py --data_dir $DATA_DIR --dataset brats --modality t2 --split test
  ```

### MOOD Splits
- [ ] Generated MOOD validation split
  ```bash
  python scripts/generate_csv_splits.py --data_dir $DATA_DIR --dataset mood --modality t2 --split val
  ```
- [ ] Generated MOOD test split
  ```bash
  python scripts/generate_csv_splits.py --data_dir $DATA_DIR --dataset mood --modality t2 --split test
  ```

## Verification

- [ ] Ran verification script
  ```bash
  python scripts/verify_dataset.py --data_dir $DATA_DIR
  ```
- [ ] All directory structure checks passed ✓
- [ ] All CSV files found ✓
- [ ] All file references verified ✓
- [ ] Environment configuration correct ✓
- [ ] (Optional) Image dimensions checked
  ```bash
  python scripts/verify_dataset.py --data_dir $DATA_DIR --check_dimensions
  ```

## Training Preparation

- [ ] Reviewed configuration in `configs/datamodule/IXI.yaml`
- [ ] Adjusted batch size if needed (default: 32)
- [ ] Adjusted number of workers if needed (default: 4)
- [ ] Confirmed modality setting (t1 or t2)
- [ ] Selected which test sets to evaluate

## Training

### Encoder Pretraining (Recommended)
- [ ] Started encoder pretraining
  ```bash
  python run.py experiment=cDDPM/Spark_2D_pretrain datamodule.cfg.mode=t1
  ```
- [ ] Noted checkpoint path from pretraining output
- [ ] Pretraining completed successfully

### Full Model Training
- [ ] Started full model training with pretrained encoder
  ```bash
  python run.py experiment=cDDPM/DDPM_cond_spark_2D encoder_path=<path> datamodule.cfg.mode=t1
  ```
- [ ] Training running without errors
- [ ] Can see loss decreasing in logs

### Alternative: Without Pretraining
- [ ] Started training without pretraining
  ```bash
  python run.py experiment=cDDPM/DDPM_cond_spark_2D model.cfg.pretrained_encoder=False
  ```

## Evaluation

- [ ] Evaluation runs automatically during training
- [ ] Results saved to `$LOG_DIR/logs/train/runs/<timestamp>/`
- [ ] Can see Dice scores, AUROC, AUPRC for test sets
- [ ] BraTS evaluation completed
- [ ] MOOD evaluation completed
- [ ] IXI test evaluation completed

## Post-Training

- [ ] Best checkpoint identified
- [ ] Training curves reviewed
- [ ] Test set performance analyzed
- [ ] Sample reconstructions and anomaly maps inspected
- [ ] Results documented

## Troubleshooting (If Issues Arise)

- [ ] Checked error messages carefully
- [ ] Verified CSV file formats match templates
- [ ] Confirmed all file paths are correct
- [ ] Ensured conda environment is activated
- [ ] Checked available GPU memory
- [ ] Reviewed DATASET_SETUP_GUIDE.md for solutions
- [ ] Reduced batch size if out of memory
- [ ] Re-ran verification script after fixing issues

## Optional: T2 Training

If training on T2 images instead of T1:

- [ ] Generated T2 CSV files (with `--modality t2`)
- [ ] Ran verification for T2 data
- [ ] Pretrained encoder on T2
  ```bash
  python run.py experiment=cDDPM/Spark_2D_pretrain datamodule.cfg.mode=t2
  ```
- [ ] Trained full model on T2
  ```bash
  python run.py experiment=cDDPM/DDPM_cond_spark_2D encoder_path=<path> datamodule.cfg.mode=t2
  ```

## Optional: Cross-Validation

If running 5-fold cross-validation:

- [ ] Fold 0 completed
- [ ] Fold 1 completed
- [ ] Fold 2 completed
- [ ] Fold 3 completed
- [ ] Fold 4 completed
- [ ] Results aggregated across folds

---

## Quick Reference Commands

```bash
# Setup
export DATA_DIR=/path/to/data
export LOG_DIR=/path/to/logs
conda activate cddpm-uad

# Generate CSVs (example for IXI T1, fold 0)
python scripts/generate_csv_splits.py --data_dir $DATA_DIR --dataset ixi --modality t1 --split train --fold 0

# Verify
python scripts/verify_dataset.py --data_dir $DATA_DIR

# Train
python run.py experiment=cDDPM/Spark_2D_pretrain datamodule.cfg.mode=t1
python run.py experiment=cDDPM/DDPM_cond_spark_2D encoder_path=<path> datamodule.cfg.mode=t1
```

---

**Progress Tracking**: Check off items as you complete them. If you get stuck on any step, refer to the detailed guides:
- `QUICKSTART.md` - Fast setup guide
- `DATASET_SETUP_GUIDE.md` - Detailed documentation
- `scripts/README.md` - Script usage help

# Quick Start Guide - Running on Your Custom Datasets

This guide will help you quickly set up and run the model on your IXI (T1/T2), MOOD, and BraTS datasets.

## Prerequisites

1. **Conda environment** is set up:
   ```bash
   conda activate cddpm-uad
   ```

2. **Your data** is organized with:
   - IXI T1 and T2 images (healthy brains for training)
   - MOOD dataset (for testing)
   - BraTS dataset (for testing)
   - Brain masks for all images
   - Ground truth segmentations for test datasets (if available)

## Step-by-Step Setup

### 1. Configure Paths

Edit `pc_environment.env`:
```bash
DATA_DIR=/path/to/your/data/directory
LOG_DIR=/path/to/save/logs/and/checkpoints
```

### 2. Organize Your Data

Create this directory structure:
```
$DATA_DIR/
├── Data/
│   ├── Train/
│   │   └── ixi/
│   │       ├── t1/              # Your IXI T1 images
│   │       ├── t2/              # Your IXI T2 images
│   │       └── mask/            # Brain masks for IXI
│   ├── Test/
│   │   ├── Brats21/
│   │   │   ├── t1/ or t2/
│   │   │   ├── mask/
│   │   │   └── seg/             # Ground truth segmentations
│   │   └── MOOD/
│   │       ├── t1/ or t2/
│   │       ├── mask/
│   │       └── seg/             # Ground truth (if available)
│   └── splits/                  # Will contain CSV files
```

**Important naming conventions:**
- Images: `<ID>_t1.nii.gz` or `<ID>_t2.nii.gz`
- Masks: `<ID>_mask.nii.gz`
- Segmentations: `<ID>_seg.nii.gz`

### 3. Generate CSV Split Files

Use the provided script to automatically create CSV files:

```bash
# For IXI training data (T1)
python scripts/generate_csv_splits.py \
    --data_dir $DATA_DIR \
    --dataset ixi \
    --modality t1 \
    --split train \
    --fold 0

# For IXI validation (create all 5 folds for cross-validation)
for fold in {0..4}; do
    python scripts/generate_csv_splits.py \
        --data_dir $DATA_DIR \
        --dataset ixi \
        --modality t1 \
        --split val \
        --fold $fold
done

# For IXI test set
python scripts/generate_csv_splits.py \
    --data_dir $DATA_DIR \
    --dataset ixi \
    --modality t1 \
    --split test

# For BraTS test data
python scripts/generate_csv_splits.py \
    --data_dir $DATA_DIR \
    --dataset brats \
    --modality t2 \
    --split test

# For BraTS validation
python scripts/generate_csv_splits.py \
    --data_dir $DATA_DIR \
    --dataset brats \
    --modality t2 \
    --split val

# For MOOD test data
python scripts/generate_csv_splits.py \
    --data_dir $DATA_DIR \
    --dataset mood \
    --modality t2 \
    --split test

# For MOOD validation
python scripts/generate_csv_splits.py \
    --data_dir $DATA_DIR \
    --dataset mood \
    --modality t2 \
    --split val
```

**Alternative:** If your dataset is small or you want more control, manually create CSV files using the templates in `Data/splits/templates/`.

### 4. Verify Your Setup

Run the verification script to ensure everything is set up correctly:

```bash
python scripts/verify_dataset.py --data_dir $DATA_DIR
```

This will check:
- ✓ Directory structure
- ✓ CSV files exist
- ✓ All referenced files exist
- ✓ Environment is configured

If any checks fail, the script will tell you what needs to be fixed.

### 5. Train the Model

#### Option A: Train without pretraining (faster, but lower performance)

```bash
python run.py experiment=cDDPM/DDPM_cond_spark_2D \
    model.cfg.pretrained_encoder=False \
    datamodule.cfg.mode=t1
```

#### Option B: Train with encoder pretraining (RECOMMENDED - better performance)

**Step 1:** Pretrain the encoder using masked pretraining (Spark)
```bash
python run.py experiment=cDDPM/Spark_2D_pretrain \
    datamodule.cfg.mode=t1
```

This will save checkpoints to `$LOG_DIR/logs/train/runs/<timestamp>/checkpoints/`. The best checkpoint path will be printed when training completes.

**Step 2:** Train the full model with the pretrained encoder
```bash
python run.py experiment=cDDPM/DDPM_cond_spark_2D \
    encoder_path=$LOG_DIR/logs/train/runs/<timestamp>/checkpoints/best.ckpt \
    datamodule.cfg.mode=t1
```

Replace `<timestamp>` with the actual timestamp from Step 1.

### 6. Train on T2 Images (instead of T1)

Simply change the `mode` parameter:

```bash
# Pretrain on T2
python run.py experiment=cDDPM/Spark_2D_pretrain \
    datamodule.cfg.mode=t2

# Train full model on T2
python run.py experiment=cDDPM/DDPM_cond_spark_2D \
    encoder_path=<path_to_pretrained_encoder> \
    datamodule.cfg.mode=t2
```

### 7. Evaluation

The model automatically evaluates on all test datasets during training. It will test on:
- ✓ BraTS dataset
- ✓ MOOD dataset  
- ✓ IXI test set (healthy controls)
- ✓ MSLUB (if you have it)

Results are saved to `$LOG_DIR/logs/train/runs/<timestamp>/`.

To run evaluation only (without training):

```bash
python run.py experiment=cDDPM/DDPM_cond_spark_2D \
    encoder_path=<path_to_trained_model> \
    trainer.max_epochs=0
```

## Expected Output

During training, you'll see:
- Training loss decreasing
- Validation metrics
- Periodic evaluation on test sets with segmentation metrics (Dice, AUPRC, etc.)
- Checkpoints saved to `$LOG_DIR`

The evaluation will report:
- **Dice Score**: Overlap between predicted and ground truth anomalies
- **AUPRC**: Area under precision-recall curve
- **AUROC**: Area under ROC curve
- Sample reconstructions and anomaly maps

## Configuration Options

### Common adjustments in `configs/datamodule/IXI.yaml`:

```yaml
batch_size: 32              # Reduce if out of memory
num_workers: 4              # Adjust based on CPU cores
mode: t1                    # 't1' or 't2'
sample_set: False           # Set to True for quick debugging
```

### Training parameters in `configs/trainer/default.yaml`:

```yaml
max_epochs: 500             # Number of training epochs
gpus: 1                     # Number of GPUs
```

## Troubleshooting

### Out of Memory Errors
```bash
python run.py experiment=cDDPM/DDPM_cond_spark_2D \
    datamodule.cfg.batch_size=16 \
    encoder_path=<path>
```

### Missing Files
Run verification script and check error messages:
```bash
python scripts/verify_dataset.py --data_dir $DATA_DIR
```

### Wrong File Paths in CSV
CSV paths should be relative to DATA_DIR and start with `/Data/Train/` or `/Data/Test/`:
```
CORRECT:   /Train/ixi/t1/image001_t1.nii.gz
INCORRECT: Train/ixi/t1/image001_t1.nii.gz
INCORRECT: /Users/name/data/Train/ixi/t1/image001_t1.nii.gz
```

### Dimension Mismatch
Ensure masks have same dimensions as images:
```bash
python scripts/verify_dataset.py --data_dir $DATA_DIR --check_dimensions
```

Use preprocessing scripts to fix:
```bash
python preprocessing/resample.py --input mask.nii.gz --reference image.nii.gz --output mask_resampled.nii.gz
```

## Cross-Validation

To run 5-fold cross-validation on IXI:

```bash
for fold in {0..4}; do
    # Pretrain encoder for this fold
    python run.py experiment=cDDPM/Spark_2D_pretrain \
        datamodule.fold=$fold
    
    # Train full model for this fold
    python run.py experiment=cDDPM/DDPM_cond_spark_2D \
        datamodule.fold=$fold \
        encoder_path=<path_from_pretraining>
done
```

## Next Steps

Once training is complete:
- Check results in `$LOG_DIR/logs/train/runs/<timestamp>/`
- View training curves with TensorBoard or W&B (if configured)
- Analyze test set performance in evaluation outputs
- Use the best checkpoint for inference on new data

## Additional Resources

- Full setup details: `DATASET_SETUP_GUIDE.md`
- Original paper: [Guided Reconstruction with Conditioned Diffusion Models](https://www.sciencedirect.com/science/article/pii/S0010482525000101)
- Original repository: https://github.com/FinnBehrendt/Conditioned-Diffusion-Models-UAD
- Example CSV templates: `Data/splits/templates/`

## Summary of Commands

```bash
# 1. Setup
conda activate cddpm-uad
export DATA_DIR=/path/to/data
export LOG_DIR=/path/to/logs

# 2. Generate CSV files
python scripts/generate_csv_splits.py --data_dir $DATA_DIR --dataset ixi --modality t1 --split train --fold 0
# ... (repeat for other splits)

# 3. Verify
python scripts/verify_dataset.py --data_dir $DATA_DIR

# 4. Pretrain
python run.py experiment=cDDPM/Spark_2D_pretrain datamodule.cfg.mode=t1

# 5. Train
python run.py experiment=cDDPM/DDPM_cond_spark_2D encoder_path=<path> datamodule.cfg.mode=t1

# 6. Results in $LOG_DIR/logs/train/runs/<timestamp>/
```

Good luck with your training! 🚀

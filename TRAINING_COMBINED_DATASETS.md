# Training on Combined Datasets (IXI T1 + IXI T2 + MOOD)

## Overview

You want to:
- ✅ **Train on**: IXI T1 + IXI T2 (and optionally MOOD if you have healthy subjects)
- ✅ **Test on**: BraTS (brain tumors)

This setup allows the model to learn from diverse healthy brain MRI data across multiple modalities and acquisition protocols.

**IMPORTANT**: The MOOD dataset typically contains ONLY anomalous images. This guide assumes:
- **Most common case**: Train on IXI T1 + T2 only, test on BraTS and MOOD
- **If you have healthy MOOD**: You can include them in training (see optional sections)

## Important Notes

### About MOOD Dataset
The MOOD dataset typically contains **anomalous** images. For training, you need:
- **Healthy MOOD images only** (label=0) in the training set
- **Anomalous MOOD images** (label=1) in the test set

If all your MOOD images have anomalies, you should:
- **Option A**: Use only IXI T1 + T2 for training, test on BraTS and MOOD
- **Option B**: If MOOD has healthy subjects, separate them for training

## Directory Structure

```
$DATA_DIR/Data/
├── Train/
│   ├── ixi/
│   │   ├── t1/              # IXI T1 images (healthy)
│   │   ├── t2/              # IXI T2 images (healthy)
│   │   └── mask/            # Brain masks for IXI
│   └── mood/
│       ├── t1/ or t2/       # MOOD HEALTHY images only (if available)
│       └── mask/            # Brain masks for MOOD
├── Test/
│   ├── Brats21/
│   │   ├── t1/ or t2/       # BraTS images (with tumors)
│   │   ├── mask/            # Brain masks
│   │   └── seg/             # Ground truth segmentations
│   └── MOOD/
│       ├── t1/ or t2/       # MOOD test images (with anomalies)
│       ├── mask/            # Brain masks
│       └── seg/             # Ground truth (if available)
└── splits/                  # CSV files
```

## Step-by-Step Setup

### 1. Generate CSV Files for Combined Training

You need separate CSV files for T1 and T2 to keep them organized:

```bash
export DATA_DIR=/path/to/your/data

# Generate IXI T1 splits (for all 5 folds)
for fold in {0..4}; do
    python scripts/generate_csv_splits.py \
        --data_dir $DATA_DIR \
        --dataset ixi \
        --modality t1 \
        --split train \
        --fold $fold \
        --output_name IXI_T1_train_fold${fold}.csv
    
    python scripts/generate_csv_splits.py \
        --data_dir $DATA_DIR \
        --dataset ixi \
        --modality t1 \
        --split val \
        --fold $fold \
        --output_name IXI_T1_val_fold${fold}.csv
done

python scripts/generate_csv_splits.py \
    --data_dir $DATA_DIR \
    --dataset ixi \
    --modality t1 \
    --split test \
    --output_name IXI_T1_test.csv

# Generate IXI T2 splits (for all 5 folds)
for fold in {0..4}; do
    python scripts/generate_csv_splits.py \
        --data_dir $DATA_DIR \
        --dataset ixi \
        --modality t2 \
        --split train \
        --fold $fold \
        --output_name IXI_T2_train_fold${fold}.csv
    
    python scripts/generate_csv_splits.py \
        --data_dir $DATA_DIR \
        --dataset ixi \
        --modality t2 \
        --split val \
        --fold $fold \
        --output_name IXI_T2_val_fold${fold}.csv
done

python scripts/generate_csv_splits.py \
    --data_dir $DATA_DIR \
    --dataset ixi \
    --modality t2 \
    --split test \
    --output_name IXI_T2_test.csv

# Generate MOOD training splits (HEALTHY subjects only!)
# Make sure these CSVs only contain healthy MOOD images (label=0)
python scripts/generate_csv_splits.py \
    --data_dir $DATA_DIR \
    --dataset mood \
    --modality t2 \
    --split train \
    --output_name MOOD_train.csv

python scripts/generate_csv_splits.py \
    --data_dir $DATA_DIR \
    --dataset mood \
    --modality t2 \
    --split val \
    --output_name MOOD_val_train.csv

# Generate BraTS test splits
python scripts/generate_csv_splits.py \
    --data_dir $DATA_DIR \
    --dataset brats \
    --modality t2 \
    --split val

python scripts/generate_csv_splits.py \
    --data_dir $DATA_DIR \
    --dataset brats \
    --modality t2 \
    --split test

# Generate MOOD test splits (anomalous images)
python scripts/generate_csv_splits.py \
    --data_dir $DATA_DIR \
    --dataset mood \
    --modality t2 \
    --split test \
    --output_name MOOD_test.csv

python scripts/generate_csv_splits.py \
    --data_dir $DATA_DIR \
    --dataset mood \
    --modality t2 \
    --split val \
    --output_name MOOD_val.csv
```

### 2. Verify Your Setup

```bash
python scripts/verify_dataset.py --data_dir $DATA_DIR
```

Make sure:
- All IXI_T1 and IXI_T2 CSV files exist
- MOOD training CSVs contain only healthy subjects (label=0)
- BraTS test CSVs contain anomalous subjects (label=1)

### 3. Train the Model

#### Option A: Train without encoder pretraining

```bash
python run.py \
    experiment=cDDPM/DDPM_cond_spark_2D \
    datamodule=IXI_T1_T2_MOOD \
    model.cfg.pretrained_encoder=False
```

#### Option B: Train with encoder pretraining (RECOMMENDED)

**Step 1: Pretrain encoder on combined datasets**

```bash
python run.py \
    experiment=cDDPM/Spark_2D_pretrain \
    datamodule=IXI_T1_T2_MOOD
```

**Step 2: Train full model with pretrained encoder**

```bash
python run.py \
    experiment=cDDPM/DDPM_cond_spark_2D \
    datamodule=IXI_T1_T2_MOOD \
    encoder_path=<path_to_pretrained_encoder>
```

### 4. Evaluation

The model will automatically evaluate on:
- ✅ BraTS test set (primary test dataset)
- ✅ MOOD test set (optional, if configured)

Results will be saved to `$LOG_DIR/logs/train/runs/<timestamp>/`

## Alternative: Train on IXI Only (If MOOD Has No Healthy Subjects)

If your MOOD dataset doesn't have healthy subjects for training, use the original datamodule:

### For IXI T1 Only:
```bash
# Pretrain
python run.py experiment=cDDPM/Spark_2D_pretrain datamodule.cfg.mode=t1

# Train
python run.py experiment=cDDPM/DDPM_cond_spark_2D \
    datamodule.cfg.mode=t1 \
    encoder_path=<path>
```

### For IXI T2 Only:
```bash
# Pretrain
python run.py experiment=cDDPM/Spark_2D_pretrain datamodule.cfg.mode=t2

# Train
python run.py experiment=cDDPM/DDPM_cond_spark_2D \
    datamodule.cfg.mode=t2 \
    encoder_path=<path>
```

### Test on Both BraTS and MOOD

The config already includes both in `testsets`:
```yaml
testsets:
  - Datamodules_eval.Brats21
  - Datamodules_eval.MOOD
```

## Key Configuration

Edit `configs/datamodule/IXI_T1_T2_MOOD.yaml` if needed:

```yaml
batch_size: 32           # Reduce if out of memory
num_workers: 4           # Adjust for your CPU
sample_set: False        # Set True for quick debugging

testsets:
  - Datamodules_eval.Brats21    # Primary test set
  - Datamodules_eval.MOOD       # Optional additional test
```

## Expected Training Data Size

When training on all three:
- IXI T1: ~400-500 subjects
- IXI T2: ~300-400 subjects (subset with T2)
- MOOD healthy: (depends on your dataset)
- **Total**: ~700-1000+ training images

This provides diverse data for robust anomaly detection!

## Troubleshooting

### "MOOD training data contains anomalies"
- Solution: Manually filter MOOD_train.csv and MOOD_val_train.csv to keep only `label=0` rows
- Or: Use only IXI for training, MOOD for testing

### "Out of memory"
- Reduce batch size: `datamodule.cfg.batch_size=16`
- Reduce workers: `datamodule.cfg.num_workers=2`

### "Imbalanced dataset sizes"
The model uses `ConcatDataset`, so larger datasets will have more samples per epoch. This is usually fine, but you can:
- Balance by subsampling larger datasets in the CSV files
- Or let the model see all data (recommended)

## Summary

**Current Setup:**
- ✅ Training on: IXI T1 + IXI T2 + MOOD (healthy subjects)
- ✅ Testing on: BraTS (tumors) + optionally MOOD (anomalies)
- ✅ New datamodule: `IXI_T1_T2_MOOD`
- ✅ Config file: `configs/datamodule/IXI_T1_T2_MOOD.yaml`

**To Use:**
```bash
python run.py experiment=cDDPM/DDPM_cond_spark_2D datamodule=IXI_T1_T2_MOOD
```

This gives your model exposure to multiple modalities and datasets during training, which should improve its ability to detect anomalies in the BraTS test set! 🚀

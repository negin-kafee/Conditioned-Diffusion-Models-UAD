# Setup Complete ✅

## Your Requirements

You want to:
- ✅ **Train on**: IXI T1 + IXI T2 (most common - MOOD typically has no healthy subjects)
- ✅ **Test on**: BraTS dataset (and optionally MOOD)

## What's Ready (FIXED VERSION)

### ✅ Code Implementation

1. **New Combined Datamodule**: `src/datamodules/Datamodules_train_combined.py`
   - Loads and combines IXI T1 and IXI T2 for training
   - Optionally includes healthy MOOD subjects if available
   - **Fixed**: Consistent path handling (absolute/relative), correct seg_path (empty string not None)
   - **Fixed**: Uses correct pipelines (Train for training/val, no test_dataloader conflicts)
   - **Fixed**: Filters MOOD for healthy subjects only (label=0)

2. **Configuration File**: `configs/datamodule/IXI_T1_T2_MOOD.yaml`
   - Pre-configured for IXI T1 + T2 training
   - MOOD training commented out (uncomment only if you have healthy MOOD data)
   - Tests on BraTS and MOOD via separate evaluation datamodules (no conflicts)

3. **Updated Scripts**: `scripts/generate_csv_splits.py`
   - Now supports custom output names (e.g., `IXI_T1_train_fold0.csv`)
   - Can generate separate CSVs for T1 and T2

### ✅ Documentation

- **`TRAINING_COMBINED_DATASETS.md`**: Complete guide for your exact use case
- **`QUICKSTART.md`**: Fast setup guide
- **`DATASET_SETUP_GUIDE.md`**: Detailed documentation
- **`SETUP_CHECKLIST.md`**: Step-by-step checklist

## Quick Start Commands

### 1. Setup Environment
```bash
export DATA_DIR=/path/to/your/data
export LOG_DIR=/path/to/logs
conda activate cddpm-uad
```

### 2. Organize Your Data
```
$DATA_DIR/Data/
├── Train/
│   ├── ixi/t1/              # IXI T1 images
│   ├── ixi/t2/              # IXI T2 images
│   ├── ixi/mask/            # Masks
│   └── mood/t2/             # MOOD healthy images (if any)
│       └── mask/
└── Test/
    └── Brats21/t2/          # BraTS test images
        ├── mask/
        └── seg/
```

### 3. Generate CSV Files

**For IXI T1:**
```bash
for fold in {0..4}; do
    python scripts/generate_csv_splits.py --data_dir $DATA_DIR --dataset ixi --modality t1 --split train --fold $fold --output_name IXI_T1_train_fold${fold}.csv
    python scripts/generate_csv_splits.py --data_dir $DATA_DIR --dataset ixi --modality t1 --split val --fold $fold --output_name IXI_T1_val_fold${fold}.csv
done
python scripts/generate_csv_splits.py --data_dir $DATA_DIR --dataset ixi --modality t1 --split test --output_name IXI_T1_test.csv
```

**For IXI T2:**
```bash
for fold in {0..4}; do
    python scripts/generate_csv_splits.py --data_dir $DATA_DIR --dataset ixi --modality t2 --split train --fold $fold --output_name IXI_T2_train_fold${fold}.csv
    python scripts/generate_csv_splits.py --data_dir $DATA_DIR --dataset ixi --modality t2 --split val --fold $fold --output_name IXI_T2_val_fold${fold}.csv
done
python scripts/generate_csv_splits.py --data_dir $DATA_DIR --dataset ixi --modality t2 --split test --output_name IXI_T2_test.csv
```

**For MOOD (ONLY IF you have healthy subjects - most MOOD datasets don't):**
```bash
# First, manually filter your MOOD data to separate healthy (label=0) from anomalous (label=1)
# Then generate CSVs for healthy subjects only:
python scripts/generate_csv_splits.py --data_dir $DATA_DIR --dataset mood --modality t2 --split train --output_name MOOD_train.csv
python scripts/generate_csv_splits.py --data_dir $DATA_DIR --dataset mood --modality t2 --split val --output_name MOOD_val_train.csv
# Important: Manually verify these CSVs contain ONLY label=0 (healthy) subjects
```

**For BraTS (test only):**
```bash
python scripts/generate_csv_splits.py --data_dir $DATA_DIR --dataset brats --modality t2 --split val
python scripts/generate_csv_splits.py --data_dir $DATA_DIR --dataset brats --modality t2 --split test
```

### 4. Verify Setup
```bash
python scripts/verify_dataset.py --data_dir $DATA_DIR
```

### 5. Train the Model

**With encoder pretraining (recommended):**
```bash
# Step 1: Pretrain encoder
python run.py experiment=cDDPM/Spark_2D_pretrain datamodule=IXI_T1_T2_MOOD

# Step 2: Train full model
python run.py experiment=cDDPM/DDPM_cond_spark_2D \
    datamodule=IXI_T1_T2_MOOD \
    encoder_path=<path_from_pretraining>
```

**Without pretraining (faster but lower performance):**
```bash
python run.py experiment=cDDPM/DDPM_cond_spark_2D \
    datamodule=IXI_T1_T2_MOOD \
    model.cfg.pretrained_encoder=False
```

## Important Notes

### ⚠️ About MOOD Dataset

The MOOD dataset typically contains **ONLY anomalous images**. 

**Default configuration (recommended):**
- Train on **IXI T1 + T2 only**
- Test on **BraTS + MOOD** (both contain anomalies)
- The config is set up for this by default (MOOD training is commented out)

**If you have healthy MOOD subjects** (uncommon):
1. Manually separate healthy (label=0) from anomalous (label=1) MOOD images
2. Create `MOOD_train.csv` and `MOOD_val_train.csv` with only healthy subjects
3. Uncomment the MOOD training section in `configs/datamodule/IXI_T1_T2_MOOD.yaml`
4. The datamodule will automatically filter for label=0 as a safety check

### CSV File Structure

After generation, you should have:
```
Data/splits/
├── IXI_T1_train_fold0.csv to IXI_T1_train_fold4.csv
├── IXI_T1_val_fold0.csv to IXI_T1_val_fold4.csv
├── IXI_T1_test.csv
├── IXI_T2_train_fold0.csv to IXI_T2_train_fold4.csv
├── IXI_T2_val_fold0.csv to IXI_T2_val_fold4.csv
├── IXI_T2_test.csv
├── MOOD_train.csv (healthy only!)
├── MOOD_val_train.csv (healthy only!)
├── Brats21_test.csv
└── Brats21_val.csv
```

## What Gets Evaluated

During training, the model will automatically test on:
- ✅ **BraTS dataset** (primary test target)
- ✅ **MOOD dataset** (if configured, optional)

Results include:
- Dice Score
- AUROC (Area Under ROC Curve)
- AUPRC (Area Under Precision-Recall Curve)
- Sample reconstructions and anomaly maps

## Configuration Options

Edit `configs/datamodule/IXI_T1_T2_MOOD.yaml` to adjust:

```yaml
batch_size: 32           # Reduce if out of memory
num_workers: 4           # CPU cores for data loading
sample_set: False        # Set True for quick debugging

testsets:
  - Datamodules_eval.Brats21    # Your main test dataset
  - Datamodules_eval.MOOD       # Optional
```

## Summary

Everything is now correctly configured:

✅ **Training**: IXI T1 + IXI T2 (standard case)  
✅ **Testing**: BraTS + MOOD (handled by separate evaluation datamodules)  
✅ **Fixed Issues**: Path handling, seg_path handling, pipeline usage, evaluation conflicts  
✅ **Optional**: Can include healthy MOOD subjects if you have them  
✅ **Scripts**: Ready to generate all CSV files  
✅ **Documentation**: Complete guides available  
✅ **Commands**: Copy-paste ready

## Next Steps

1. ✅ Follow `TRAINING_COMBINED_DATASETS.md` for detailed instructions
2. → Organize your data in the correct directory structure
3. → Generate CSV files using the commands above
4. → Verify with `verify_dataset.py`
5. → Start training!

## Need Help?

Refer to:
- **`TRAINING_COMBINED_DATASETS.md`** - Your specific use case
- **`QUICKSTART.md`** - Fast setup
- **`DATASET_SETUP_GUIDE.md`** - Detailed reference
- **`SETUP_CHECKLIST.md`** - Track your progress

---

**Ready to go!** 🚀 Everything is set up for training on IXI T1 + T2 + MOOD and testing on BraTS.

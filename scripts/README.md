# Dataset Preparation Scripts

This directory contains utility scripts to help you prepare your datasets for training and evaluation.

## Scripts

### 1. `generate_csv_splits.py`

Automatically generates CSV split files from your organized data directory.

**Usage:**
```bash
# Generate training split for IXI
python generate_csv_splits.py \
    --data_dir /path/to/DATA_DIR \
    --dataset ixi \
    --modality t1 \
    --split train \
    --fold 0

# Generate test split for BraTS
python generate_csv_splits.py \
    --data_dir /path/to/DATA_DIR \
    --dataset brats \
    --modality t2 \
    --split test

# Generate test split for MOOD
python generate_csv_splits.py \
    --data_dir /path/to/DATA_DIR \
    --dataset mood \
    --modality t2 \
    --split test

# Verify an existing CSV file
python generate_csv_splits.py \
    --data_dir /path/to/DATA_DIR \
    --verify Data/splits/IXI_train_fold0.csv
```

**Parameters:**
- `--data_dir`: Path to your data directory (required)
- `--dataset`: Dataset name: 'ixi', 'brats', or 'mood'
- `--modality`: MRI modality: 't1' or 't2'
- `--split`: Data split: 'train', 'val', or 'test'
- `--fold`: Fold number for cross-validation (0-4, IXI only)
- `--verify`: Path to CSV file to verify (relative to data_dir)

**What it does:**
- Scans your data directory for NIfTI files
- Matches images with their corresponding masks (and segmentations for test data)
- Creates properly formatted CSV files
- Automatically splits data into train/val/test sets
- Verifies that all files exist

**Requirements:**
Your data must follow the naming convention:
- Images: `<ID>_t1.nii.gz` or `<ID>_t2.nii.gz`
- Masks: `<ID>_mask.nii.gz`
- Segmentations: `<ID>_seg.nii.gz`

### 2. `verify_dataset.py`

Comprehensive verification of your dataset setup before training.

**Usage:**
```bash
# Basic verification
python verify_dataset.py --data_dir /path/to/DATA_DIR

# Include dimension checking
python verify_dataset.py \
    --data_dir /path/to/DATA_DIR \
    --check_dimensions

# Detailed error reporting
python verify_dataset.py \
    --data_dir /path/to/DATA_DIR \
    --detailed
```

**Parameters:**
- `--data_dir`: Path to your data directory (required)
- `--check_dimensions`: Also verify that image and mask dimensions match
- `--detailed`: Show detailed error information

**What it checks:**
- ✓ Directory structure exists
- ✓ CSV files are present
- ✓ All files referenced in CSVs exist
- ✓ Image and mask dimensions match (if requested)
- ✓ Environment configuration is set up
- ✓ Counts of images in each directory

**Example output:**
```
=== Checking Directory Structure ===
  ✓ Data/Train/ixi/t1
  ✓ Data/Train/ixi/t2
  ✓ Data/Train/ixi/mask
  ✓ Data/Test
  ✓ Data/splits

=== Checking CSV Files ===
  Found 7 CSV files:
    ✓ IXI_train_fold0.csv (389 entries)
    ✓ IXI_val_fold0.csv (97 entries)
    ✓ IXI_test.csv (100 entries)
    ...

=== Verifying CSV File Contents ===
  ✓ IXI_train_fold0.csv: 389 entries
  ✓ Brats21_test.csv: 1153 entries
  ✓ MOOD_test.csv: 250 entries

✓ All checks passed! You're ready to start training.
```

## Workflow

Here's the recommended workflow for setting up your datasets:

1. **Organize your data** into the required directory structure
2. **Run `generate_csv_splits.py`** for each dataset and split
3. **Run `verify_dataset.py`** to ensure everything is correct
4. **Start training!**

### Example Complete Setup

```bash
# Set your data directory
export DATA_DIR=/mnt/data/brain_mri

# Generate all IXI splits (for 5-fold cross-validation)
for fold in {0..4}; do
    python generate_csv_splits.py --data_dir $DATA_DIR --dataset ixi --modality t1 --split train --fold $fold
    python generate_csv_splits.py --data_dir $DATA_DIR --dataset ixi --modality t1 --split val --fold $fold
done

python generate_csv_splits.py --data_dir $DATA_DIR --dataset ixi --modality t1 --split test

# Generate test dataset splits
python generate_csv_splits.py --data_dir $DATA_DIR --dataset brats --modality t2 --split val
python generate_csv_splits.py --data_dir $DATA_DIR --dataset brats --modality t2 --split test

python generate_csv_splits.py --data_dir $DATA_DIR --dataset mood --modality t2 --split val
python generate_csv_splits.py --data_dir $DATA_DIR --dataset mood --modality t2 --split test

# Verify everything
python verify_dataset.py --data_dir $DATA_DIR --check_dimensions

# If all checks pass, you're ready to train!
python ../run.py experiment=cDDPM/Spark_2D_pretrain
```

## Troubleshooting

### "No images found in directory"
- Check that your images are in `.nii.gz` format
- Verify the directory path is correct
- Ensure images follow naming convention: `*_t1.nii.gz` or `*_t2.nii.gz`

### "Mask not found for image"
- Masks must have the same base name as images but with `_mask.nii.gz` suffix
- Example: `IXI001_t1.nii.gz` → `IXI001_mask.nii.gz`

### "Dimension mismatch"
- Use preprocessing tools to resample masks to match image dimensions:
  ```bash
  python ../preprocessing/resample.py \
      --input mask.nii.gz \
      --reference image.nii.gz \
      --output mask_resampled.nii.gz
  ```

### Missing packages
If you get import errors, ensure the conda environment is activated:
```bash
conda activate cddpm-uad
```

## Tips

1. **Start small**: Test with a small subset first using the `sample_set: True` option in configs
2. **Verify often**: Run `verify_dataset.py` after any changes to your data
3. **Backup CSVs**: Once you have working CSV files, back them up
4. **Check logs**: If training fails, check the CSV files first - they're the most common source of errors

## Additional Resources

- Full documentation: `../DATASET_SETUP_GUIDE.md`
- Quick start: `../QUICKSTART.md`
- CSV templates: `../Data/splits/templates/`
- Preprocessing tools: `../preprocessing/`

## Need Help?

If you encounter issues:
1. Run `verify_dataset.py` and read the error messages carefully
2. Check that your data follows the expected directory structure
3. Verify file naming conventions match the examples
4. Review the CSV template files for the correct format

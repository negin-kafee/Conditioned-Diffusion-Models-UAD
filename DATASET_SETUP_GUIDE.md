# Dataset Setup Guide for Custom Datasets

This guide explains how to prepare your own datasets (IXI T1/T2, MOOD, and BraTS) for training and testing with the Conditioned Diffusion Models for Unsupervised Anomaly Detection.

## Overview

The model expects:
- **Training Data**: Healthy brain MRI images (IXI T1 and T2)
- **Test Data with Anomalies**: Brain MRI with pathologies (MOOD, BraTS)
- **CSV Split Files**: Metadata files specifying which images to use for training/validation/testing

## Directory Structure

Your data directory should be organized as follows:

```
<DATA_DIR>/
├── Data/
│   ├── Train/
│   │   ├── ixi/
│   │   │   ├── t1/              # IXI T1 images (healthy)
│   │   │   │   └── *.nii.gz
│   │   │   ├── t2/              # IXI T2 images (healthy)
│   │   │   │   └── *.nii.gz
│   │   │   └── mask/            # Brain masks for IXI
│   │   │       └── *_mask.nii.gz
│   │   └── mood/                # MOOD training data (if any healthy subjects)
│   │       ├── t1/ or t2/
│   │       └── mask/
│   ├── Test/
│   │   ├── Brats21/
│   │   │   ├── t1/              # BraTS T1 images
│   │   │   │   └── *.nii.gz
│   │   │   ├── t2/              # BraTS T2 images  
│   │   │   │   └── *.nii.gz
│   │   │   ├── mask/            # Brain masks
│   │   │   │   └── *_mask.nii.gz
│   │   │   └── seg/             # Ground truth segmentations
│   │   │       └── *_seg.nii.gz
│   │   └── MOOD/
│   │       ├── t1/ or t2/       # MOOD test images
│   │       ├── mask/            # Brain masks
│   │       └── seg/             # Ground truth (if available)
│   └── splits/
│       ├── IXI_train_fold0.csv
│       ├── IXI_val_fold0.csv
│       ├── IXI_test.csv
│       ├── Brats21_val.csv
│       ├── Brats21_test.csv
│       ├── MOOD_val.csv
│       └── MOOD_test.csv
```

## CSV File Format

Each CSV file contains metadata about the images. The required columns are:

### For Training Data (IXI)

```csv
,img_name,SEX_ID (1=m, 2=f),HEIGHT,WEIGHT,ETHNIC_ID,MARITAL_ID,OCCUPATION_ID,QUALIFICATION_ID,DATE_AVAILABLE,STUDY_DATE,age,label,img_path,mask_path,seg_path,Agegroup
0,IXI001-HH-1234_t1.nii.gz,1,175,70,1,2,1,4,1,2005-08-12,55.83,0,/Train/ixi/t1/IXI001-HH-1234_t1.nii.gz,/Train/ixi/mask/IXI001-HH-1234_mask.nii.gz,,2.0
```

**Required columns**:
- `img_name`: Filename of the image
- `age`: Patient age (can be empty if not available)
- `label`: 0 for healthy, 1 for anomaly
- `img_path`: Relative path from DATA_DIR to image file
- `mask_path`: Relative path from DATA_DIR to brain mask
- `seg_path`: Empty for training data, path to segmentation for test data

### For Test Data (BraTS, MOOD)

```csv
,img_name,age,label,img_path,mask_path,seg_path
0,BraTS2021_00001_t1.nii.gz,45.5,1,/Test/Brats21/t1/BraTS2021_00001_t1.nii.gz,/Test/Brats21/mask/BraTS2021_00001_mask.nii.gz,/Test/Brats21/seg/BraTS2021_00001_seg.nii.gz
```

**Required columns**:
- `img_name`: Filename of the image
- `age`: Patient age (can be empty)
- `label`: 1 for images with anomalies, 0 for healthy
- `img_path`: Relative path from DATA_DIR to image file
- `mask_path`: Relative path from DATA_DIR to brain mask
- `seg_path`: Relative path to ground truth segmentation (can be empty if not available)

## Image Requirements

### File Format
- All images must be in NIfTI format (`.nii` or `.nii.gz`)
- Recommended: `.nii.gz` (compressed)

### Image Properties
- **3D volumes**: MRI brain scans
- **Intensity normalization**: Will be handled by the preprocessing pipeline
- **Orientation**: Standard neuroimaging orientation (RAS or LPS)
- **Resolution**: Images will be resampled during preprocessing

### Brain Masks
- Binary masks (0 for background, 1 for brain tissue)
- Same spatial dimensions as the corresponding image
- If you don't have masks, they can be generated using tools like FSL's BET or HD-BET

### Segmentations (for test data)
- Binary or multi-class masks indicating anomaly locations
- Same spatial dimensions as the corresponding image
- 0 for healthy tissue, 1+ for anomalies

## Setup Steps

### 1. Configure Environment

Edit `pc_environment.env`:

```bash
DATA_DIR=/path/to/your/data
LOG_DIR=/path/to/logs
```

### 2. Organize Your Data

Place your MRI images in the directory structure shown above.

### 3. Create CSV Split Files

You have two options:

#### Option A: Manual Creation
Create CSV files manually following the format above. See example templates in `Data/splits/templates/`.

#### Option B: Use the Generation Script
Use the provided script to automatically generate CSV files from your data directory:

```bash
python scripts/generate_csv_splits.py \
    --data_dir /path/to/DATA_DIR \
    --dataset ixi \
    --modality t1 \
    --split train \
    --fold 0
```

### 4. Preprocessing (Optional but Recommended)

The repository includes preprocessing scripts for standardizing your data:

```bash
# For IXI dataset
bash preprocessing/prepare_IXI.sh /path/to/raw/IXI /path/to/DATA_DIR/Data/Train/ixi

# For BraTS dataset
bash preprocessing/prepare_Brats21.sh /path/to/raw/BraTS /path/to/DATA_DIR/Data/Test/Brats21
```

Key preprocessing steps:
- N4 bias field correction (`preprocessing/n4filter.py`)
- Registration to standard template (`preprocessing/registration.py`)
- Resampling to consistent resolution (`preprocessing/resample.py`)
- Brain extraction (`preprocessing/extract_masks.py`)

## Running Training

### Train on IXI T1 Data

```bash
# Without encoder pretraining
python run.py experiment=cDDPM/DDPM_cond_spark_2D model.cfg.pretrained_encoder=False

# With encoder pretraining (recommended for better performance)
# Step 1: Pretrain encoder
python run.py experiment=cDDPM/Spark_2D_pretrain

# Step 2: Train full model with pretrained encoder
python run.py experiment=cDDPM/DDPM_cond_spark_2D encoder_path=<path_to_pretrained_encoder>
```

### Train on IXI T2 Data

Modify the configuration to use T2 images:

```bash
python run.py experiment=cDDPM/DDPM_cond_spark_2D datamodule.cfg.mode=t2
```

## Running Evaluation

### Test on BraTS Dataset

The model will automatically evaluate on all test sets specified in the config:

```bash
python run.py experiment=cDDPM/DDPM_cond_spark_2D \
    encoder_path=<path_to_pretrained_encoder> \
    trainer.max_epochs=0  # Skip training, only evaluate
```

### Test on MOOD Dataset

MOOD dataset will be included automatically if you've:
1. Added MOOD data to your directory structure
2. Created MOOD CSV files
3. The config file includes `Datamodules_eval.MOOD` in the testsets list (already done)

## Common Issues and Solutions

### Issue: "FileNotFoundError: Image not found"
**Solution**: Check that:
- Paths in CSV files are relative to DATA_DIR
- Paths start with `/Data/Train/` or `/Data/Test/`
- File extensions match exactly (`.nii.gz` vs `.nii`)

### Issue: "Shape mismatch between image and mask"
**Solution**: 
- Ensure masks have the same spatial dimensions as images
- Use `preprocessing/resample.py` to resample masks to match images

### Issue: "Missing segmentation files"
**Solution**: 
- For MOOD dataset without segmentations, set `seg_path` to empty or NaN in CSV
- The datamodule will handle missing segmentations automatically

### Issue: "Out of memory during training"
**Solution**: 
- Reduce batch size in config: `datamodule.cfg.batch_size=16`
- Reduce number of workers: `datamodule.cfg.num_workers=2`
- Use smaller image dimensions: `datamodule.cfg.imageDim=[128,128,128]`

## Advanced Configuration

### Multi-fold Cross-Validation

To train on different folds:

```bash
python run.py experiment=cDDPM/DDPM_cond_spark_2D datamodule.fold=0
python run.py experiment=cDDPM/DDPM_cond_spark_2D datamodule.fold=1
# ... etc
```

### Custom Data Augmentation

Edit `configs/datamodule/IXI.yaml`:

```yaml
# Enable augmentations
randomRotate: True
rotateDegree: 10
horizontalFlip: True
randomBrightness: True
brightnessRange: (0.7,1.3)
```

### Using Mixed Datasets for Training

To train on both IXI and MOOD healthy subjects, create a combined CSV file or modify the datamodule to load from multiple sources.

## Verification Checklist

Before running training, verify:

- [ ] `pc_environment.env` has correct DATA_DIR and LOG_DIR
- [ ] Data directory structure matches expected format
- [ ] All images are in NIfTI format
- [ ] CSV files exist for all splits (train/val/test)
- [ ] CSV paths are relative to DATA_DIR
- [ ] All files referenced in CSVs exist
- [ ] Brain masks have same dimensions as images
- [ ] Training data is labeled as healthy (label=0)
- [ ] Test data with anomalies is labeled (label=1)
- [ ] Environment is set up: `conda activate cddpm-uad`

## Getting Help

If you encounter issues:
1. Check error messages carefully
2. Verify CSV file formats match examples
3. Ensure all paths are correct
4. Check that preprocessing was applied consistently
5. Review the original paper and README for additional details

## Example Workflow

Here's a complete example workflow:

```bash
# 1. Set up environment
conda activate cddpm-uad
export DATA_DIR=/mnt/data/brain_mri
export LOG_DIR=/mnt/logs/cddpm

# 2. Generate CSV files for your datasets
python scripts/generate_csv_splits.py --data_dir $DATA_DIR --dataset ixi --modality t1
python scripts/generate_csv_splits.py --data_dir $DATA_DIR --dataset brats --modality t1
python scripts/generate_csv_splits.py --data_dir $DATA_DIR --dataset mood --modality t2

# 3. Verify data
python scripts/verify_dataset.py --data_dir $DATA_DIR

# 4. Pretrain encoder
python run.py experiment=cDDPM/Spark_2D_pretrain

# 5. Train full model
python run.py experiment=cDDPM/DDPM_cond_spark_2D encoder_path=<path_from_step4>

# 6. Evaluate on all test sets
# (Evaluation happens automatically during training)
```

The model will evaluate on BraTS, MSLUB (if available), MOOD, and IXI test sets automatically.

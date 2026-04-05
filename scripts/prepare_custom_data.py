#!/usr/bin/env python
"""
Data Preparation Script for cDDPM Training

This script prepares the MOOD_IXI and BraTS datasets for training with the
Conditioned Diffusion Models for Unsupervised Anomaly Detection (cDDPM) repo.

Steps performed:
1. Creates directory structure matching repo expectations
2. Creates symlinks for IXI T2 images
3. Converts tissue segmentations to binary brain masks
4. Prepares BraTS T2 images with masks and tumor segmentations
5. Generates train/val/test CSV split files

Usage:
    python scripts/prepare_custom_data.py
"""

import os
import sys
import glob
import shutil
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from tqdm import tqdm

# Configuration - set these via environment variables or modify directly
SOURCE_MOOD_IXI = os.environ.get("SOURCE_MOOD_IXI", "/path/to/MOOD_IXI_all")
SOURCE_BRATS = os.environ.get("SOURCE_BRATS", "/path/to/BraTS/BraTS_raw")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/path/to/output/cddpm_data")

# Number of cross-validation folds
NUM_FOLDS = 5
# Random seed for reproducibility
RANDOM_SEED = 42


def create_directory_structure():
    """Create the expected directory structure."""
    dirs = [
        f"{OUTPUT_DIR}/Data/Train/ixi/t2",
        f"{OUTPUT_DIR}/Data/Train/ixi/mask",
        f"{OUTPUT_DIR}/Data/Test/Brats20/t2",
        f"{OUTPUT_DIR}/Data/Test/Brats20/mask",
        f"{OUTPUT_DIR}/Data/Test/Brats20/seg",
        f"{OUTPUT_DIR}/Data/splits",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"✓ Created directory structure at {OUTPUT_DIR}")


def convert_seg_to_mask(seg_path, mask_path):
    """Convert tissue segmentation (0-3 values) to binary brain mask."""
    seg_img = nib.load(seg_path)
    seg_data = seg_img.get_fdata()
    
    # Combine all non-zero labels (CSF=1, GM=2, WM=3) into binary mask
    mask_data = (seg_data > 0).astype(np.uint8)
    
    mask_img = nib.Nifti1Image(mask_data, seg_img.affine, seg_img.header)
    nib.save(mask_img, mask_path)


def create_brain_mask_from_image(img_path, mask_path):
    """Create binary brain mask from non-zero voxels in image."""
    img = nib.load(img_path)
    data = img.get_fdata()
    
    # Create mask from non-zero voxels
    mask_data = (data > 0).astype(np.uint8)
    
    mask_img = nib.Nifti1Image(mask_data, img.affine, img.header)
    nib.save(mask_img, mask_path)


def prepare_ixi_t2():
    """Prepare IXI T2 images and masks."""
    print("\n=== Preparing IXI T2 Data ===")
    
    # Find all T2 files (IXI naming: IXI***-***-****-T2_brain.nii.gz)
    t2_files = sorted(glob.glob(f"{SOURCE_MOOD_IXI}/input_noseg/*-T2_brain.nii.gz"))
    print(f"Found {len(t2_files)} IXI T2 files")
    
    processed = []
    
    for t2_path in tqdm(t2_files, desc="Processing IXI T2"):
        filename = os.path.basename(t2_path)
        # Expected: IXI002-Guys-0828-T2_brain.nii.gz
        
        # Create output paths
        # Rename to match expected format: IXI002-Guys-0828_t2.nii.gz
        base_name = filename.replace("-T2_brain.nii.gz", "")
        out_img_name = f"{base_name}_t2.nii.gz"
        out_mask_name = f"{base_name}_mask.nii.gz"
        
        out_img_path = f"{OUTPUT_DIR}/Data/Train/ixi/t2/{out_img_name}"
        out_mask_path = f"{OUTPUT_DIR}/Data/Train/ixi/mask/{out_mask_name}"
        
        # Create symlink for image
        if not os.path.exists(out_img_path):
            os.symlink(t2_path, out_img_path)
        
        # Find corresponding segmentation file and convert to mask
        seg_filename = filename.replace(".nii.gz", "_seg.nii.gz")
        seg_path = f"{SOURCE_MOOD_IXI}/input_seg/{seg_filename}"
        
        if os.path.exists(seg_path):
            if not os.path.exists(out_mask_path):
                convert_seg_to_mask(seg_path, out_mask_path)
        else:
            # Create mask from image if no seg file
            print(f"  Warning: No seg file for {filename}, creating mask from image")
            if not os.path.exists(out_mask_path):
                create_brain_mask_from_image(t2_path, out_mask_path)
        
        processed.append({
            'img_name': out_img_name,
            'img_path': f"/Train/ixi/t2/{out_img_name}",
            'mask_path': f"/Train/ixi/mask/{out_mask_name}",
            'age': 0,  # Age not available in filenames
            'label': 0,  # Healthy
        })
    
    print(f"✓ Processed {len(processed)} IXI T2 images")
    return pd.DataFrame(processed)


def prepare_brats():
    """Prepare BraTS T2 images, masks, and segmentations."""
    print("\n=== Preparing BraTS Data ===")
    
    # Find all subject folders
    subject_dirs = sorted(glob.glob(f"{SOURCE_BRATS}/BraTS20_Training_*"))
    print(f"Found {len(subject_dirs)} BraTS subjects")
    
    processed = []
    
    for subj_dir in tqdm(subject_dirs, desc="Processing BraTS"):
        subj_name = os.path.basename(subj_dir)
        
        # Find T2 file
        t2_file = glob.glob(f"{subj_dir}/*_t2.nii.gz")
        seg_file = glob.glob(f"{subj_dir}/*_seg.nii.gz")
        
        if not t2_file:
            print(f"  Warning: No T2 file in {subj_name}")
            continue
        
        t2_path = t2_file[0]
        t2_filename = os.path.basename(t2_path)
        
        # Output paths
        out_img_name = f"{subj_name}_t2.nii.gz"
        out_mask_name = f"{subj_name}_mask.nii.gz"
        out_seg_name = f"{subj_name}_seg.nii.gz"
        
        out_img_path = f"{OUTPUT_DIR}/Data/Test/Brats20/t2/{out_img_name}"
        out_mask_path = f"{OUTPUT_DIR}/Data/Test/Brats20/mask/{out_mask_name}"
        out_seg_path = f"{OUTPUT_DIR}/Data/Test/Brats20/seg/{out_seg_name}"
        
        # Create symlink for image
        if not os.path.exists(out_img_path):
            os.symlink(t2_path, out_img_path)
        
        # Create brain mask from image (BraTS is already skull-stripped)
        if not os.path.exists(out_mask_path):
            create_brain_mask_from_image(t2_path, out_mask_path)
        
        # Symlink segmentation if exists
        if seg_file and not os.path.exists(out_seg_path):
            os.symlink(seg_file[0], out_seg_path)
        
        processed.append({
            'img_name': out_img_name,
            'img_path': f"/Test/Brats20/t2/{out_img_name}",
            'mask_path': f"/Test/Brats20/mask/{out_mask_name}",
            'seg_path': f"/Test/Brats20/seg/{out_seg_name}" if seg_file else None,
            'age': 0,  # Age not available
            'label': 1,  # Pathological
        })
    
    print(f"✓ Processed {len(processed)} BraTS subjects")
    return pd.DataFrame(processed)


def create_csv_splits(ixi_df, brats_df):
    """Create train/val/test CSV split files."""
    print("\n=== Creating CSV Splits ===")
    
    np.random.seed(RANDOM_SEED)
    
    # Shuffle IXI data
    ixi_df = ixi_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    # K-Fold cross-validation splits
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(ixi_df)):
        train_df = ixi_df.iloc[train_idx].copy()
        val_df = ixi_df.iloc[val_idx].copy()
        
        # Add required columns
        for df in [train_df, val_df]:
            df['seg_path'] = None
        
        train_df.to_csv(f"{OUTPUT_DIR}/Data/splits/IXI_train_fold{fold}.csv", index=True)
        val_df.to_csv(f"{OUTPUT_DIR}/Data/splits/IXI_val_fold{fold}.csv", index=True)
        
        print(f"  Fold {fold}: train={len(train_df)}, val={len(val_df)}")
    
    # Create test split (use fold 0 validation as test for IXI)
    test_df = ixi_df.iloc[list(kf.split(ixi_df))[0][1]].copy()
    test_df['seg_path'] = None
    test_df.to_csv(f"{OUTPUT_DIR}/Data/splits/IXI_test.csv", index=True)
    print(f"  IXI test: {len(test_df)}")
    
    # Create avail_t2.csv - list of all T2 filenames for the keep_t2 filter
    # The code expects t1 filenames and replaces t2 with t1 for matching
    avail_t2 = pd.DataFrame({'0': ixi_df['img_name'].str.replace('_t2', '_t1')})
    avail_t2.to_csv(f"{OUTPUT_DIR}/Data/splits/avail_t2.csv", index=False)
    print(f"  avail_t2.csv: {len(avail_t2)} entries")
    
    # BraTS test split (all BraTS data is for testing)
    brats_df.to_csv(f"{OUTPUT_DIR}/Data/splits/Brats20_test.csv", index=True)
    print(f"  BraTS test: {len(brats_df)}")
    
    # Also create a validation split for BraTS (use 10% for validation)
    brats_val = brats_df.sample(frac=0.1, random_state=RANDOM_SEED)
    brats_val.to_csv(f"{OUTPUT_DIR}/Data/splits/Brats20_val.csv", index=True)
    print(f"  BraTS val: {len(brats_val)}")
    
    print("✓ Created all CSV split files")


def main():
    print("=" * 60)
    print("cDDPM Data Preparation Script")
    print("=" * 60)
    
    # Step 1: Create directory structure
    create_directory_structure()
    
    # Step 2: Prepare IXI T2 data
    ixi_df = prepare_ixi_t2()
    
    # Step 3: Prepare BraTS data
    brats_df = prepare_brats()
    
    # Step 4: Create CSV splits
    create_csv_splits(ixi_df, brats_df)
    
    print("\n" + "=" * 60)
    print("✓ Data preparation complete!")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("\nNext steps:")
    print(f"  1. Update pc_environment.env with DATA_DIR={OUTPUT_DIR}")
    print("  2. Run training: python run.py experiment=cDDPM/DDPM_cond_spark_2D")
    print("=" * 60)


if __name__ == "__main__":
    main()

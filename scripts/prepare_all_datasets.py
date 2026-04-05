#!/usr/bin/env python
"""
Data Preparation Script for All cDDPM Training Datasets

This script prepares all dataset variants for cDDPM training:
- T1_only: MOOD_3T, MOOD_IXI_3T, MOOD_IXI_3T_15T  
- T2_only: IXI_3T, IXI_15T
- T1T2_combined: MOOD_IXI_3T_T1T2

And BraTS evaluation sets:
- BraTS_T1 (for T1 model evaluation)
- BraTS_T2 (for T2 model evaluation)

Usage:
    python scripts/prepare_all_datasets.py
"""

import os
import sys
import glob
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm

# Configuration - set these via environment variables or modify directly
BASE_DATASETS = os.environ.get("BASE_DATASETS", "/path/to/datasets")
OUTPUT_BASE = os.environ.get("OUTPUT_BASE", "/path/to/cddpm_prepared")

# Dataset definitions
DATASETS = {
    # T1 only datasets
    "MOOD_3T_T1": {
        "source": f"{BASE_DATASETS}/T1_only/MOOD_3T",
        "modality": "t1",
        "raw_folder": "raw",
        "seg_folder": "seg",
    },
    "MOOD_IXI_3T_T1": {
        "source": f"{BASE_DATASETS}/T1_only/MOOD_IXI_3T",
        "modality": "t1",
        "raw_folder": "raw",
        "seg_folder": "seg",
    },
    "MOOD_IXI_3T_15T_T1": {
        "source": f"{BASE_DATASETS}/T1_only/MOOD_IXI_3T_15T",
        "modality": "t1",
        "raw_folder": "raw",
        "seg_folder": "seg",
    },
    # T2 only datasets
    "IXI_3T_T2": {
        "source": f"{BASE_DATASETS}/T2_only/IXI_3T",
        "modality": "t2",
        "raw_folder": "raw",
        "seg_folder": "seg",
    },
    "IXI_15T_T2": {
        "source": f"{BASE_DATASETS}/T2_only/IXI_15T",
        "modality": "t2",
        "raw_folder": "raw",
        "seg_folder": "seg",
    },
    # Combined T1T2 dataset (trained as single modality but data is mixed)
    "MOOD_IXI_3T_T1T2": {
        "source": f"{BASE_DATASETS}/T1T2_combined/MOOD_IXI_3T_T1T2",
        "modality": "mixed",  # Contains both T1 and T2
        "raw_folder": "raw",
        "seg_folder": "seg",
    },
}

# BraTS evaluation sets
BRATS_CONFIG = {
    "BraTS_T1": {
        "source": f"{BASE_DATASETS}/BraTS/BraTS_T1_seg",
        "modality": "t1",
    },
    "BraTS_T2": {
        "source": f"{BASE_DATASETS}/BraTS/BraTS_T2_seg",
        "modality": "t2",
    },
}

NUM_FOLDS = 5
RANDOM_SEED = 42
VAL_RATIO = 0.1  # 10% for validation
TEST_RATIO = 0.1  # 10% for test


def create_directory_structure(dataset_name):
    """Create the expected directory structure for a dataset."""
    output_dir = f"{OUTPUT_BASE}/{dataset_name}"
    dirs = [
        f"{output_dir}/Data/Train/{dataset_name.lower()}/raw",
        f"{output_dir}/Data/Train/{dataset_name.lower()}/mask",
        f"{output_dir}/Data/Test/BraTS_T1/t1",
        f"{output_dir}/Data/Test/BraTS_T1/mask",
        f"{output_dir}/Data/Test/BraTS_T1/seg",
        f"{output_dir}/Data/Test/BraTS_T2/t2",
        f"{output_dir}/Data/Test/BraTS_T2/mask",
        f"{output_dir}/Data/Test/BraTS_T2/seg",
        f"{output_dir}/Data/splits",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"✓ Created directory structure at {output_dir}")
    return output_dir


def create_brain_mask_from_image(img_path, mask_path):
    """Create binary brain mask from non-zero voxels in image."""
    img = nib.load(img_path)
    data = img.get_fdata()
    mask_data = (data > 0).astype(np.uint8)
    mask_img = nib.Nifti1Image(mask_data, img.affine, img.header)
    nib.save(mask_img, mask_path)


def prepare_training_data(dataset_name, config, output_dir):
    """Prepare training data with symlinks and masks."""
    source_dir = config["source"]
    modality = config["modality"]
    raw_folder = config["raw_folder"]
    
    train_dir = f"{output_dir}/Data/Train/{dataset_name.lower()}"
    raw_dir = f"{train_dir}/raw"
    mask_dir = f"{train_dir}/mask"
    
    # Find all raw images
    raw_files = sorted(glob.glob(f"{source_dir}/{raw_folder}/*.nii.gz"))
    print(f"Found {len(raw_files)} images in {dataset_name}")
    
    subjects = []
    for raw_file in tqdm(raw_files, desc=f"Processing {dataset_name}"):
        filename = os.path.basename(raw_file)
        subject_id = filename.replace('.nii.gz', '')
        
        # Create symlink for raw image
        raw_link = f"{raw_dir}/{filename}"
        if not os.path.exists(raw_link):
            os.symlink(raw_file, raw_link)
        
        # Create or link mask
        mask_file = f"{mask_dir}/{subject_id}_mask.nii.gz"
        seg_file = f"{source_dir}/seg/{subject_id}_seg.nii.gz" if config.get("seg_folder") else None
        
        if not os.path.exists(mask_file):
            if seg_file and os.path.exists(seg_file):
                # Convert segmentation to binary mask
                seg_img = nib.load(seg_file)
                seg_data = seg_img.get_fdata()
                mask_data = (seg_data > 0).astype(np.uint8)
                mask_img = nib.Nifti1Image(mask_data, seg_img.affine, seg_img.header)
                nib.save(mask_img, mask_file)
            else:
                # Create mask from non-zero voxels
                create_brain_mask_from_image(raw_file, mask_file)
        
        subjects.append({
            'img_name': subject_id,
            'img_path': f"Train/{dataset_name.lower()}/raw/{filename}",
            'mask_path': f"Train/{dataset_name.lower()}/mask/{subject_id}_mask.nii.gz",
            'seg_path': '',
            'age': 0,
            'label': 0,  # healthy
        })
    
    return pd.DataFrame(subjects)


def prepare_brats_eval(output_dir, brats_type="T1"):
    """Prepare BraTS evaluation data."""
    brats_config = BRATS_CONFIG[f"BraTS_{brats_type}"]
    source_dir = brats_config["source"]
    modality = brats_config["modality"]
    
    test_dir = f"{output_dir}/Data/Test/BraTS_{brats_type}"
    img_dir = f"{test_dir}/{modality}"
    mask_dir = f"{test_dir}/mask"
    seg_dir = f"{test_dir}/seg"
    
    # Find all seg files and derive image names
    seg_files = sorted(glob.glob(f"{source_dir}/*_seg.nii.gz"))
    print(f"Found {len(seg_files)} BraTS {brats_type} subjects")
    
    subjects = []
    for seg_file in tqdm(seg_files, desc=f"Processing BraTS_{brats_type}"):
        filename = os.path.basename(seg_file)
        # e.g., BraTS20_Training_00100_t1_seg.nii.gz -> BraTS20_Training_00100
        subject_id = filename.replace(f'_{modality}_seg.nii.gz', '')
        
        # Find corresponding image file
        img_filename = filename.replace('_seg.nii.gz', '.nii.gz')
        img_source = f"{source_dir}/{img_filename}"
        
        if not os.path.exists(img_source):
            # Try without modality in filename
            img_source = seg_file.replace('_seg.nii.gz', '.nii.gz')
        
        # Create symlinks
        img_link = f"{img_dir}/{img_filename}"
        seg_link = f"{seg_dir}/{filename}"
        mask_file = f"{mask_dir}/{subject_id}_mask.nii.gz"
        
        if os.path.exists(img_source) and not os.path.exists(img_link):
            os.symlink(img_source, img_link)
        
        if not os.path.exists(seg_link):
            os.symlink(seg_file, seg_link)
        
        # Create mask from image
        if os.path.exists(img_source) and not os.path.exists(mask_file):
            create_brain_mask_from_image(img_source, mask_file)
        
        subjects.append({
            'img_name': subject_id,
            'img_path': f"Test/BraTS_{brats_type}/{modality}/{img_filename}",
            'mask_path': f"Test/BraTS_{brats_type}/mask/{subject_id}_mask.nii.gz",
            'seg_path': f"Test/BraTS_{brats_type}/seg/{filename}",
            'age': 0,
            'label': 1,  # pathological
        })
    
    return pd.DataFrame(subjects)


def create_cv_splits(df, output_dir, dataset_name, num_folds=5):
    """Create cross-validation splits."""
    splits_dir = f"{output_dir}/Data/splits"
    
    # Shuffle data
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    # Split into train+val and test
    train_val_df, test_df = train_test_split(df, test_size=TEST_RATIO, random_state=RANDOM_SEED)
    
    # Save test set
    test_df.to_csv(f"{splits_dir}/{dataset_name}_test.csv", index=False)
    print(f"  Test set: {len(test_df)} subjects")
    
    # Create k-fold splits
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=RANDOM_SEED)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_df)):
        train_fold = train_val_df.iloc[train_idx]
        val_fold = train_val_df.iloc[val_idx]
        
        train_fold.to_csv(f"{splits_dir}/{dataset_name}_train_fold{fold}.csv", index=False)
        val_fold.to_csv(f"{splits_dir}/{dataset_name}_val_fold{fold}.csv", index=False)
        print(f"  Fold {fold}: train={len(train_fold)}, val={len(val_fold)}")
    
    # Also save full available list for T2 compatibility
    df.to_csv(f"{splits_dir}/avail_{dataset_name.lower()}.csv", index=False)


def create_brats_splits(df, output_dir, brats_type):
    """Create BraTS val/test splits."""
    splits_dir = f"{output_dir}/Data/splits"
    
    # Split into val and test
    val_df, test_df = train_test_split(df, test_size=0.9, random_state=RANDOM_SEED)  # 10% val, 90% test
    
    val_df.to_csv(f"{splits_dir}/BraTS_{brats_type}_val.csv", index=False)
    test_df.to_csv(f"{splits_dir}/BraTS_{brats_type}_test.csv", index=False)
    print(f"  BraTS_{brats_type}: val={len(val_df)}, test={len(test_df)}")


def prepare_dataset(dataset_name, config):
    """Prepare a complete dataset for training."""
    print(f"\n{'='*60}")
    print(f"Preparing {dataset_name}")
    print(f"{'='*60}")
    
    output_dir = create_directory_structure(dataset_name)
    
    # Prepare training data
    train_df = prepare_training_data(dataset_name, config, output_dir)
    
    # Create CV splits
    create_cv_splits(train_df, output_dir, dataset_name)
    
    # Prepare BraTS evaluation data based on modality
    modality = config["modality"]
    
    if modality == "t1":
        brats_df = prepare_brats_eval(output_dir, "T1")
        create_brats_splits(brats_df, output_dir, "T1")
    elif modality == "t2":
        brats_df = prepare_brats_eval(output_dir, "T2")
        create_brats_splits(brats_df, output_dir, "T2")
    else:  # mixed - prepare both
        brats_t1_df = prepare_brats_eval(output_dir, "T1")
        brats_t2_df = prepare_brats_eval(output_dir, "T2")
        create_brats_splits(brats_t1_df, output_dir, "T1")
        create_brats_splits(brats_t2_df, output_dir, "T2")
    
    print(f"✓ {dataset_name} preparation complete!")
    return output_dir


def main():
    """Main function to prepare all datasets."""
    print("="*60)
    print("cDDPM Multi-Dataset Preparation")
    print("="*60)
    
    # Create base output directory
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    prepared = {}
    for dataset_name, config in DATASETS.items():
        try:
            output_dir = prepare_dataset(dataset_name, config)
            prepared[dataset_name] = output_dir
        except Exception as e:
            print(f"ERROR preparing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, path in prepared.items():
        print(f"  {name}: {path}")
    
    print("\nAll datasets prepared successfully!")


if __name__ == "__main__":
    main()

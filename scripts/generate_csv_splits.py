#!/usr/bin/env python3
"""
Generate CSV split files for custom datasets.

This script automatically scans your data directory and creates CSV files
with the required format for training and evaluation.

Usage:
    python generate_csv_splits.py --data_dir /path/to/DATA_DIR --dataset ixi --modality t1 --split train
    python generate_csv_splits.py --data_dir /path/to/DATA_DIR --dataset brats --split test
    python generate_csv_splits.py --data_dir /path/to/DATA_DIR --dataset mood --split test
"""

import os
import argparse
import pandas as pd
from pathlib import Path
import glob
import re


def extract_age_from_filename(filename):
    """Try to extract age from filename if present."""
    # Look for patterns like _age45_ or _45y or similar
    age_patterns = [
        r'_age(\d+)',
        r'_(\d+)y',
        r'age(\d+)',
    ]
    for pattern in age_patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None  # Return None if age not found


def generate_ixi_csv(data_dir, modality='t1', split='train', fold=0, output_name=None):
    """
    Generate CSV file for IXI dataset.
    
    Args:
        data_dir: Base data directory
        modality: 't1' or 't2'
        split: 'train', 'val', or 'test'
        fold: Fold number for cross-validation (0-4)
        output_name: Custom output filename (optional)
    """
    img_dir = os.path.join(data_dir, 'Data', 'Train', 'ixi', modality)
    mask_dir = os.path.join(data_dir, 'Data', 'Train', 'ixi', 'mask')
    
    if not os.path.exists(img_dir):
        print(f"Error: Image directory not found: {img_dir}")
        return None
    
    # Find all images
    images = sorted(glob.glob(os.path.join(img_dir, '*.nii.gz')))
    
    if len(images) == 0:
        print(f"Warning: No images found in {img_dir}")
        return None
    
    data = []
    for img_path in images:
        img_name = os.path.basename(img_path)
        base_name = img_name.replace(f'_{modality}.nii.gz', '').replace('.nii.gz', '')
        
        # Construct paths
        mask_name = f"{base_name}_mask.nii.gz"
        mask_path = os.path.join(mask_dir, mask_name)
        
        # Check if mask exists
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {img_name}, skipping...")
            continue
        
        # Extract age from filename if possible
        age = extract_age_from_filename(img_name)
        if age is None:
            age = 50.0  # Default age if not found
        
        # Determine age group (optional, for consistency with original IXI format)
        if age < 35:
            agegroup = 1.0
        elif age < 50:
            agegroup = 2.0
        else:
            agegroup = 3.0
        
        # Create relative paths
        rel_img_path = f"/Train/ixi/{modality}/{img_name}"
        rel_mask_path = f"/Train/ixi/mask/{mask_name}"
        
        row = {
            'img_name': img_name,
            'SEX_ID (1=m, 2=f)': 1,  # Default, can be updated manually
            'HEIGHT': 0,
            'WEIGHT': 0,
            'ETHNIC_ID': 1,
            'MARITAL_ID': 2,
            'OCCUPATION_ID': 1,
            'QUALIFICATION_ID': 4,
            'DATE_AVAILABLE': 1,
            'STUDY_DATE': '2023-01-01',
            'age': age,
            'label': 0,  # Healthy subjects
            'img_path': rel_img_path,
            'mask_path': rel_mask_path,
            'seg_path': '',
            'Agegroup': agegroup
        }
        data.append(row)
    
    if len(data) == 0:
        print("Error: No valid image-mask pairs found")
        return None
    
    df = pd.DataFrame(data)
    
    # Split data into train/val/test
    n_total = len(df)
    n_test = max(int(n_total * 0.15), 1)
    n_val = max(int(n_total * 0.15), 1)
    n_train = n_total - n_test - n_val
    
    # For cross-validation, create 5 folds
    n_fold = n_total // 5
    
    if split == 'test':
        output_df = df.iloc[:n_test].copy()
        default_name = 'IXI_test.csv'
    elif split == 'val':
        start_idx = n_fold * fold
        end_idx = n_fold * (fold + 1)
        output_df = df.iloc[start_idx:end_idx].copy()
        default_name = f'IXI_val_fold{fold}.csv'
    else:  # train
        # Exclude the validation fold
        val_start = n_fold * fold
        val_end = n_fold * (fold + 1)
        output_df = pd.concat([df.iloc[:val_start], df.iloc[val_end:]]).copy()
        default_name = f'IXI_train_fold{fold}.csv'
    
    output_file = os.path.join(data_dir, 'Data', 'splits', output_name if output_name else default_name)
    
    output_df.to_csv(output_file, index=True)
    print(f"Created: {output_file} with {len(output_df)} entries")
    return output_file


def generate_test_dataset_csv(data_dir, dataset='brats', modality='t1', split='test', output_name=None):
    """
    Generate CSV file for test datasets (BraTS, MOOD, etc.).
    
    Args:
        data_dir: Base data directory
        dataset: 'brats' or 'mood'
        modality: 't1' or 't2'
        split: 'test' or 'val'
        output_name: Custom output filename (optional)
    """
    dataset_name = dataset.capitalize()
    if dataset.lower() == 'brats':
        dataset_name = 'Brats21'
    elif dataset.lower() == 'mood':
        dataset_name = 'MOOD'
    
    img_dir = os.path.join(data_dir, 'Data', 'Test', dataset_name, modality)
    mask_dir = os.path.join(data_dir, 'Data', 'Test', dataset_name, 'mask')
    seg_dir = os.path.join(data_dir, 'Data', 'Test', dataset_name, 'seg')
    
    if not os.path.exists(img_dir):
        print(f"Error: Image directory not found: {img_dir}")
        return None
    
    # Find all images
    images = sorted(glob.glob(os.path.join(img_dir, '*.nii.gz')))
    
    if len(images) == 0:
        print(f"Warning: No images found in {img_dir}")
        return None
    
    data = []
    for img_path in images:
        img_name = os.path.basename(img_path)
        base_name = img_name.replace(f'_{modality}.nii.gz', '').replace('.nii.gz', '')
        
        # Construct paths
        mask_name = f"{base_name}_mask.nii.gz"
        seg_name = f"{base_name}_seg.nii.gz"
        
        mask_path = os.path.join(mask_dir, mask_name)
        seg_path = os.path.join(seg_dir, seg_name)
        
        # Check if files exist
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {img_name}, skipping...")
            continue
        
        # Segmentation is optional
        has_seg = os.path.exists(seg_path)
        
        # Extract age from filename if possible
        age = extract_age_from_filename(img_name)
        
        # Create relative paths
        rel_img_path = f"/Test/{dataset_name}/{modality}/{img_name}"
        rel_mask_path = f"/Test/{dataset_name}/mask/{mask_name}"
        rel_seg_path = f"/Test/{dataset_name}/seg/{seg_name}" if has_seg else ''
        
        row = {
            'img_name': img_name,
            'age': age if age is not None else '',
            'label': 1,  # Anomaly present
            'img_path': rel_img_path,
            'mask_path': rel_mask_path,
            'seg_path': rel_seg_path
        }
        
        # Add extra columns for BraTS format compatibility
        if dataset.lower() == 'brats':
            row['Cohort Name (if publicly available)'] = 'Custom'
            row['Site No (represents the originating institution)'] = ''
            row['Local ID'] = base_name
        
        data.append(row)
    
    if len(data) == 0:
        print("Error: No valid image-mask pairs found")
        return None
    
    df = pd.DataFrame(data)
    
    # For test datasets, use all data for test or split for val/test
    if split == 'val':
        n_val = max(int(len(df) * 0.3), 1)
        output_df = df.iloc[:n_val].copy()
    else:  # test
        output_df = df.copy()
    
    default_name = f'{dataset_name}_{split}.csv'
    output_file = os.path.join(data_dir, 'Data', 'splits', output_name if output_name else default_name)
    output_df.to_csv(output_file, index=True)
    print(f"Created: {output_file} with {len(output_df)} entries")
    return output_file


def verify_csv_file(csv_path, data_dir):
    """
    Verify that all files referenced in CSV exist.
    
    Args:
        csv_path: Path to CSV file
        data_dir: Base data directory
    """
    print(f"\nVerifying CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    
    missing_files = []
    for idx, row in df.iterrows():
        # Check image
        img_full_path = os.path.join(data_dir, 'Data', row['img_path'].lstrip('/'))
        if not os.path.exists(img_full_path):
            missing_files.append(('image', row['img_name'], img_full_path))
        
        # Check mask
        mask_full_path = os.path.join(data_dir, 'Data', row['mask_path'].lstrip('/'))
        if not os.path.exists(mask_full_path):
            missing_files.append(('mask', row['img_name'], mask_full_path))
        
        # Check segmentation (if specified)
        if pd.notna(row.get('seg_path', '')) and row['seg_path'] != '':
            seg_full_path = os.path.join(data_dir, 'Data', row['seg_path'].lstrip('/'))
            if not os.path.exists(seg_full_path):
                missing_files.append(('segmentation', row['img_name'], seg_full_path))
    
    if missing_files:
        print(f"WARNING: Found {len(missing_files)} missing files:")
        for file_type, img_name, path in missing_files[:10]:  # Show first 10
            print(f"  - {file_type} for {img_name}: {path}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
        return False
    else:
        print(f"✓ All {len(df)} entries verified successfully!")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Generate CSV split files for datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate IXI training split for fold 0
  python generate_csv_splits.py --data_dir /path/to/DATA --dataset ixi --modality t1 --split train --fold 0
  
  # Generate IXI validation splits for all folds
  for fold in {0..4}; do
    python generate_csv_splits.py --data_dir /path/to/DATA --dataset ixi --modality t1 --split val --fold $fold
  done
  
  # Generate BraTS test split
  python generate_csv_splits.py --data_dir /path/to/DATA --dataset brats --modality t2 --split test
  
  # Generate MOOD test split
  python generate_csv_splits.py --data_dir /path/to/DATA --dataset mood --modality t2 --split test
  
  # Verify existing CSV
  python generate_csv_splits.py --data_dir /path/to/DATA --verify Data/splits/IXI_train_fold0.csv
        """
    )
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the base data directory (DATA_DIR)')
    parser.add_argument('--dataset', type=str, choices=['ixi', 'brats', 'mood'],
                        help='Dataset name')
    parser.add_argument('--modality', type=str, choices=['t1', 't2'], default='t1',
                        help='MRI modality')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], default='test',
                        help='Data split to generate')
    parser.add_argument('--fold', type=int, default=0, choices=range(5),
                        help='Fold number for cross-validation (IXI only)')
    parser.add_argument('--output_name', type=str,
                        help='Custom output filename (e.g., IXI_T1_train_fold0.csv)')
    parser.add_argument('--verify', type=str,
                        help='Path to CSV file to verify (relative to data_dir)')
    
    args = parser.parse_args()
    
    # Verify mode
    if args.verify:
        csv_path = os.path.join(args.data_dir, args.verify)
        if not os.path.exists(csv_path):
            print(f"Error: CSV file not found: {csv_path}")
            return
        verify_csv_file(csv_path, args.data_dir)
        return
    
    # Generate mode
    if not args.dataset:
        parser.error("--dataset is required when not using --verify")
    
    # Create splits directory if it doesn't exist
    splits_dir = os.path.join(args.data_dir, 'Data', 'splits')
    os.makedirs(splits_dir, exist_ok=True)
    
    # Generate appropriate CSV
    if args.dataset == 'ixi':
        csv_file = generate_ixi_csv(args.data_dir, args.modality, args.split, args.fold, args.output_name)
    else:
        csv_file = generate_test_dataset_csv(args.data_dir, args.dataset, args.modality, args.split, args.output_name)
    
    # Verify the generated file
    if csv_file and os.path.exists(csv_file):
        verify_csv_file(csv_file, args.data_dir)


if __name__ == '__main__':
    main()

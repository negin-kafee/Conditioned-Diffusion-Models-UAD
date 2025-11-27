#!/usr/bin/env python3
"""
Verify dataset setup before training.

This script checks:
- Directory structure
- CSV files exist and are properly formatted
- All referenced files exist
- Image and mask dimensions match

Usage:
    python verify_dataset.py --data_dir /path/to/DATA_DIR
"""

import os
import argparse
import pandas as pd
from pathlib import Path
import glob


def check_directory_structure(data_dir):
    """Check if the required directory structure exists."""
    print("\n=== Checking Directory Structure ===")
    
    required_dirs = [
        'Data/Train/ixi/t1',
        'Data/Train/ixi/t2',
        'Data/Train/ixi/mask',
        'Data/Test',
        'Data/splits',
    ]
    
    all_good = True
    for dir_path in required_dirs:
        full_path = os.path.join(data_dir, dir_path)
        exists = os.path.exists(full_path)
        status = "✓" if exists else "✗"
        print(f"  {status} {dir_path}")
        if not exists:
            all_good = False
    
    return all_good


def check_csv_files(data_dir):
    """Check if CSV split files exist."""
    print("\n=== Checking CSV Files ===")
    
    splits_dir = os.path.join(data_dir, 'Data', 'splits')
    if not os.path.exists(splits_dir):
        print(f"  ✗ Splits directory not found: {splits_dir}")
        return False
    
    csv_files = glob.glob(os.path.join(splits_dir, '*.csv'))
    if len(csv_files) == 0:
        print(f"  ✗ No CSV files found in {splits_dir}")
        return False
    
    print(f"  Found {len(csv_files)} CSV files:")
    for csv_file in sorted(csv_files):
        basename = os.path.basename(csv_file)
        df = pd.read_csv(csv_file)
        print(f"    ✓ {basename} ({len(df)} entries)")
    
    return True


def verify_csv_file(csv_path, data_dir, check_dimensions=False):
    """
    Verify that all files referenced in CSV exist and optionally check dimensions.
    
    Returns:
        tuple: (success, num_entries, num_errors)
    """
    if not os.path.exists(csv_path):
        return False, 0, 0
    
    df = pd.read_csv(csv_path)
    
    missing_images = []
    missing_masks = []
    missing_segs = []
    dimension_mismatches = []
    
    for idx, row in df.iterrows():
        # Check image
        img_full_path = os.path.join(data_dir, 'Data', row['img_path'].lstrip('/'))
        if not os.path.exists(img_full_path):
            missing_images.append((row['img_name'], img_full_path))
        
        # Check mask
        mask_full_path = os.path.join(data_dir, 'Data', row['mask_path'].lstrip('/'))
        if not os.path.exists(mask_full_path):
            missing_masks.append((row['img_name'], mask_full_path))
        
        # Check segmentation (if specified)
        if pd.notna(row.get('seg_path', '')) and row['seg_path'] != '':
            seg_full_path = os.path.join(data_dir, 'Data', row['seg_path'].lstrip('/'))
            if not os.path.exists(seg_full_path):
                missing_segs.append((row['img_name'], seg_full_path))
        
        # Check dimensions if requested
        if check_dimensions and os.path.exists(img_full_path) and os.path.exists(mask_full_path):
            try:
                import nibabel as nib
                img = nib.load(img_full_path)
                mask = nib.load(mask_full_path)
                if img.shape != mask.shape:
                    dimension_mismatches.append((row['img_name'], img.shape, mask.shape))
            except Exception as e:
                print(f"    Warning: Could not check dimensions for {row['img_name']}: {e}")
    
    errors = []
    if missing_images:
        errors.extend(missing_images)
    if missing_masks:
        errors.extend(missing_masks)
    if missing_segs:
        errors.extend(missing_segs)
    if dimension_mismatches:
        errors.extend(dimension_mismatches)
    
    return len(errors) == 0, len(df), len(errors)


def check_all_csv_files(data_dir, check_dimensions=False):
    """Check all CSV files in the splits directory."""
    print("\n=== Verifying CSV File Contents ===")
    
    splits_dir = os.path.join(data_dir, 'Data', 'splits')
    csv_files = sorted(glob.glob(os.path.join(splits_dir, '*.csv')))
    
    if len(csv_files) == 0:
        print("  ✗ No CSV files found to verify")
        return False
    
    all_good = True
    total_entries = 0
    total_errors = 0
    
    for csv_file in csv_files:
        basename = os.path.basename(csv_file)
        success, num_entries, num_errors = verify_csv_file(csv_file, data_dir, check_dimensions)
        total_entries += num_entries
        total_errors += num_errors
        
        status = "✓" if success else "✗"
        error_msg = f" ({num_errors} errors)" if not success else ""
        print(f"  {status} {basename}: {num_entries} entries{error_msg}")
        
        if not success:
            all_good = False
    
    print(f"\nTotal: {total_entries} entries across {len(csv_files)} files")
    if total_errors > 0:
        print(f"⚠ Found {total_errors} issues that need to be fixed")
    else:
        print("✓ All entries verified successfully!")
    
    return all_good


def count_images(data_dir):
    """Count images in each directory."""
    print("\n=== Image Counts ===")
    
    dirs_to_check = [
        ('IXI T1 (train)', 'Data/Train/ixi/t1'),
        ('IXI T2 (train)', 'Data/Train/ixi/t2'),
        ('IXI Masks', 'Data/Train/ixi/mask'),
    ]
    
    # Check test datasets
    test_dir = os.path.join(data_dir, 'Data', 'Test')
    if os.path.exists(test_dir):
        for dataset in os.listdir(test_dir):
            dataset_path = os.path.join(test_dir, dataset)
            if os.path.isdir(dataset_path):
                for modality in ['t1', 't2']:
                    modality_path = os.path.join(dataset_path, modality)
                    if os.path.exists(modality_path):
                        dirs_to_check.append((f'{dataset} {modality.upper()}', f'Data/Test/{dataset}/{modality}'))
    
    for name, rel_path in dirs_to_check:
        full_path = os.path.join(data_dir, rel_path)
        if os.path.exists(full_path):
            nii_files = glob.glob(os.path.join(full_path, '*.nii*'))
            print(f"  {name}: {len(nii_files)} images")
        else:
            print(f"  {name}: directory not found")


def check_environment_file():
    """Check if pc_environment.env is properly configured."""
    print("\n=== Environment Configuration ===")
    
    env_file = 'pc_environment.env'
    if not os.path.exists(env_file):
        print(f"  ✗ {env_file} not found")
        return False
    
    with open(env_file, 'r') as f:
        content = f.read()
    
    has_data_dir = 'DATA_DIR=' in content and '<path_to_data>' not in content
    has_log_dir = 'LOG_DIR=' in content and '<path_to_logs>' not in content
    
    status_data = "✓" if has_data_dir else "✗"
    status_log = "✓" if has_log_dir else "✗"
    
    print(f"  {status_data} DATA_DIR configured")
    print(f"  {status_log} LOG_DIR configured")
    
    if not has_data_dir or not has_log_dir:
        print("\n  Please edit pc_environment.env and set DATA_DIR and LOG_DIR")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Verify dataset setup for training',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the base data directory (DATA_DIR)')
    parser.add_argument('--check_dimensions', action='store_true',
                        help='Also check if image and mask dimensions match (requires nibabel)')
    parser.add_argument('--detailed', action='store_true',
                        help='Show detailed error information')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Dataset Verification")
    print("=" * 60)
    print(f"\nData directory: {args.data_dir}")
    
    # Run all checks
    results = []
    
    results.append(("Directory Structure", check_directory_structure(args.data_dir)))
    results.append(("CSV Files Exist", check_csv_files(args.data_dir)))
    results.append(("CSV File Contents", check_all_csv_files(args.data_dir, args.check_dimensions)))
    
    # Count images (informational, always succeeds)
    count_images(args.data_dir)
    
    # Check environment configuration
    results.append(("Environment Config", check_environment_file()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = True
    for check_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status}: {check_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All checks passed! You're ready to start training.")
    else:
        print("⚠ Some checks failed. Please fix the issues above before training.")
        print("\nCommon solutions:")
        print("  - Run generate_csv_splits.py to create CSV files")
        print("  - Check that image files are in the correct directories")
        print("  - Ensure file naming conventions match (e.g., *_t1.nii.gz, *_mask.nii.gz)")
        print("  - Edit pc_environment.env with correct paths")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())

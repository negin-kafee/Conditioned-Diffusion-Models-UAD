#!/usr/bin/env python3
"""
Fix BraTS segmentation masks to match original preprocessing.

The original preprocessing binarizes the segmentation to keep only 
the tumor core (class 1 in original, or class 4 which is enhancing tumor).

BraTS 2020/2021 segmentation labels:
- 0: Background
- 1: Necrotic and Non-enhancing Tumor Core (NCR/NET)
- 2: Peritumoral Edema (ED)
- 4: GD-enhancing Tumor (ET) - this is class 3 in some files

The original codebase used binarized segmentation with only 
the tumor core visible (~1-2% of volume), not the full tumor
region (~15-20% of volume).

This script fixes the segmentations to use only class 1 (tumor core).
"""

import os
import sys
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

def binarize_segmentation(seg_data, keep_class=1):
    """
    Binarize segmentation to keep only the specified class.
    
    In BraTS:
    - Class 1 = NCR/NET (Necrotic and Non-enhancing Tumor Core)
    - Class 2 = ED (Peritumoral Edema)  
    - Class 4 = ET (Enhancing Tumor) - sometimes stored as 3
    
    The original preprocessing used class 1 only.
    """
    # Create binary mask where only the specified class is 1
    binary_seg = (seg_data == keep_class).astype(np.float32)
    return binary_seg


def fix_brats_seg_folder(seg_folder, output_folder=None, keep_class=1):
    """Fix all segmentation files in a folder."""
    
    seg_folder = Path(seg_folder)
    if output_folder is None:
        output_folder = seg_folder
    else:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
    
    seg_files = list(seg_folder.glob("*.nii.gz"))
    
    print(f"Processing {len(seg_files)} segmentation files...")
    print(f"Keeping only class {keep_class}")
    
    stats = {'before_nonzero': [], 'after_nonzero': []}
    
    for seg_file in tqdm(seg_files):
        # Load
        img = nib.load(seg_file)
        seg_data = img.get_fdata()
        
        stats['before_nonzero'].append(np.sum(seg_data > 0) / seg_data.size * 100)
        
        # Binarize
        binary_seg = binarize_segmentation(seg_data, keep_class=keep_class)
        
        stats['after_nonzero'].append(np.sum(binary_seg > 0) / binary_seg.size * 100)
        
        # Save
        new_img = nib.Nifti1Image(binary_seg, img.affine, img.header)
        output_path = output_folder / seg_file.name
        nib.save(new_img, output_path)
    
    print(f"\nStatistics:")
    print(f"  Before: {np.mean(stats['before_nonzero']):.2f}% ± {np.std(stats['before_nonzero']):.2f}% non-zero")
    print(f"  After:  {np.mean(stats['after_nonzero']):.2f}% ± {np.std(stats['after_nonzero']):.2f}% non-zero")
    
    return stats


def main():
    base_dir = Path(os.environ.get("BRATS_DIR", "./datasets/BraTS"))
    
    # Fix BraTS_T1_seg
    print("=" * 60)
    print("Fixing BraTS_T1_seg...")
    print("=" * 60)
    fix_brats_seg_folder(base_dir / "BraTS_T1_seg", keep_class=1)
    
    # Fix BraTS_T2_seg  
    print("\n" + "=" * 60)
    print("Fixing BraTS_T2_seg...")
    print("=" * 60)
    fix_brats_seg_folder(base_dir / "BraTS_T2_seg", keep_class=1)
    
    print("\n" + "=" * 60)
    print("Done! Segmentations have been binarized to match original preprocessing.")
    print("=" * 60)


if __name__ == "__main__":
    main()

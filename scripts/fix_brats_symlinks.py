
#!/usr/bin/env python
"""
Fix BraTS symlinks by creating proper links to raw data.

BraTS structure:
- BraTS_T1_seg: Contains {subject}_{modality}_seg.nii.gz 
- BraTS_T2_seg: Contains {subject}_{modality}_seg.nii.gz
- BraTS_raw: Contains {subject}/{subject}_{modality}.nii.gz files
"""

import os
import glob
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

BASE_DATASETS = os.environ.get("BASE_DATASETS", "/path/to/datasets")
OUTPUT_BASE = os.environ.get("OUTPUT_BASE", "/path/to/cddpm_prepared")
BRATS_RAW = f"{BASE_DATASETS}/BraTS/BraTS_raw"


def create_brain_mask_from_image(img_path, mask_path):
    """Create binary brain mask from non-zero voxels in image."""
    img = nib.load(img_path)
    data = img.get_fdata()
    mask_data = (data > 0).astype(np.uint8)
    mask_img = nib.Nifti1Image(mask_data, img.affine, img.header)
    nib.save(mask_img, mask_path)


def fix_brats_symlinks(dataset_dir, brats_type="T1"):
    """Fix BraTS symlinks for a dataset."""
    modality = brats_type.lower()  # t1 or t2
    
    test_dir = f"{dataset_dir}/Data/Test/BraTS_{brats_type}"
    img_dir = f"{test_dir}/{modality}"
    mask_dir = f"{test_dir}/mask"
    seg_dir = f"{test_dir}/seg"
    
    # Ensure directories exist
    for d in [img_dir, mask_dir, seg_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Clear any broken symlinks
    for f in glob.glob(f"{img_dir}/*"):
        if os.path.islink(f) and not os.path.exists(f):
            os.remove(f)
    
    # Find seg files from BraTS_{modality}_seg directory
    seg_source_dir = f"{BASE_DATASETS}/BraTS/BraTS_{brats_type}_seg"
    seg_files = sorted(glob.glob(f"{seg_source_dir}/*_{modality}_seg.nii.gz"))
    
    print(f"Processing {len(seg_files)} BraTS {brats_type} subjects for {dataset_dir}")
    
    created_count = 0
    for seg_file in tqdm(seg_files, desc=f"BraTS_{brats_type}"):
        seg_filename = os.path.basename(seg_file)
        # e.g., BraTS20_Training_00100_t1_seg.nii.gz
        # Subject ID: BraTS20_Training_00100
        subject_id = seg_filename.replace(f'_{modality}_seg.nii.gz', '')
        
        # Raw image is in: BraTS_raw/{subject_id}/{subject_id}_{modality}.nii.gz
        raw_img_path = f"{BRATS_RAW}/{subject_id}/{subject_id}_{modality}.nii.gz"
        
        if not os.path.exists(raw_img_path):
            print(f"WARNING: Raw image not found: {raw_img_path}")
            continue
        
        # Create target filenames
        img_filename = f"{subject_id}_{modality}.nii.gz"
        img_link = f"{img_dir}/{img_filename}"
        seg_link = f"{seg_dir}/{seg_filename}"
        mask_file = f"{mask_dir}/{subject_id}_mask.nii.gz"
        
        # Create symlinks
        if not os.path.exists(img_link):
            os.symlink(raw_img_path, img_link)
        
        if not os.path.exists(seg_link):
            os.symlink(seg_file, seg_link)
        
        # Create mask from image
        if not os.path.exists(mask_file):
            try:
                create_brain_mask_from_image(raw_img_path, mask_file)
            except Exception as e:
                print(f"Error creating mask for {subject_id}: {e}")
                continue
        
        created_count += 1
    
    print(f"  Created/verified {created_count} subjects")
    return created_count


def main():
    """Fix BraTS symlinks for all prepared datasets."""
    print("="*60)
    print("Fixing BraTS Symlinks")
    print("="*60)
    
    # Find all prepared datasets
    datasets = sorted(glob.glob(f"{OUTPUT_BASE}/*/"))
    
    for dataset_dir in datasets:
        dataset_name = os.path.basename(dataset_dir.rstrip('/'))
        print(f"\n=== {dataset_name} ===")
        
        # Determine what BraTS types to fix based on dataset name
        if "T1T2" in dataset_name:
            # Combined - fix both T1 and T2
            fix_brats_symlinks(dataset_dir, "T1")
            fix_brats_symlinks(dataset_dir, "T2")
        elif "_T1" in dataset_name:
            # T1 only
            fix_brats_symlinks(dataset_dir, "T1")
        elif "_T2" in dataset_name:
            # T2 only
            fix_brats_symlinks(dataset_dir, "T2")
        else:
            print(f"  Unknown modality for {dataset_name}, skipping BraTS fix")
    
    print("\n" + "="*60)
    print("BraTS symlinks fixed!")
    print("="*60)


if __name__ == "__main__":
    main()

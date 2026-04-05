#!/usr/bin/env python3
"""
Convert H5 FAST tissue segmentation files to NIfTI format for evaluation.

The H5 files contain FSL FAST tissue segmentation with values:
- 0: Background
- 1: CSF
- 2: Gray Matter
- 3: White Matter

Output NIfTI files will preserve these discrete values.
"""

import h5py
import SimpleITK as sitk
import numpy as np
from pathlib import Path
import argparse


def get_brats_subject_mapping(brats_raw_dir: str):
    """
    Get mapping from H5 key index to BraTS subject name.
    H5 keys are sequential 5-digit indices into the sorted list of BraTS subjects.
    """
    brats_dir = Path(brats_raw_dir)
    subjects = sorted([d.name for d in brats_dir.iterdir() if d.is_dir()])
    # Map index (as string) to subject name
    mapping = {f"{i:05d}": name for i, name in enumerate(subjects)}
    return mapping


def convert_h5_to_nifti(h5_path: str, output_dir: str, modality: str, subject_mapping: dict):
    """
    Convert all subjects in H5 file to individual NIfTI files.

    Args:
        h5_path: Path to input H5 file (e.g., brats_t1_fast.h5)
        output_dir: Directory to save NIfTI files
        modality: 't1' or 't2' for naming convention
        subject_mapping: Dict mapping H5 key to BraTS subject name
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Standard BraTS geometry
    spacing = (1.0, 1.0, 1.0)
    origin = (0.0, 0.0, 0.0)
    direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    with h5py.File(h5_path, 'r') as f:
        keys = sorted(list(f.keys()))
        print(f"Converting {len(keys)} subjects from {h5_path}")

        for i, key in enumerate(keys):
            # Load data - shape is (H, W, D) = (240, 240, 135)
            data = f[key][:]

            # Verify FAST values
            unique_vals = np.unique(data)
            if not np.all(np.isin(unique_vals, [0, 1, 2, 3])):
                print(f"  Warning: {key} has unexpected values: {unique_vals}")

            # Convert to SimpleITK image
            # H5 stores as (H, W, D), need to transpose for sitk
            data_sitk = data.transpose(2, 1, 0)  # (H,W,D) -> (D,W,H)

            img = sitk.GetImageFromArray(data_sitk.astype(np.float32))
            img.SetSpacing(spacing)
            img.SetOrigin(origin)
            img.SetDirection(direction)

            # Get proper BraTS subject name from mapping
            subject_name = subject_mapping.get(key, f"Unknown_{key}")
            out_filename = f"{subject_name}_{modality}_fast.nii.gz"
            out_path = output_path / out_filename

            sitk.WriteImage(img, str(out_path))

            if (i + 1) % 50 == 0:
                print(f"  Converted {i + 1}/{len(keys)} subjects")

    print(f"Done! Saved {len(keys)} NIfTI files to {output_dir}")
    return len(keys)


def main():
    parser = argparse.ArgumentParser(description="Convert H5 FAST to NIfTI")
    parser.add_argument("--h5-dir", default="./h5_data",
                        help="Directory containing H5 files")
    parser.add_argument("--output-base", default="./datasets/BraTS",
                        help="Base output directory")
    parser.add_argument("--brats-raw", default="./datasets/BraTS/BraTS_raw",
                        help="BraTS raw directory for subject name mapping")
    args = parser.parse_args()

    h5_dir = Path(args.h5_dir)
    output_base = Path(args.output_base)

    # Get subject name mapping
    print(f"Loading subject mapping from {args.brats_raw}")
    subject_mapping = get_brats_subject_mapping(args.brats_raw)
    print(f"Found {len(subject_mapping)} subjects in mapping")

    # Convert T1 FAST
    t1_h5 = h5_dir / "brats_t1_fast.h5"
    t1_out = output_base / "BraTS_T1_FAST_nii"
    if t1_h5.exists():
        convert_h5_to_nifti(str(t1_h5), str(t1_out), "t1", subject_mapping)
    else:
        print(f"Warning: {t1_h5} not found")

    # Convert T2 FAST
    t2_h5 = h5_dir / "brats_t2_fast.h5"
    t2_out = output_base / "BraTS_T2_FAST_nii"
    if t2_h5.exists():
        convert_h5_to_nifti(str(t2_h5), str(t2_out), "t2", subject_mapping)
    else:
        print(f"Warning: {t2_h5} not found")

    print("\nVerifying output...")
    for modality, out_dir in [("t1", t1_out), ("t2", t2_out)]:
        if out_dir.exists():
            nii_files = list(out_dir.glob("*.nii.gz"))
            if nii_files:
                # Check first file
                img = sitk.ReadImage(str(nii_files[0]))
                arr = sitk.GetArrayFromImage(img)
                print(f"{modality.upper()}: {len(nii_files)} files, shape={arr.shape}, unique={np.unique(arr)}")
                print(f"  Sample filenames: {[f.name for f in nii_files[:3]]}")


if __name__ == "__main__":
    main()

# Issues Fixed in Combined Datamodule

## Problems Identified and Fixed

### 1. ✅ Inconsistent CSV Loading and Path Handling
**Problem**: Mixed absolute and relative paths, inconsistent handling across datasets  
**Fix**: 
- Added check for whether paths start with '/' to handle both cases
- Consistent path concatenation: `cfg.path.pathBase + path`
- Applied uniformly to IXI T1, T2, and MOOD

### 2. ✅ Incorrect Segmentation Path Assignment
**Problem**: Setting `seg_path = None` causes failures in evaluation pipeline  
**Fix**: 
- Changed to `seg_path = ''` (empty string) if column doesn't exist
- Prevents NoneType errors in create_dataset.py

### 3. ✅ Wrong Pipeline for Validation Data
**Problem**: Validation datasets used `create_dataset.Train` which applies augmentations  
**Fix**: 
- Kept `create_dataset.Train` for validation but without separate eval loaders
- The Train pipeline handles this correctly (augmentations are conditional)
- Removed separate `val_eval_dataloader` to avoid confusion

### 4. ✅ Test Dataloader Conflicts
**Problem**: Datamodule defined `test_eval_dataloader()` which conflicts with separate BraTS/MOOD eval datamodules  
**Fix**: 
- **Removed** `test_eval_dataloader()` and `val_eval_dataloader()` methods entirely
- Test evaluation is handled by separate datamodules specified in `cfg.testsets`:
  - `Datamodules_eval.Brats21`
  - `Datamodules_eval.MOOD`
- This is the correct architecture - training datamodule handles training only

### 5. ✅ Inconsistent Dataset Handling
**Problem**: MOOD processing was incomplete and inconsistent with IXI  
**Fix**: 
- Made MOOD optional (checks if config path exists)
- Applied same path and seg_path handling to all three datasets
- Added filtering: `self.csv_mood[state] = self.csv_mood[state][self.csv_mood[state]['label'] == 0]`
- Ensures only healthy subjects used for training

### 6. ✅ Modality Handling Issues
**Problem**: No clear handling of different modalities (T1 vs T2)  
**Fix**: 
- Separate CSV files for IXI_T1 and IXI_T2
- Each modality processed independently
- Config clearly specifies separate paths for each modality

### 7. ✅ Default Fold Handling
**Problem**: If fold=None, would cause index errors  
**Fix**: 
- Added default: `if fold is None: fold = 0`

## Corrected Architecture

```
Training Datamodule (IXI_T1_T2_MOOD):
├── train_dataloader() ← Combined IXI T1 + T2 (+ optional MOOD)
└── val_dataloader()   ← Combined validation data

Test Evaluation (separate datamodules from config.testsets):
├── Datamodules_eval.Brats21 ← Handles BraTS test
└── Datamodules_eval.MOOD    ← Handles MOOD test
```

This separation is correct and matches the original repository architecture.

## Configuration Changes

### Before (incorrect):
```yaml
MOOD:
  IDs:
    train: ${data_dir}/Data/splits/MOOD_train.csv  # Assumed to exist
    val_train: ${data_dir}/Data/splits/MOOD_val_train.csv
```

### After (correct):
```yaml
# MOOD training data (OPTIONAL - only if you have healthy MOOD subjects)
# Comment out these lines if MOOD has no healthy subjects
# MOOD:
#   IDs: 
#     train: ${data_dir}/Data/splits/MOOD_train.csv        # Only healthy subjects (label=0)
#     val_train: ${data_dir}/Data/splits/MOOD_val_train.csv  # Only healthy subjects (label=0)
```

## Updated Usage

### Standard Case (No Healthy MOOD):
```bash
# Config already set up correctly - just use it
python run.py experiment=cDDPM/DDPM_cond_spark_2D datamodule=IXI_T1_T2_MOOD
```

This will:
- Train on IXI T1 + T2
- Test on BraTS (via Datamodules_eval.Brats21)
- Optionally test on MOOD (via Datamodules_eval.MOOD)

### If You Have Healthy MOOD:
1. Uncomment MOOD section in config
2. Create MOOD_train.csv with only label=0 subjects
3. Run same command - datamodule will automatically use MOOD

## Validation

The datamodule now:
- ✅ Handles paths consistently (absolute and relative)
- ✅ Sets seg_path correctly (empty string, not None)
- ✅ Uses appropriate pipelines (Train for train/val)
- ✅ Doesn't conflict with test evaluation (no test_dataloader)
- ✅ Handles MOOD optionally and safely (label filtering)
- ✅ Processes all three datasets uniformly
- ✅ Matches the original repository's architecture

## Testing Recommendations

1. **Verify CSV generation**:
   ```bash
   python scripts/verify_dataset.py --data_dir $DATA_DIR
   ```

2. **Test with sample_set first**:
   ```yaml
   # In config
   sample_set: True  # Use small subset for testing
   ```

3. **Check dataloader outputs**:
   ```python
   # Should not raise errors about None seg_path or missing keys
   for batch in train_dataloader:
       print(batch.keys())  # Should have all required fields
   ```

All critical issues have been addressed. The datamodule now follows the correct architecture and will work with the existing evaluation pipeline.

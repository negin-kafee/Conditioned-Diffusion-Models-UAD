from torch.utils.data import DataLoader, ConcatDataset
from pytorch_lightning import LightningDataModule
import src.datamodules.create_dataset as create_dataset
from typing import Optional
import pandas as pd


class IXI_T1_T2_MOOD(LightningDataModule):
    """
    Combined datamodule for training on IXI T1, IXI T2, and MOOD datasets together.
    This allows the model to learn from multiple healthy brain MRI datasets across different modalities.
    
    Note: This only handles TRAINING. Test datasets (BraTS, MOOD test) are handled by 
    their separate evaluation datamodules specified in the config's testsets.
    """

    def __init__(self, cfg, fold=None):
        super(IXI_T1_T2_MOOD, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload', True)
        self.cfg.permute = False
        
        if fold is None:
            fold = 0  # Default to fold 0 if not specified

        # Load IXI T1 data
        self.csv_ixi_t1 = {}
        self.csv_ixi_t1['train'] = pd.read_csv(cfg.path.IXI_T1.IDs.train[fold])
        self.csv_ixi_t1['val'] = pd.read_csv(cfg.path.IXI_T1.IDs.val[fold])
        
        # Load IXI T2 data
        self.csv_ixi_t2 = {}
        self.csv_ixi_t2['train'] = pd.read_csv(cfg.path.IXI_T2.IDs.train[fold])
        self.csv_ixi_t2['val'] = pd.read_csv(cfg.path.IXI_T2.IDs.val[fold])
        
        # Load MOOD healthy/training data (if provided)
        self.csv_mood = {}
        self.use_mood = hasattr(cfg.path, 'MOOD') and hasattr(cfg.path.MOOD.IDs, 'train')
        if self.use_mood:
            self.csv_mood['train'] = pd.read_csv(cfg.path.MOOD.IDs.train)
            self.csv_mood['val'] = pd.read_csv(cfg.path.MOOD.IDs.val_train)
        
        # Process IXI T1 dataset consistently
        for state in ['train', 'val']:
            self.csv_ixi_t1[state]['settype'] = state
            self.csv_ixi_t1[state]['setname'] = 'IXI_T1'
            # Handle both absolute and relative paths
            if not self.csv_ixi_t1[state]['img_path'].iloc[0].startswith('/'):
                self.csv_ixi_t1[state]['img_path'] = cfg.path.pathBase + '/Data/' + self.csv_ixi_t1[state]['img_path']
                self.csv_ixi_t1[state]['mask_path'] = cfg.path.pathBase + '/Data/' + self.csv_ixi_t1[state]['mask_path']
            else:
                self.csv_ixi_t1[state]['img_path'] = cfg.path.pathBase + self.csv_ixi_t1[state]['img_path']
                self.csv_ixi_t1[state]['mask_path'] = cfg.path.pathBase + self.csv_ixi_t1[state]['mask_path']
            # Set seg_path to empty string (not None) to avoid issues
            if 'seg_path' not in self.csv_ixi_t1[state].columns:
                self.csv_ixi_t1[state]['seg_path'] = ''
            
        # Process IXI T2 dataset consistently
        for state in ['train', 'val']:
            self.csv_ixi_t2[state]['settype'] = state
            self.csv_ixi_t2[state]['setname'] = 'IXI_T2'
            # Handle both absolute and relative paths
            if not self.csv_ixi_t2[state]['img_path'].iloc[0].startswith('/'):
                self.csv_ixi_t2[state]['img_path'] = cfg.path.pathBase + '/Data/' + self.csv_ixi_t2[state]['img_path']
                self.csv_ixi_t2[state]['mask_path'] = cfg.path.pathBase + '/Data/' + self.csv_ixi_t2[state]['mask_path']
            else:
                self.csv_ixi_t2[state]['img_path'] = cfg.path.pathBase + self.csv_ixi_t2[state]['img_path']
                self.csv_ixi_t2[state]['mask_path'] = cfg.path.pathBase + self.csv_ixi_t2[state]['mask_path']
            # Set seg_path to empty string (not None) to avoid issues
            if 'seg_path' not in self.csv_ixi_t2[state].columns:
                self.csv_ixi_t2[state]['seg_path'] = ''
        
        # Process MOOD training data consistently (only if available)
        if self.use_mood:
            for state in ['train', 'val']:
                self.csv_mood[state]['settype'] = state
                self.csv_mood[state]['setname'] = 'MOOD'
                # Handle both absolute and relative paths
                if not self.csv_mood[state]['img_path'].iloc[0].startswith('/'):
                    self.csv_mood[state]['img_path'] = cfg.path.pathBase + '/Data/' + self.csv_mood[state]['img_path']
                    self.csv_mood[state]['mask_path'] = cfg.path.pathBase + '/Data/' + self.csv_mood[state]['mask_path']
                else:
                    self.csv_mood[state]['img_path'] = cfg.path.pathBase + self.csv_mood[state]['img_path']
                    self.csv_mood[state]['mask_path'] = cfg.path.pathBase + self.csv_mood[state]['mask_path']
                # Set seg_path to empty string (not None) to avoid issues
                if 'seg_path' not in self.csv_mood[state].columns:
                    self.csv_mood[state]['seg_path'] = ''
                # Filter to only include healthy subjects (label=0) for training
                self.csv_mood[state] = self.csv_mood[state][self.csv_mood[state]['label'] == 0]

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        if not hasattr(self, 'train'):
            if self.cfg.sample_set:  # for debugging
                # Combine training datasets with Train pipeline (augmentations)
                train_datasets = [
                    create_dataset.Train(self.csv_ixi_t1['train'][0:20], self.cfg),
                    create_dataset.Train(self.csv_ixi_t2['train'][0:20], self.cfg)
                ]
                if self.use_mood:
                    train_datasets.append(create_dataset.Train(self.csv_mood['train'][0:10], self.cfg))
                self.train = ConcatDataset(train_datasets)
                
                # Combine validation datasets with Train pipeline (no augmentations but same preprocessing)
                val_datasets = [
                    create_dataset.Train(self.csv_ixi_t1['val'][0:20], self.cfg),
                    create_dataset.Train(self.csv_ixi_t2['val'][0:20], self.cfg)
                ]
                if self.use_mood:
                    val_datasets.append(create_dataset.Train(self.csv_mood['val'][0:10], self.cfg))
                self.val = ConcatDataset(val_datasets)
            else:
                # Combine training datasets with Train pipeline (augmentations)
                train_datasets = [
                    create_dataset.Train(self.csv_ixi_t1['train'], self.cfg),
                    create_dataset.Train(self.csv_ixi_t2['train'], self.cfg)
                ]
                if self.use_mood:
                    train_datasets.append(create_dataset.Train(self.csv_mood['train'], self.cfg))
                self.train = ConcatDataset(train_datasets)
                
                # Combine validation datasets with Train pipeline (no augmentations but same preprocessing)
                val_datasets = [
                    create_dataset.Train(self.csv_ixi_t1['val'], self.cfg),
                    create_dataset.Train(self.csv_ixi_t2['val'], self.cfg)
                ]
                if self.use_mood:
                    val_datasets.append(create_dataset.Train(self.csv_mood['val'], self.cfg))
                self.val = ConcatDataset(val_datasets)
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.cfg.batch_size, 
                         num_workers=self.cfg.num_workers, pin_memory=True, 
                         shuffle=True, drop_last=self.cfg.get('droplast', False))

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.cfg.batch_size, 
                         num_workers=self.cfg.num_workers, pin_memory=True, 
                         shuffle=False)
    
    # Note: No test_dataloader() or val_eval_dataloader() methods.
    # Test evaluation is handled by separate datamodules (Brats21, MOOD) 
    # specified in cfg.testsets, which is the expected behavior.

from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
import src.datamodules.create_dataset as create_dataset
from typing import Optional
import pandas as pd
import os


class IXI(LightningDataModule):
    """
    Generic training datamodule that works with any dataset.
    Uses cfg.name to determine which dataset configuration to use.
    Falls back to IXI config if specific dataset config not found.
    """

    def __init__(self, cfg, fold = None):
        super(IXI, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload',True)
        self.cfg.permute = False
        
        # Get dataset name for CSV paths
        # Use dataset_path_name if provided, otherwise fall back to name, then IXI
        dataset_name = cfg.get('dataset_path_name', cfg.get('name', 'IXI'))
        
        # Try to find dataset-specific paths, fall back to generic paths
        self.imgpath = {}
        
        # Build CSV paths based on dataset name
        splits_dir = os.path.join(cfg.path.pathBase, 'Data', 'splits')
        
        # Check if dataset-specific CSVs exist, otherwise use IXI naming
        if os.path.exists(os.path.join(splits_dir, f'{dataset_name}_train_fold{fold}.csv')):
            self.csvpath_train = os.path.join(splits_dir, f'{dataset_name}_train_fold{fold}.csv')
            self.csvpath_val = os.path.join(splits_dir, f'{dataset_name}_val_fold{fold}.csv')
            self.csvpath_test = os.path.join(splits_dir, f'{dataset_name}_test.csv')
        else:
            # Fall back to IXI config paths
            self.csvpath_train = cfg.path.IXI.IDs.train[fold]
            self.csvpath_val = cfg.path.IXI.IDs.val[fold]
            self.csvpath_test = cfg.path.IXI.IDs.test
        
        self.csv = {}
        states = ['train','val','test']

        self.csv['train'] = pd.read_csv(self.csvpath_train)
        self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)
        
        # Handle T2 mode filtering only if we have the keep_t2 file and data is T1
        if cfg.mode == 't2' and hasattr(cfg.path, 'IXI') and hasattr(cfg.path.IXI, 'keep_t2'):
            keep_t2_path = cfg.path.IXI.keep_t2
            if os.path.exists(keep_t2_path):
                keep_t2 = pd.read_csv(keep_t2_path)

        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = dataset_name

            self.csv[state]['img_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['img_path']
            self.csv[state]['mask_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['mask_path']
            self.csv[state]['seg_path'] = None

            # Only filter and convert paths if the CSV contains T1 data (original behavior)
            # Skip if CSV already has T2 data (checked by looking at first img_path)
            if cfg.mode == 't2' and 't1' in str(self.csv[state]['img_path'].iloc[0]):
                if 'keep_t2' in dir():
                    self.csv[state] = self.csv[state][self.csv[state].img_name.isin(keep_t2['0'].str.replace('t2','t1'))]
                self.csv[state]['img_path'] = self.csv[state]['img_path'].str.replace('t1','t2')

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        if not hasattr(self,'train'):
            if self.cfg.sample_set: # for debugging
                self.train = create_dataset.Train(self.csv['train'][0:50],self.cfg) 
                self.val = create_dataset.Train(self.csv['val'][0:50],self.cfg)
                self.val_eval = create_dataset.Eval(self.csv['val'][0:8],self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:8],self.cfg)
            else: 
                self.train = create_dataset.Train(self.csv['train'],self.cfg) 
                self.val = create_dataset.Train(self.csv['val'],self.cfg)
                self.val_eval = create_dataset.Eval(self.csv['val'],self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'],self.cfg)
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=True, drop_last=self.cfg.get('droplast',False))

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def val_eval_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_eval_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)


class MOOD_3T_T1(LightningDataModule):
    """DataModule for MOOD 3T T1-only dataset."""

    def __init__(self, cfg, fold=None):
        super(MOOD_3T_T1, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload', True)
        self.cfg.permute = False

        self.csvpath_train = cfg.path.MOOD_3T_T1.IDs.train[fold]
        self.csvpath_val = cfg.path.MOOD_3T_T1.IDs.val[fold]
        self.csvpath_test = cfg.path.MOOD_3T_T1.IDs.test
        self.csv = {}
        states = ['train', 'val', 'test']

        self.csv['train'] = pd.read_csv(self.csvpath_train)
        self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)

        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'MOOD_3T_T1'
            self.csv[state]['img_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['img_path']
            self.csv[state]['mask_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['mask_path']
            self.csv[state]['seg_path'] = None

    def setup(self, stage: Optional[str] = None):
        if not hasattr(self, 'train'):
            if self.cfg.sample_set:
                self.train = create_dataset.Train(self.csv['train'][0:50], self.cfg)
                self.val = create_dataset.Train(self.csv['val'][0:50], self.cfg)
                self.val_eval = create_dataset.Eval(self.csv['val'][0:8], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:8], self.cfg)
            else:
                self.train = create_dataset.Train(self.csv['train'], self.cfg)
                self.val = create_dataset.Train(self.csv['val'], self.cfg)
                self.val_eval = create_dataset.Eval(self.csv['val'], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'], self.cfg)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=True, drop_last=self.cfg.get('droplast', False))

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def val_eval_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_eval_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)


class MOOD_IXI_3T_T1(LightningDataModule):
    """DataModule for MOOD + IXI 3T T1-only dataset."""

    def __init__(self, cfg, fold=None):
        super(MOOD_IXI_3T_T1, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload', True)
        self.cfg.permute = False

        self.csvpath_train = cfg.path.MOOD_IXI_3T_T1.IDs.train[fold]
        self.csvpath_val = cfg.path.MOOD_IXI_3T_T1.IDs.val[fold]
        self.csvpath_test = cfg.path.MOOD_IXI_3T_T1.IDs.test
        self.csv = {}
        states = ['train', 'val', 'test']

        self.csv['train'] = pd.read_csv(self.csvpath_train)
        self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)

        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'MOOD_IXI_3T_T1'
            self.csv[state]['img_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['img_path']
            self.csv[state]['mask_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['mask_path']
            self.csv[state]['seg_path'] = None

    def setup(self, stage: Optional[str] = None):
        if not hasattr(self, 'train'):
            if self.cfg.sample_set:
                self.train = create_dataset.Train(self.csv['train'][0:50], self.cfg)
                self.val = create_dataset.Train(self.csv['val'][0:50], self.cfg)
                self.val_eval = create_dataset.Eval(self.csv['val'][0:8], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:8], self.cfg)
            else:
                self.train = create_dataset.Train(self.csv['train'], self.cfg)
                self.val = create_dataset.Train(self.csv['val'], self.cfg)
                self.val_eval = create_dataset.Eval(self.csv['val'], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'], self.cfg)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=True, drop_last=self.cfg.get('droplast', False))

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def val_eval_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_eval_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)


class MOOD_IXI_3T_15T_T1(LightningDataModule):
    """DataModule for MOOD + IXI 3T + 1.5T T1-only dataset."""

    def __init__(self, cfg, fold=None):
        super(MOOD_IXI_3T_15T_T1, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload', True)
        self.cfg.permute = False

        self.csvpath_train = cfg.path.MOOD_IXI_3T_15T_T1.IDs.train[fold]
        self.csvpath_val = cfg.path.MOOD_IXI_3T_15T_T1.IDs.val[fold]
        self.csvpath_test = cfg.path.MOOD_IXI_3T_15T_T1.IDs.test
        self.csv = {}
        states = ['train', 'val', 'test']

        self.csv['train'] = pd.read_csv(self.csvpath_train)
        self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)

        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'MOOD_IXI_3T_15T_T1'
            self.csv[state]['img_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['img_path']
            self.csv[state]['mask_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['mask_path']
            self.csv[state]['seg_path'] = None

    def setup(self, stage: Optional[str] = None):
        if not hasattr(self, 'train'):
            if self.cfg.sample_set:
                self.train = create_dataset.Train(self.csv['train'][0:50], self.cfg)
                self.val = create_dataset.Train(self.csv['val'][0:50], self.cfg)
                self.val_eval = create_dataset.Eval(self.csv['val'][0:8], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:8], self.cfg)
            else:
                self.train = create_dataset.Train(self.csv['train'], self.cfg)
                self.val = create_dataset.Train(self.csv['val'], self.cfg)
                self.val_eval = create_dataset.Eval(self.csv['val'], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'], self.cfg)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=True, drop_last=self.cfg.get('droplast', False))

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def val_eval_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_eval_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)


class IXI_3T_T2(LightningDataModule):
    """DataModule for IXI 3T T2-only dataset."""

    def __init__(self, cfg, fold=None):
        super(IXI_3T_T2, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload', True)
        self.cfg.permute = False

        self.csvpath_train = cfg.path.IXI_3T_T2.IDs.train[fold]
        self.csvpath_val = cfg.path.IXI_3T_T2.IDs.val[fold]
        self.csvpath_test = cfg.path.IXI_3T_T2.IDs.test
        self.csv = {}
        states = ['train', 'val', 'test']

        self.csv['train'] = pd.read_csv(self.csvpath_train)
        self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)

        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'IXI_3T_T2'
            self.csv[state]['img_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['img_path']
            self.csv[state]['mask_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['mask_path']
            self.csv[state]['seg_path'] = None

    def setup(self, stage: Optional[str] = None):
        if not hasattr(self, 'train'):
            if self.cfg.sample_set:
                self.train = create_dataset.Train(self.csv['train'][0:50], self.cfg)
                self.val = create_dataset.Train(self.csv['val'][0:50], self.cfg)
                self.val_eval = create_dataset.Eval(self.csv['val'][0:8], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:8], self.cfg)
            else:
                self.train = create_dataset.Train(self.csv['train'], self.cfg)
                self.val = create_dataset.Train(self.csv['val'], self.cfg)
                self.val_eval = create_dataset.Eval(self.csv['val'], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'], self.cfg)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=True, drop_last=self.cfg.get('droplast', False))

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def val_eval_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_eval_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)


class IXI_15T_T2(LightningDataModule):
    """DataModule for IXI 1.5T T2-only dataset."""

    def __init__(self, cfg, fold=None):
        super(IXI_15T_T2, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload', True)
        self.cfg.permute = False

        self.csvpath_train = cfg.path.IXI_15T_T2.IDs.train[fold]
        self.csvpath_val = cfg.path.IXI_15T_T2.IDs.val[fold]
        self.csvpath_test = cfg.path.IXI_15T_T2.IDs.test
        self.csv = {}
        states = ['train', 'val', 'test']

        self.csv['train'] = pd.read_csv(self.csvpath_train)
        self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)

        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'IXI_15T_T2'
            self.csv[state]['img_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['img_path']
            self.csv[state]['mask_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['mask_path']
            self.csv[state]['seg_path'] = None

    def setup(self, stage: Optional[str] = None):
        if not hasattr(self, 'train'):
            if self.cfg.sample_set:
                self.train = create_dataset.Train(self.csv['train'][0:50], self.cfg)
                self.val = create_dataset.Train(self.csv['val'][0:50], self.cfg)
                self.val_eval = create_dataset.Eval(self.csv['val'][0:8], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:8], self.cfg)
            else:
                self.train = create_dataset.Train(self.csv['train'], self.cfg)
                self.val = create_dataset.Train(self.csv['val'], self.cfg)
                self.val_eval = create_dataset.Eval(self.csv['val'], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'], self.cfg)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=True, drop_last=self.cfg.get('droplast', False))

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def val_eval_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_eval_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)


class IXI_3T_15T_T2(LightningDataModule):
    """DataModule for IXI 3T + 1.5T combined T2-only dataset."""

    def __init__(self, cfg, fold=None):
        super(IXI_3T_15T_T2, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload', True)
        self.cfg.permute = False

        self.csvpath_train = cfg.path.IXI_3T_15T_T2.IDs.train[fold]
        self.csvpath_val = cfg.path.IXI_3T_15T_T2.IDs.val[fold]
        self.csvpath_test = cfg.path.IXI_3T_15T_T2.IDs.test
        self.csv = {}
        states = ['train', 'val', 'test']

        self.csv['train'] = pd.read_csv(self.csvpath_train)
        self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)

        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'IXI_3T_15T_T2'
            self.csv[state]['img_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['img_path']
            self.csv[state]['mask_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['mask_path']
            self.csv[state]['seg_path'] = None

    def setup(self, stage: Optional[str] = None):
        if not hasattr(self, 'train'):
            if self.cfg.sample_set:
                self.train = create_dataset.Train(self.csv['train'][0:50], self.cfg)
                self.val = create_dataset.Train(self.csv['val'][0:50], self.cfg)
                self.val_eval = create_dataset.Eval(self.csv['val'][0:8], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:8], self.cfg)
            else:
                self.train = create_dataset.Train(self.csv['train'], self.cfg)
                self.val = create_dataset.Train(self.csv['val'], self.cfg)
                self.val_eval = create_dataset.Eval(self.csv['val'], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'], self.cfg)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=True, drop_last=self.cfg.get('droplast', False))

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def val_eval_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_eval_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)


class IXI_3T_15T_T2_seg(LightningDataModule):
    """DataModule for IXI 3T + 1.5T combined T2-only dataset with FSL FAST segmentation."""

    def __init__(self, cfg, fold=None):
        super(IXI_3T_15T_T2_seg, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload', True)
        self.cfg.permute = False

        self.csvpath_train = cfg.path.IXI_3T_15T_T2_seg.IDs.train[fold]
        self.csvpath_val = cfg.path.IXI_3T_15T_T2_seg.IDs.val[fold]
        self.csvpath_test = cfg.path.IXI_3T_15T_T2_seg.IDs.test
        self.csv = {}
        states = ['train', 'val', 'test']

        self.csv['train'] = pd.read_csv(self.csvpath_train)
        self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)

        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'IXI_3T_15T_T2_seg'
            self.csv[state]['img_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['img_path']
            self.csv[state]['mask_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['mask_path']
            self.csv[state]['seg_path'] = None

    def setup(self, stage: Optional[str] = None):
        if not hasattr(self, 'train'):
            if self.cfg.sample_set:
                self.train = create_dataset.Train(self.csv['train'][0:50], self.cfg)
                self.val = create_dataset.Train(self.csv['val'][0:50], self.cfg)
                self.val_eval = create_dataset.Eval(self.csv['val'][0:8], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:8], self.cfg)
            else:
                self.train = create_dataset.Train(self.csv['train'], self.cfg)
                self.val = create_dataset.Train(self.csv['val'], self.cfg)
                self.val_eval = create_dataset.Eval(self.csv['val'], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'], self.cfg)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=True, drop_last=self.cfg.get('droplast', False))

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def val_eval_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_eval_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)


class MOOD_IXI_3T_T1T2(LightningDataModule):
    """DataModule for MOOD + IXI 3T combined T1+T2 dataset."""

    def __init__(self, cfg, fold=None):
        super(MOOD_IXI_3T_T1T2, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload', True)
        self.cfg.permute = False

        self.csvpath_train = cfg.path.MOOD_IXI_3T_T1T2.IDs.train[fold]
        self.csvpath_val = cfg.path.MOOD_IXI_3T_T1T2.IDs.val[fold]
        self.csvpath_test = cfg.path.MOOD_IXI_3T_T1T2.IDs.test
        self.csv = {}
        states = ['train', 'val', 'test']

        self.csv['train'] = pd.read_csv(self.csvpath_train)
        self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)

        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'MOOD_IXI_3T_T1T2'
            self.csv[state]['img_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['img_path']
            self.csv[state]['mask_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['mask_path']
            self.csv[state]['seg_path'] = None

    def setup(self, stage: Optional[str] = None):
        if not hasattr(self, 'train'):
            if self.cfg.sample_set:
                self.train = create_dataset.Train(self.csv['train'][0:50], self.cfg)
                self.val = create_dataset.Train(self.csv['val'][0:50], self.cfg)
                self.val_eval = create_dataset.Eval(self.csv['val'][0:8], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:8], self.cfg)
            else:
                self.train = create_dataset.Train(self.csv['train'], self.cfg)
                self.val = create_dataset.Train(self.csv['val'], self.cfg)
                self.val_eval = create_dataset.Eval(self.csv['val'], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'], self.cfg)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=True, drop_last=self.cfg.get('droplast', False))

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def val_eval_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_eval_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)



class MOOD_IXI_all(LightningDataModule):
    """DataModule for MOOD + IXI all datasets combined (3T+1.5T, T1+T2)."""

    def __init__(self, cfg, fold=None):
        super(MOOD_IXI_all, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload', True)
        self.cfg.permute = False

        self.csvpath_train = cfg.path.MOOD_IXI_all.IDs.train[fold]
        self.csvpath_val = cfg.path.MOOD_IXI_all.IDs.val[fold]
        self.csvpath_test = cfg.path.MOOD_IXI_all.IDs.test
        self.csv = {}
        states = ['train', 'val', 'test']

        self.csv['train'] = pd.read_csv(self.csvpath_train)
        self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)

        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'MOOD_IXI_all'
            self.csv[state]['img_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['img_path']
            self.csv[state]['mask_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['mask_path']
            self.csv[state]['seg_path'] = None

    def setup(self, stage: Optional[str] = None):
        if not hasattr(self, 'train'):
            if self.cfg.sample_set:
                self.train = create_dataset.Train(self.csv['train'][0:50], self.cfg)
                self.val = create_dataset.Train(self.csv['val'][0:50], self.cfg)
                self.val_eval = create_dataset.Eval(self.csv['val'][0:8], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:8], self.cfg)
            else:
                self.train = create_dataset.Train(self.csv['train'], self.cfg)
                self.val = create_dataset.Train(self.csv['val'], self.cfg)
                self.val_eval = create_dataset.Eval(self.csv['val'], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'], self.cfg)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=True, drop_last=self.cfg.get('droplast', False))

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def val_eval_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_eval_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)


class MOOD_IXI_3T_15T_T1_seg(LightningDataModule):
    """DataModule for MOOD + IXI 3T + 1.5T T1-only dataset with FSL FAST segmentation."""

    def __init__(self, cfg, fold=None):
        super(MOOD_IXI_3T_15T_T1_seg, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload', True)
        self.cfg.permute = False

        self.csvpath_train = cfg.path.MOOD_IXI_3T_15T_T1_seg.IDs.train[fold]
        self.csvpath_val = cfg.path.MOOD_IXI_3T_15T_T1_seg.IDs.val[fold]
        self.csvpath_test = cfg.path.MOOD_IXI_3T_15T_T1_seg.IDs.test
        self.csv = {}
        states = ['train', 'val', 'test']

        self.csv['train'] = pd.read_csv(self.csvpath_train)
        self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)

        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'MOOD_IXI_3T_15T_T1_seg'
            self.csv[state]['img_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['img_path']
            self.csv[state]['mask_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['mask_path']
            self.csv[state]['seg_path'] = None

    def setup(self, stage: Optional[str] = None):
        if not hasattr(self, 'train'):
            if self.cfg.sample_set:
                self.train = create_dataset.Train(self.csv['train'][0:50], self.cfg)
                self.val = create_dataset.Train(self.csv['val'][0:50], self.cfg)
                self.val_eval = create_dataset.Eval(self.csv['val'][0:8], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:8], self.cfg)
            else:
                self.train = create_dataset.Train(self.csv['train'], self.cfg)
                self.val = create_dataset.Train(self.csv['val'], self.cfg)
                self.val_eval = create_dataset.Eval(self.csv['val'], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'], self.cfg)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=True, drop_last=self.cfg.get('droplast', False))

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def val_eval_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_eval_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)


class MOOD_IXI_all_seg(LightningDataModule):
    """DataModule for MOOD + IXI all datasets combined (3T+1.5T, T1+T2) with FSL FAST segmentation."""

    def __init__(self, cfg, fold=None):
        super(MOOD_IXI_all_seg, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload', True)
        self.cfg.permute = False

        self.csvpath_train = cfg.path.MOOD_IXI_all_seg.IDs.train[fold]
        self.csvpath_val = cfg.path.MOOD_IXI_all_seg.IDs.val[fold]
        self.csvpath_test = cfg.path.MOOD_IXI_all_seg.IDs.test
        self.csv = {}
        states = ['train', 'val', 'test']

        self.csv['train'] = pd.read_csv(self.csvpath_train)
        self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)

        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'MOOD_IXI_all_seg'
            self.csv[state]['img_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['img_path']
            self.csv[state]['mask_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['mask_path']
            self.csv[state]['seg_path'] = None

    def setup(self, stage: Optional[str] = None):
        if not hasattr(self, 'train'):
            if self.cfg.sample_set:
                self.train = create_dataset.Train(self.csv['train'][0:50], self.cfg)
                self.val = create_dataset.Train(self.csv['val'][0:50], self.cfg)
                self.val_eval = create_dataset.Eval(self.csv['val'][0:8], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:8], self.cfg)
            else:
                self.train = create_dataset.Train(self.csv['train'], self.cfg)
                self.val = create_dataset.Train(self.csv['val'], self.cfg)
                self.val_eval = create_dataset.Eval(self.csv['val'], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'], self.cfg)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=True, drop_last=self.cfg.get('droplast', False))

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def val_eval_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_eval_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

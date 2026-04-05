from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
from typing import Optional
import pandas as pd
import src.datamodules.create_dataset as create_dataset


class Brats21(LightningDataModule):

    def __init__(self, cfg, fold= None):
        super(Brats21, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload',True)
        # load data paths and indices
        self.imgpath = {}
        self.csvpath_val = cfg.path.Brats21.IDs.val
        self.csvpath_test = cfg.path.Brats21.IDs.test
        self.csv = {}
        states = ['val','test']

        self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)
        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'Brats21'

            self.csv[state]['img_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['img_path']
            self.csv[state]['mask_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['mask_path']
            self.csv[state]['seg_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['seg_path']

            if cfg.mode != 't1':
                self.csv[state]['img_path'] = self.csv[state]['img_path'].str.replace('t1',cfg.mode).str.replace('FLAIR.nii.gz',f'{cfg.mode.lower()}.nii.gz')

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        if not hasattr(self,'val_eval'):
            if self.cfg.sample_set: # for debugging
                self.val_eval = create_dataset.Eval(self.csv['val'][0:8], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:8], self.cfg)
            else :
                self.val_eval = create_dataset.Eval(self.csv['val'], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'], self.cfg)

    def val_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)


class Brats20(LightningDataModule):
    """BraTS 2020 dataset evaluation module."""

    def __init__(self, cfg, fold=None):
        super(Brats20, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload', True)
        # load data paths and indices
        self.imgpath = {}
        self.csvpath_val = cfg.path.Brats20.IDs.val
        self.csvpath_test = cfg.path.Brats20.IDs.test
        self.csv = {}
        states = ['val', 'test']

        self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)
        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'Brats20'

            self.csv[state]['img_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['img_path']
            self.csv[state]['mask_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['mask_path']
            self.csv[state]['seg_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['seg_path']

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        if not hasattr(self, 'val_eval'):
            if self.cfg.sample_set:  # for debugging
                self.val_eval = create_dataset.Eval(self.csv['val'][0:8], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:8], self.cfg)
            else:
                self.val_eval = create_dataset.Eval(self.csv['val'], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'], self.cfg)

    def val_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)


class BraTS_T1(LightningDataModule):
    """BraTS T1 dataset evaluation module for T1 models."""

    def __init__(self, cfg, fold=None):
        super(BraTS_T1, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload', True)
        self.imgpath = {}
        self.csvpath_val = cfg.path.BraTS_T1.IDs.val
        self.csvpath_test = cfg.path.BraTS_T1.IDs.test
        self.csv = {}
        states = ['val', 'test']

        self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)
        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'BraTS_T1'

            self.csv[state]['img_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['img_path']
            self.csv[state]['mask_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['mask_path']
            self.csv[state]['seg_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['seg_path']

    def setup(self, stage: Optional[str] = None):
        if not hasattr(self, 'val_eval'):
            if self.cfg.sample_set:
                self.val_eval = create_dataset.Eval(self.csv['val'][0:8], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:8], self.cfg)
            else:
                self.val_eval = create_dataset.Eval(self.csv['val'], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'], self.cfg)

    def val_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)


class BraTS_T2(LightningDataModule):
    """BraTS T2 dataset evaluation module for T2 models."""

    def __init__(self, cfg, fold=None):
        super(BraTS_T2, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload', True)
        self.imgpath = {}
        self.csvpath_val = cfg.path.BraTS_T2.IDs.val
        self.csvpath_test = cfg.path.BraTS_T2.IDs.test
        self.csv = {}
        states = ['val', 'test']

        self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)
        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'BraTS_T2'

            self.csv[state]['img_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['img_path']
            self.csv[state]['mask_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['mask_path']
            self.csv[state]['seg_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['seg_path']

    def setup(self, stage: Optional[str] = None):
        if not hasattr(self, 'val_eval'):
            if self.cfg.sample_set:
                self.val_eval = create_dataset.Eval(self.csv['val'][0:8], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:8], self.cfg)
            else:
                self.val_eval = create_dataset.Eval(self.csv['val'], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'], self.cfg)

    def val_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)


class MSLUB(LightningDataModule):

    def __init__(self, cfg, fold= None):
        super(MSLUB, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload',True)
        # load data paths and indices
        self.imgpath = {}
        self.csvpath_val = cfg.path.MSLUB.IDs.val
        self.csvpath_test = cfg.path.MSLUB.IDs.test
        self.csv = {}
        states = ['val','test']

        self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)
        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'MSLUB'

            self.csv[state]['img_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['img_path']
            self.csv[state]['mask_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['mask_path']
            self.csv[state]['seg_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['seg_path']
            
            if cfg.mode != 't1':
                self.csv[state]['img_path'] = self.csv[state]['img_path'].str.replace('uniso/t1',f'uniso/{cfg.mode}').str.replace('t1.nii.gz',f'{cfg.mode}.nii.gz')
    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        if not hasattr(self,'val_eval'):
            if self.cfg.sample_set: # for debugging
                self.val_eval = create_dataset.Eval(self.csv['val'][0:4], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:4], self.cfg)
            else :
                self.val_eval = create_dataset.Eval(self.csv['val'], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'], self.cfg)

    def val_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)


class BraTS_T1_seg(LightningDataModule):
    """BraTS T1 FSL FAST segmented dataset evaluation module for T1 seg models."""

    def __init__(self, cfg, fold=None):
        super(BraTS_T1_seg, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload', True)
        self.imgpath = {}
        self.csvpath_val = cfg.path.BraTS_T1_seg.IDs.val
        self.csvpath_test = cfg.path.BraTS_T1_seg.IDs.test
        self.csv = {}
        states = ['val', 'test']

        self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)
        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'BraTS_T1_seg'

            self.csv[state]['img_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['img_path']
            self.csv[state]['mask_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['mask_path']
            self.csv[state]['seg_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['seg_path']

    def setup(self, stage: Optional[str] = None):
        if not hasattr(self, 'val_eval'):
            if self.cfg.sample_set:
                self.val_eval = create_dataset.Eval(self.csv['val'][0:8], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:8], self.cfg)
            else:
                self.val_eval = create_dataset.Eval(self.csv['val'], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'], self.cfg)

    def val_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)


class BraTS_T2_seg(LightningDataModule):
    """BraTS T2 FSL FAST segmented dataset evaluation module for T2 seg models."""

    def __init__(self, cfg, fold=None):
        super(BraTS_T2_seg, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload', True)
        self.imgpath = {}
        self.csvpath_val = cfg.path.BraTS_T2_seg.IDs.val
        self.csvpath_test = cfg.path.BraTS_T2_seg.IDs.test
        self.csv = {}
        states = ['val', 'test']

        self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)
        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'BraTS_T2_seg'

            self.csv[state]['img_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['img_path']
            self.csv[state]['mask_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['mask_path']
            self.csv[state]['seg_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['seg_path']

    def setup(self, stage: Optional[str] = None):
        if not hasattr(self, 'val_eval'):
            if self.cfg.sample_set:
                self.val_eval = create_dataset.Eval(self.csv['val'][0:8], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:8], self.cfg)
            else:
                self.val_eval = create_dataset.Eval(self.csv['val'], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'], self.cfg)

    def val_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)


class BraTS_T1_FAST(LightningDataModule):
    """BraTS T1 FSL FAST evaluation using H5 files with correct tissue segmentation."""

    def __init__(self, cfg, fold=None):
        super(BraTS_T1_FAST, self).__init__()
        self.cfg = cfg
        self.img_h5_path = cfg.path.BraTS_T1_FAST.img_h5
        self.gt_h5_path = cfg.path.BraTS_T1_FAST.gt_h5

    def setup(self, stage: Optional[str] = None):
        if not hasattr(self, 'val_eval'):
            # Use same data for val and test (all BraTS samples)
            if self.cfg.sample_set:
                self.val_eval = create_dataset.EvalH5(
                    self.img_h5_path, self.gt_h5_path, self.cfg,
                    setname='BraTS_T1_FAST', stage='val'
                )
                self.test_eval = create_dataset.EvalH5(
                    self.img_h5_path, self.gt_h5_path, self.cfg,
                    setname='BraTS_T1_FAST', stage='test'
                )
            else:
                self.val_eval = create_dataset.EvalH5(
                    self.img_h5_path, self.gt_h5_path, self.cfg,
                    setname='BraTS_T1_FAST', stage='val'
                )
                self.test_eval = create_dataset.EvalH5(
                    self.img_h5_path, self.gt_h5_path, self.cfg,
                    setname='BraTS_T1_FAST', stage='test'
                )

    def val_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)


class BraTS_T2_FAST(LightningDataModule):
    """BraTS T2 FSL FAST evaluation using H5 files with correct tissue segmentation."""

    def __init__(self, cfg, fold=None):
        super(BraTS_T2_FAST, self).__init__()
        self.cfg = cfg
        self.img_h5_path = cfg.path.BraTS_T2_FAST.img_h5
        self.gt_h5_path = cfg.path.BraTS_T2_FAST.gt_h5

    def setup(self, stage: Optional[str] = None):
        if not hasattr(self, 'val_eval'):
            if self.cfg.sample_set:
                self.val_eval = create_dataset.EvalH5(
                    self.img_h5_path, self.gt_h5_path, self.cfg,
                    setname='BraTS_T2_FAST', stage='val'
                )
                self.test_eval = create_dataset.EvalH5(
                    self.img_h5_path, self.gt_h5_path, self.cfg,
                    setname='BraTS_T2_FAST', stage='test'
                )
            else:
                self.val_eval = create_dataset.EvalH5(
                    self.img_h5_path, self.gt_h5_path, self.cfg,
                    setname='BraTS_T2_FAST', stage='val'
                )
                self.test_eval = create_dataset.EvalH5(
                    self.img_h5_path, self.gt_h5_path, self.cfg,
                    setname='BraTS_T2_FAST', stage='test'
                )

    def val_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

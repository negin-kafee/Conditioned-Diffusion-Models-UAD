
from torch.utils.data import Dataset
import numpy as np
import torch
import SimpleITK as sitk
import torchio as tio
import h5py
sitk.ProcessObject.SetGlobalDefaultThreader("Platform")
from multiprocessing import Manager

def Train(csv,cfg,preload=True):
    subjects = []
    for _, sub in csv.iterrows():
        subject_dict = {
            'vol' : tio.ScalarImage(sub.img_path,reader=sitk_reader), 
            'age' : sub.age,
            'ID' : sub.img_name,
            'label' : sub.label,
            'Dataset' : sub.setname,
            'stage' : sub.settype,
            'path' : sub.img_path
        }
        if sub.mask_path != None: # if we have masks
            subject_dict['mask'] = tio.LabelMap(sub.mask_path,reader=sitk_reader)
        else: # if we don't have masks, we create a mask from the image
            subject_dict['mask'] = tio.LabelMap(tensor=tio.ScalarImage(sub.img_path,reader=sitk_reader).data>0)

        subject = tio.Subject(subject_dict)
        subjects.append(subject)
    
    if preload: 
        manager = Manager()
        cache = DatasetCache(manager)
        ds = tio.SubjectsDataset(subjects, transform = get_transform(cfg))
        ds = preload_wrapper(ds, cache, augment = get_augment(cfg))
    else: 
        ds = tio.SubjectsDataset(subjects, transform = tio.Compose([get_transform(cfg),get_augment(cfg)]))
        
    if cfg.spatialDims == '2D':
        slice_ind = cfg.get('startslice',None) 
        seq_slices = cfg.get('sequentialslices',None) 
        ds = vol2slice(ds,cfg,slice=slice_ind,seq_slices=seq_slices)
    return ds
 
def Eval(csv,cfg): 
    subjects = []
    for _, sub in csv.iterrows():
        if sub.mask_path is not None and tio.ScalarImage(sub.img_path,reader=sitk_reader).shape != tio.ScalarImage(sub.mask_path,reader=sitk_reader).shape:
            print(f'different shapes of vol and mask detected. Shape vol: {tio.ScalarImage(sub.img_path,reader=sitk_reader).shape}, shape mask: {tio.ScalarImage(sub.mask_path,reader=sitk_reader).shape} \nsamples will be resampled to the same dimension')
            
        subject_dict = {
            'vol' : tio.ScalarImage(sub.img_path,reader=sitk_reader),
            'vol_orig' : tio.ScalarImage(sub.img_path,reader=sitk_reader), # we need the image in original size for evaluation
            'age' : sub.age,
            'ID' : sub.img_name,
            'label' : sub.label,
            'Dataset' : sub.setname,
            'stage' : sub.settype,
            'seg_available': False,
            'path' : sub.img_path }
        if sub.seg_path != None: # if we have segmentations
            subject_dict['seg'] = tio.LabelMap(sub.seg_path,reader=sitk_reader),
            subject_dict['seg_orig'] = tio.LabelMap(sub.seg_path,reader=sitk_reader)# we need the image in original size for evaluation
            subject_dict['seg_available'] = True
        if sub.mask_path != None: # if we have masks
            subject_dict['mask'] = tio.LabelMap(sub.mask_path,reader=sitk_reader)
            subject_dict['mask_orig'] = tio.LabelMap(sub.mask_path,reader=sitk_reader)# we need the image in original size for evaluation
        else: 
            tens=tio.ScalarImage(sub.img_path,reader=sitk_reader).data>0
            subject_dict['mask'] = tio.LabelMap(tensor=tens)
            subject_dict['mask_orig'] = tio.LabelMap(tensor=tens)

        subject = tio.Subject(subject_dict)
        subjects.append(subject)
    ds = tio.SubjectsDataset(subjects, transform = get_transform(cfg))
    return ds
## got it from https://discuss.pytorch.org/t/best-practice-to-cache-the-entire-dataset-during-first-epoch/19608/12
class DatasetCache(object):
    def __init__(self, manager, use_cache=True):
        self.use_cache = use_cache
        self.manager = manager
        self._dict = manager.dict()

    def is_cached(self, key):
        if not self.use_cache:
            return False
        return str(key) in self._dict

    def reset(self):
        self._dict.clear()

    def get(self, key):
        if not self.use_cache:
            raise AttributeError('Data caching is disabled and get funciton is unavailable! Check your config.')
        return self._dict[str(key)]

    def cache(self, key, subject):
        # only store if full data in memory is enabled
        if not self.use_cache:
            return
        # only store if not already cached
        if str(key) in self._dict:
            return
        self._dict[str(key)] = (subject)

class preload_wrapper(Dataset):
    def __init__(self,ds,cache,augment=None):
            self.cache = cache
            self.ds = ds
            self.augment = augment
    def reset_memory(self):
        self.cache.reset()
    def __len__(self):
            return len(self.ds)
            
    def __getitem__(self, index):
        if self.cache.is_cached(index) :
            subject = self.cache.get(index)
        else:
            subject = self.ds.__getitem__(index)
            self.cache.cache(index, subject)
        if self.augment:
            subject = self.augment(subject)
        return subject

class vol2slice(Dataset):
    def __init__(self,ds,cfg,onlyBrain=False,slice=None,seq_slices=None):
            self.ds = ds
            self.onlyBrain = onlyBrain
            self.slice = slice
            self.seq_slices = seq_slices
            self.counter = 0 
            self.ind = None
            self.cfg = cfg

    def __len__(self):
            return len(self.ds)
            
    def __getitem__(self, index):
        subject = self.ds.__getitem__(index)
        if self.onlyBrain:
            start_ind = None
            for i in range(subject['vol'].data.shape[-1]):
                if subject['mask'].data[0,:,:,i].any() and start_ind is None: # only do this once
                    start_ind = i 
                if not subject['mask'].data[0,:,:,i].any() and start_ind is not None: # only do this when start_ind is set
                    stop_ind = i 
            low = start_ind
            high = stop_ind
        else: 
            low = 0
            high = subject['vol'].data.shape[-1]
        if self.slice is not None:
            self.ind = self.slice
            if self.seq_slices is not None:
                low = self.ind
                high = self.ind + self.seq_slices
                self.ind = torch.randint(low,high,size=[1])
        else:
            if self.cfg.get('unique_slice',False): # if all slices in one batch need to be at the same location
                if self.counter % self.cfg.batch_size == 0 or self.ind is None: # only change the index when changing to new batch
                    self.ind = torch.randint(low,high,size=[1])
                self.counter = self.counter +1
            else: 
                self.ind = torch.randint(low,high,size=[1])

        subject['ind'] = self.ind

        subject['vol'].data = subject['vol'].data[...,self.ind]
        subject['mask'].data = subject['mask'].data[...,self.ind]

        return subject


def get_transform(cfg): # only transforms that are applied once before preloading
    h, w, d = tuple(cfg.get('imageDim',(160,192,160)))

    if not cfg.resizedEvaluation: 
        exclude_from_resampling = ['vol_orig','mask_orig','seg_orig']
    else: 
        exclude_from_resampling = None
        
    if cfg.get('unisotropic_sampling',True):
        preprocess = tio.Compose([
        tio.CropOrPad((h,w,d),padding_mode=0),
        tio.RescaleIntensity((0, 1),percentiles=(cfg.get('perc_low',1),cfg.get('perc_high',99)),masking_method='mask'),
        tio.Resample(cfg.get('rescaleFactor',3.0),image_interpolation='bspline',exclude=exclude_from_resampling),#,exclude=['vol_orig','mask_orig','seg_orig']), # we do not want to resize *_orig volumes
        ])

    else: 
        preprocess = tio.Compose([
                tio.RescaleIntensity((0, 1),percentiles=(cfg.get('perc_low',1),cfg.get('perc_high',99)),masking_method='mask'),
                tio.Resample(cfg.get('rescaleFactor',3.0),image_interpolation='bspline',exclude=exclude_from_resampling),#,exclude=['vol_orig','mask_orig','seg_orig']), # we do not want to resize *_orig volumes
            ])


    return preprocess 

def get_augment(cfg): # augmentations that may change every epoch
    augmentations = []

    # individual augmentations
    if cfg.get('random_bias',False):
        augmentations.append(tio.RandomBiasField(p=0.25))
    if cfg.get('random_motion',False):
        augmentations.append(tio.RandomMotion(p=0.1))
    if cfg.get('random_noise',False):
        augmentations.append(tio.RandomNoise(p=0.5))
    if cfg.get('random_ghosting',False):
        augmentations.append(tio.RandomGhosting(p=0.5))
    if cfg.get('random_blur',False):
        augmentations.append(tio.RandomBlur(p=0.5))
    if cfg.get('random_gamma',False):        
        augmentations.append(tio.RandomGamma(p=0.5))
    if cfg.get('random_elastic',False):
        augmentations.append(tio.RandomElasticDeformation(p=0.5))
    if cfg.get('random_affine',False):
        augmentations.append(tio.RandomAffine(p=0.5))
    if cfg.get('random_flip',False):
        augmentations.append(tio.RandomFlip(p=0.5))

    # policies/groups of augmentations
    if cfg.get('aug_intensity',False): # augmentations that change the intensity of the image rather than the geometry
        augmentations.append(tio.RandomGamma(p=0.5))
        augmentations.append(tio.RandomBiasField(p=0.25))
        augmentations.append(tio.RandomBlur(p=0.25))
        augmentations.append(tio.RandomGhosting(p=0.5))

    augment = tio.Compose(augmentations)
    return augment
def sitk_reader(path):
                
    image_nii = sitk.ReadImage(str(path), sitk.sitkFloat32)
    if not 'mask' in str(path) and not 'seg' in str(path) : # only for volumes / scalar images
        image_nii = sitk.CurvatureFlow(image1 = image_nii, timeStep = 0.125, numberOfIterations = 3)
    vol = sitk.GetArrayFromImage(image_nii).transpose(2,1,0)
    return vol, None


class EvalH5(Dataset):
    """Evaluation dataset that loads from H5 files.
    
    Args:
        img_h5_path: Path to H5 file with FSL FAST segmented images
        gt_h5_path: Path to H5 file with tumor ground truth masks
        cfg: Config object with imageDim, rescaleFactor, etc.
        setname: Name of the dataset (e.g., 'BraTS_T1_FAST')
        stage: 'val' or 'test'
    """
    def __init__(self, img_h5_path, gt_h5_path, cfg, setname='BraTS_FAST', stage='test'):
        self.img_h5_path = img_h5_path
        self.gt_h5_path = gt_h5_path
        self.cfg = cfg
        self.setname = setname
        self.stage = stage
        
        # Get keys from H5 file
        with h5py.File(img_h5_path, 'r') as f:
            self.keys = sorted(list(f.keys()))
        
        # Get target dimensions from config
        self.target_shape = tuple(cfg.get('imageDim', (160, 192, 160)))
        self.rescale_factor = cfg.get('rescaleFactor', 3.0)
        
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        
        # Load image and GT from H5
        with h5py.File(self.img_h5_path, 'r') as f:
            img_data = f[key][:]  # Shape: (H, W, D) with values 0,1,2,3
        with h5py.File(self.gt_h5_path, 'r') as f:
            gt_data = f[key][:]  # Shape: (H, W, D) with values 0,1
        
        # Normalize FSL FAST values (0,1,2,3) to (0, 0.33, 0.67, 1.0)
        img_data = img_data.astype(np.float32) / 3.0
        gt_data = gt_data.astype(np.float32)
        
        # Create brain mask from image (non-zero voxels)
        mask_data = (img_data > 0).astype(np.float32)
        
        # Convert to torch tensors with channel dim: (1, H, W, D)
        img_tensor = torch.from_numpy(img_data).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt_data).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_data).unsqueeze(0)
        
        # Create TorchIO subject for transforms
        subject = tio.Subject(
            vol=tio.ScalarImage(tensor=img_tensor),
            vol_orig=tio.ScalarImage(tensor=img_tensor.clone()),
            seg_orig=tio.LabelMap(tensor=gt_tensor),
            mask=tio.LabelMap(tensor=mask_tensor),
            mask_orig=tio.LabelMap(tensor=mask_tensor.clone()),
        )
        
        # Apply transforms (resize, etc.)
        transform = get_transform_h5(self.cfg)
        subject = transform(subject)
        
        # Return dict compatible with existing eval pipeline
        return {
            'vol': subject['vol'],
            'vol_orig': subject['vol_orig'],
            'seg_orig': subject['seg_orig'],
            'mask': subject['mask'],
            'mask_orig': subject['mask_orig'],
            'age': torch.tensor([0]),
            'ID': f'BraTS_{key}',
            'label': torch.tensor([1]),  # All BraTS samples have tumors
            'Dataset': self.setname,
            'stage': self.stage,
            'seg_available': True,
        }


def get_transform_h5(cfg):
    """Transform for H5-based evaluation (no intensity rescaling needed for FSL FAST)."""
    h, w, d = tuple(cfg.get('imageDim', (160, 192, 160)))

    if not cfg.resizedEvaluation:
        exclude_from_resampling = ['vol_orig', 'mask_orig', 'seg_orig']
    else:
        exclude_from_resampling = None

    preprocess = tio.Compose([
        tio.CropOrPad((h, w, d), padding_mode=0),
        tio.Resample(cfg.get('rescaleFactor', 3.0), image_interpolation='nearest', exclude=exclude_from_resampling),
    ])
    
    return preprocess

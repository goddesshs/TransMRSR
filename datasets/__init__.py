from . import mri_dataset
from . import clinic_dataset
from .augmentation import *
adjust_train = (
    Crop(keys=['ref_image_full','ref_image_sub','tag_image_full','tag_image_sub'], crop_size=(256, 256), is_pad_zeros=True, random_crop=False),
    Flip(keys=['ref_image_full','ref_image_sub','tag_image_full','tag_image_sub'], flip_ratio=0.5,
        direction='horizontal'),
    
    Flip(keys=['ref_image_full','ref_image_sub','tag_image_full','tag_image_sub'], flip_ratio=0.5, direction='vertical'),
    RandomTransposeHW(keys=['ref_image_full','ref_image_sub','tag_image_full','tag_image_sub'], transpose_ratio=0.5),
    RescaleToZeroOne(keys=['ref_image_full','ref_image_sub','tag_image_full','tag_image_sub']),
    Normalize(keys=['ref_image_full','ref_image_sub','tag_image_full','tag_image_sub'], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ImageToTensor(keys=['ref_image_full','ref_image_sub','tag_image_full','tag_image_sub'])
)
adjust_test= (
    Crop(keys=['ref_image_full','ref_image_sub','tag_image_full','tag_image_sub'], crop_size=(256, 256), is_pad_zeros=True, random_crop=False),
    RescaleToZeroOne(keys=['ref_image_full','ref_image_sub','tag_image_full','tag_image_sub']),
    Normalize(keys=['ref_image_full','ref_image_sub','tag_image_full','tag_image_sub'], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    
    ImageToTensor(keys=['ref_image_full','ref_image_sub','tag_image_full','tag_image_sub'])
)
def get_datasets(opts):

    if opts.dataset == 'mri':
        trainset = mri_dataset.mriDataset(data_root=opts.data_root, scale=opts.scale, adjust=adjust_train, target_modal=opts.target_modal)
        valset = mri_dataset.mriDataset(data_root=opts.data_root, scale=opts.scale, method='val', adjust=adjust_test, target_modal=opts.target_modal)
        testset = mri_dataset.mriDataset(data_root=opts.data_root, scale=opts.scale, method='test', adjust=adjust_test, target_modal=opts.target_modal)
        
        
    elif opts.dataset == 'real':
        trainset = clinic_dataset.mriDatasetTest(data_root=opts.data_root, scale=opts.scale, adjust=adjust_train, target_modal=opts.target_modal)
        valset = clinic_dataset.mriDatasetTest(data_root=opts.data_root, scale=opts.scale, method='val', adjust=adjust_test, target_modal=opts.target_modal)
        testset = clinic_dataset.mriDatasetTest(data_root=opts.data_root, scale=opts.scale, method='test', adjust=adjust_test, target_modal=opts.target_modal)
    return trainset, valset, testset

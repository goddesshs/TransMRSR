from typing import Collection
import torch.utils.data
import numpy as np, h5py
from torch.utils.data import Dataset
# import torch.utils.Dataset as Dataset
from PIL import Image
import random
from torchvision import transforms
import os
import cv2
from .augmentation import *

# transform = transforms.Compose([
#     # transforms.Resize([256, 256]),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]),
# ])
# import torch
def judgePath(path):
    if not os.path.exists(path):
        return False
    return True

class mriDatasetTest(Dataset):
    def __init__(self, data_root=None, target_modal='t1', method='train', scale=6, adjust=None, size=256):
        # super().__init__()
        super(Dataset).__init__()
        random.seed(123)
        #水平
        # if target_modal == 't1':
        self.t1_lr_folder = data_root
        
        self.imgs = os.listdir(self.t1_lr_folder)

        random.shuffle(self.imgs)
        self.adjust = adjust
        self.size = size
        
    def __len__(self):
        # return len(self.imgs)
        size = len(self.imgs)
        # size = 1
        print('size: ',size)
        return size
    
    def __getitem__(self, index):
        t1_lr_imgs = Image.open(os.path.join(self.t1_lr_folder, self.imgs[index]))
        t2_lr_imgs = Image.open(os.path.join(self.t1_lr_folder, self.imgs[index]))
        t2_hr_imgs = Image.open(os.path.join(self.t1_lr_folder, self.imgs[index]))
        t1_hr_imgs = Image.open(os.path.join(self.t1_lr_folder, self.imgs[index]))
        
        
        t1_lr_imgs = t1_lr_imgs.convert('RGB')
        t2_lr_imgs = t2_lr_imgs.convert('RGB')
        
        t1_hr_imgs = t1_hr_imgs.convert('RGB')
        t2_hr_imgs = t2_lr_imgs.convert('RGB')
        
        t1_lr_imgs = np.array(t1_lr_imgs)
        t1_hr_imgs = np.array(t1_hr_imgs)
        
        t2_lr_imgs = np.array(t2_lr_imgs)
        t2_hr_imgs = np.array(t2_hr_imgs)
        
        results = {'ref_image_full': t2_hr_imgs,
                'ref_image_sub': t2_lr_imgs,
                # 'ref_kspace_full': T1_ks,

                'tag_image_full': t1_hr_imgs,

                'tag_image_sub': t1_lr_imgs,
                'img_name': self.imgs[index]
                # 'tag_image_sub_sub': T2_128_img,
                # 'tag_kspace_sub': T2_128_ks,
                }
        for op in self.adjust:
            results = op(results)
        return results

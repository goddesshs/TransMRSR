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

class mriDataset(Dataset):
    def __init__(self, data_root=None, target_modal='t1', method='train', scale=6, adjust=None, size=256):
        # super().__init__()
        super(Dataset).__init__()
        random.seed(123)
        #水平
        if target_modal == 't1':
            self.t1_lr_folder = os.path.join(data_root, 't1', '2', str(scale), 'sag', method)
            self.t1_hr_folder = os.path.join(data_root, 't1', '2', '1', 'sag', method)
            
            self.t2_lr_folder = os.path.join(data_root, 't2', '2', '1', 'sag', method)
            self.t2_hr_folder = os.path.join(data_root, 't2', '2', '1', 'sag', method)
            
        else:
            self.t1_lr_folder = os.path.join(data_root, 't2', '0', str(scale), 'axial', method)
            self.t1_hr_folder = os.path.join(data_root, 't2', '0', '1', 'axial', method)
            
            self.t2_lr_folder = os.path.join(data_root, 't1', '2', str(scale), 'sag', method)
            self.t2_hr_folder = os.path.join(data_root, 't1', '2', '1', 'sag', method)
        
        self.imgs = []
        # self.imgs = os.listdir(self.t1_lr_folder)
        # print(len(os.listdir(self.t1_lr_folder)))
        for img in os.listdir(self.t1_lr_folder):
            if judgePath(os.path.join(self.t2_lr_folder, img)) and judgePath(os.path.join(self.t2_hr_folder, img)) and judgePath(os.path.join(self.t1_hr_folder, img)):
                self.imgs.append(img)
        # self.t1_imgs= [os.path.join(source_folder, img) for img in os.listdir(source_folder)]
        random.shuffle(self.imgs)
        # if method != 'test':
        #     self.t2_imgs= [os.path.join(t2_folder, img) for img in os.listdir(t2_folder)]
        #     random.shuffle(self.t2_imgs)
        # self.transforms = transform
        self.adjust = adjust
        self.size = size
        
    def __len__(self):
        # return len(self.imgs)
        size = len(self.imgs)
        # size = 1
        print('size: ',size)
        return size
    
    def __getitem__(self, index):
        # print(self.imgs[index])
        # t1_lr_imgs = Image.open(os.path.join(self.t1_lr_folder, self.imgs[index]))
        # t2_lr_imgs = Image.open(os.path.join(self.t2_lr_folder, self.imgs[index]))
        # t2_hr_imgs = Image.open(os.path.join(self.t2_hr_folder, self.imgs[index]))
        # t1_hr_imgs = Image.open(os.path.join(self.t1_hr_folder, self.imgs[index]))
        t1_lr_imgs = Image.open(os.path.join(self.t1_lr_folder, self.imgs[index]))
        t2_lr_imgs = Image.open(os.path.join(self.t2_lr_folder, self.imgs[index]))
        t2_hr_imgs = Image.open(os.path.join(self.t2_hr_folder, self.imgs[index]))
        t1_hr_imgs = Image.open(os.path.join(self.t1_hr_folder, self.imgs[index]))
        
        
        t1_lr_imgs = t1_lr_imgs.convert('RGB')
        t2_lr_imgs = t2_lr_imgs.convert('RGB')
        
        t1_hr_imgs = t1_hr_imgs.convert('RGB')
        t2_hr_imgs = t2_lr_imgs.convert('RGB')
        
        t1_lr_imgs = np.array(t1_lr_imgs)
        t1_hr_imgs = np.array(t1_hr_imgs)
        
        t2_lr_imgs = np.array(t2_lr_imgs)
        t2_hr_imgs = np.array(t2_hr_imgs)
        
        
        # im = Image.fromarray(t1_hr_imgs)
        # # im.save('/lustre/home/acct-seesb/seesb-user1/hs/superResolution/mscmr/1.png')
        # h, w, _ = t1_hr_imgs.shape
        # pad_h, pad_w = (self.size-h)//2, (self.size-w)//2
        # t1_lr_imgs = np.pad(t1_lr_imgs, ((pad_h,  self.size-h-pad_h), (pad_w, self.size-w-pad_w), (0,0)))
        # t1_hr_imgs = np.pad(t1_hr_imgs, ((pad_h,  self.size-h-pad_h), (pad_w, self.size-w-pad_w), (0,0)))
        # im = Image.fromarray(t1_hr_imgs)
        # # im.save('/lustre/home/acct-seesb/seesb-user1/hs/superResolution/mscmr/2.png')
        # t2_lr_imgs = np.pad(t2_lr_imgs, ((pad_h,  self.size-h-pad_h), (pad_w, self.size-w-pad_w), (0,0)))
        # t2_hr_imgs = np.pad(t2_hr_imgs, ((pad_h,  self.size-h-pad_h), (pad_w, self.size-w-pad_w), (0,0)))
        
        # t1_lr_imgs = self.transforms(t1_lr_imgs)
       
        # t1_hr_imgs = self.transforms(t1_hr_imgs)       
        # t2_lr_imgs = self.transforms(t2_lr_imgs)
        # t2_hr_imgs = self.transforms(t2_hr_imgs)
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
        # results = 
        # return {'ref_image_full': t2_hr_imgs,
        #         'ref_image_sub': t2_lr_imgs,
        #         # 'ref_kspace_full': T1_ks,

        #         'tag_image_full': t1_hr_imgs,

        #         'tag_image_sub': t1_lr_imgs,
        #         # 'tag_image_sub_sub': T2_128_img,
        #         # 'tag_kspace_sub': T2_128_ks,
        #         }

# def CreateDatasetSynthesis(phase, input_path, contrast1 = 'T1', contrast2 = 'T2'):
# {}
#     target_file = input_path + "/data_{}_{}.mat".format(phase, contrast1)
#     data_fs_s1=LoadDataSet(target_file)
    
#     target_file = input_path + "/data_{}_{}.mat".format(phase, contrast2)
#     data_fs_s2=LoadDataSet(target_file)

#     dataset=torch.utils.data.TensorDataset(torch.from_numpy(data_fs_s1),torch.from_numpy(data_fs_s2))  
#     return dataset 
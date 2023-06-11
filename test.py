import os
import argparse
import json
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.backends.cudnn as cudnn

from torchvision.utils import save_image
from utils import prepare_sub_folder
from datasets import get_datasets
from models import create_model
import scipy.io as sio
import csv
import pdb
import torch.nn as nn
parser = argparse.ArgumentParser(description='McMRSR')

# model architectures
# model name
parser.add_argument('--experiment_name', type=str, default='your save model name', help='give a experiment name before training')
parser.add_argument('--model_type', type=str, default='McMRSR', help='model type')
parser.add_argument('--resume', type=str, default=None, help='Filename of the checkpoint to resume')

# dataset
# parser.add_argument('--dataset', type=str, default='mri', help='data root folder')

parser.add_argument('--data_root', type=str, default='/lustre/home/acct-seesb/seesb-user1/hs/superResolution/SynDiff-main/dataset/resample1/resample1', help='data root folder')
parser.add_argument('--list_dir', type=str, default='your data list', help='data list_dir root folder')
parser.add_argument('--mask_path', type=str, default='data_demo/dc_mask/lr_4x', help='data list_dir root folder')
parser.add_argument('--dataset', type=str, default='mri', help='dataset name')
parser.add_argument('--img_size', type=int, default=256, help='input image size')
parser.add_argument('--scale', type=int, default=6, help='input image size')
parser.add_argument('--target_modal', type=str, default='t1', help='input image size')
# parser.add_argument('--save_folder', type=str, default='/lustre/home/acct-seesb/seesb-user1/hs/superResolution/SynDiff-main/dataset/resample1/resample1', help='data root folder')

#results
parser.add_argument('--output_path', default='./', type=str, help='Output path.')


# model architectures
parser.add_argument('--net_G', type=str, default='swinIR', help='generator network')
parser.add_argument('--style_dim', type=int, default=512, help='generator network')
parser.add_argument('--fix_decoder', action='store_true', help='pretrained styleSwin')
# parser.add_argument('--sft_half', type=bool, default=True, help='stitch')
parser.add_argument('--pretrained_path', type=str, default=None, help='pretrained styleSwin')
# if opts.model_type 
config = dict(
    in_chans=3,
    out_chans=3,
    chans=32,
    num_pool_layers=4,
    drop_prob=0.0,
    mask_type="random",
    center_fractions=[0.08],
    accelerations=[4],
    n_channels_in=3,
    n_channels_out=3,
    n_resgroups = 6,    
    n_resblocks = 6,    
    n_feats = 64,    
)
parser.set_defaults(**config)
# parser.add_argument('--n_recurrent', type=int, default=1, help='No modification required')
# parser.add_argument('--use_prior', default=False, action='store_true', help='No modification required')
# parser.add_argument('--in_chans', type=int, default=3, help='in channels')

#Swin pama
parser.add_argument('--upscale', type=int, default=4, help='upscale')
parser.add_argument('--window_size', type=int, default=8, help='window_size')
parser.add_argument('--height', type=int, default=256, help='lr_height')
parser.add_argument('--width', type=int, default=256, help='lr_width')
parser.add_argument('--embed_dim', type=int, default=48, help='embed_dim')

parser.add_argument('--truncation_path', type=str, default=None, help='latent space')

parser.add_argument('--anchor_window_down_factor', type=int, default=2, help='embed_dim')

# anchor_window_down_factor
#tgp
parser.add_argument('--sft_half', type=bool, default=True, help='stitch')

# loss options
parser.add_argument('--wr_L1', type=float, default=1, help='weight for reconstruction L1 loss')

# batch size
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')


# optimi

# other
parser.add_argument('--num_workers', type=int, default=0, help='number of threads to load data')
# parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='list of gpu ids')
parser.add_argument('--gpu_ids', type=int, default=[0,1], help='list of gpu ids')
opts = parser.parse_args()

options_str = json.dumps(opts.__dict__, indent=4, sort_keys=False)
print("------------------- Options -------------------")
print(options_str[2:-2])
print("-----------------------------------------------")

cudnn.benchmark = True
# device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    opts.gpu_ids = [0,1]
else:
    opts.gpu_ids = [-1]

model = create_model(opts)
model.setgpu(opts.gpu_ids)

# if opts.gpu_ids[0] != -1:
#     model = model.to('cuda')
#     model = nn.DataParallel(model)


num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)

print('Number of parameters: {} \n'.format(num_param))

if opts.resume is None:
    model.initialize()
    ep0 = -1
    total_iter = 0
else:
    ep0, total_iter = model.resume(opts.resume)

model.set_scheduler(opts, ep0)
ep0 += 1
print('Start training at epoch {} \n'.format(ep0))

# select dataset
train_set, val_set, test_set = get_datasets(opts)
# train_loader = DataLoader(dataset=train_set, num_workers=opts.num_workers, batch_size=opts.batch_size, shuffle=True)
# val_loader = DataLoader(dataset=val_set, num_workers=opts.num_workers, batch_size=1, shuffle=False)
test_loader = DataLoader(dataset=test_set, num_workers=opts.num_workers, batch_size=opts.batch_size, shuffle=False)

# Setup directories
output_directory = os.path.join(opts.output_path, 'outputs', opts.experiment_name)
# checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
eval_directory = os.path.join(output_directory, 'eval_images')
if not os.path.exists(eval_directory):
    print("Creating directory: {}".format(eval_directory))
    os.makedirs(eval_directory)
# evaluation
print('Test Evaluation ......')
model.eval()
with torch.no_grad():
    model.evaluate(test_loader, save_folder=eval_directory)
    
# model = self.results['recon'] 
sio.savemat(os.path.join(eval_directory, 'test_eval.mat'), model.results)

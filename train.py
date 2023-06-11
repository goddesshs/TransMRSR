import os
import argparse
import json
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
print(os.getenv("CUDA_VISIBLE_DEVICES"))
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
import sys
sys.path.append('..')
sys.path.append('.')
import time
torch.backends.cudnn.benchmark = False

print('now: ', time.strftime('%Y-%m-%d %H:%M:%S'))
from utils.utils import complexity, tensor_transform_reverse

parser = argparse.ArgumentParser(description='McMRSR')
# parent_parser = ArgumentParser(add_help=False)
# config=(

# num_gpus = 1
# backend = "ddp"
# batch_size = 4 if backend == "ddp" else num_gpus

# model architectures
# model name
parser.add_argument('--experiment_name', type=str, default='your save model name', help='give a experiment name before training')
parser.add_argument('--model_type', type=str, default='tgp', help='model type')
parser.add_argument('--resume', type=str, default=None, help='Filename of the checkpoint to resume')

# dataset
parser.add_argument('--data_root', type=str, default='/lustre/home/acct-seesb/seesb-user1/hs/superResolution/SynDiff-main/dataset/resample1/resample1', help='data root folder')
parser.add_argument('--list_dir', type=str, default='your data list', help='data list_dir root folder')
parser.add_argument('--mask_path', type=str, default='data_demo/dc_mask/lr_4x', help='data list_dir root folder')
parser.add_argument('--dataset', type=str, default='mri1', help='dataset name')
parser.add_argument('--img_size', type=int, default=256, help='input image size')
parser.add_argument('--scale', type=int, default=6, help='input image size')
parser.add_argument('--target_modal', type=str, default='t1', help='input image size')


# model architectures
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
parser.add_argument('--net_G', type=str, default='swinIR', help='generator network')
parser.add_argument('--style_dim', type=int, default=512, help='generator network')
parser.add_argument('--fix_decoder', action='store_true', help='pretrained styleSwin')
parser.add_argument('--pretrained_path', type=str, default=None, help='pretrained styleSwin')


#Swin pama
parser.add_argument('--upscale', type=int, default=1, help='upscale')
parser.add_argument('--window_size', type=int, default=8, help='window_size')
parser.add_argument('--height', type=int, default=256, help='lr_height')
parser.add_argument('--width', type=int, default=256, help='lr_width')
parser.add_argument('--embed_dim', type=int, default=36, help='embed_dim')
parser.add_argument('--truncation_path', type=str, default=None, help='latent space')

parser.add_argument('--anchor_window_down_factor', type=int, default=2, help='embed_dim')

# anchor_window_down_factor
#tgp
parser.add_argument('--sft_half', type=bool, default=True, help='stitch')


# loss options
parser.add_argument('--wr_L1', type=float, default=1, help='weight for reconstruction L1 loss')

# training options
parser.add_argument('--n_epochs', type=int, default=200, help='number of epoch')
parser.add_argument('--batch_size', type=int, default=4, help='training batch size')

# evaluation options
parser.add_argument('--eval_epochs', type=int, default=4, help='evaluation epochs')
parser.add_argument('--save_epochs', type=int, default=4, help='save evaluation for every number of epochs')

parser.add_argument('--center_fractions', type=float, default=1.0/8.0, help='Cartesian: cernter fraction')
parser.add_argument('--accelerations', type=float, default=5.0, help='Cartesian: acceleration rate')

parser.add_argument('--n_lines', type=float, default=np.round(256 * 0.16), help='Radial: number of radial lines')

parser.add_argument('--n_interleaves', type=float, default=11, help='Spiral: number of interleaves')

# optimizer
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for ADAM')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for ADAM')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')

# learning rate policy
parser.add_argument('--lr_policy', type=str, default='step', help='learning rate decay policy')
parser.add_argument('--step_size', type=int, default=1000, help='step size for step scheduler')
parser.add_argument('--gamma', type=float, default=0.5, help='decay ratio for step scheduler')

# logger options
parser.add_argument('--snapshot_epochs', type=int, default=10, help='save model for every number of epochs')
parser.add_argument('--log_freq', type=int, default=1, help='log loss for every number of epochs')
parser.add_argument('--output_path', default='./', type=str, help='Output path.')

# other
parser.add_argument('--num_workers', type=int, default=0, help='number of threads to load data')
parser.add_argument('--gpu_ids', type=int, default=[0,1], help='list of gpu ids')
opts = parser.parse_args()
print('ooo', opts.fix_decoder)
options_str = json.dumps(opts.__dict__, indent=4, sort_keys=False)
print("------------------- Options -------------------")
print(options_str[2:-2])
print("-----------------------------------------------")

cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# if device=='cpu'
if not torch.cuda.is_available():
    opts.gpu_ids=[-1]
model = create_model(opts)
# complexity(model)
model.setgpu(opts.gpu_ids)

# if opts.gpu_ids[0] != -1:
#     device = torch.device('cuda')
#     model = model.to('cuda')
    # if torch.cuda.device_count():
    #     model = nn.DataParallel(model)
# else:
#     self.device = torch.device('cpu')
# model.cuda()

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
train_loader = DataLoader(dataset=train_set, num_workers=opts.num_workers, batch_size=opts.batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_set, num_workers=opts.num_workers, batch_size=1, shuffle=False)
test_loader = DataLoader(dataset=test_set, num_workers=opts.num_workers, batch_size=1, shuffle=False)

# Setup directories
output_directory = os.path.join(opts.output_path, 'outputs', opts.experiment_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)

with open(os.path.join(output_directory, 'options.json'), 'w') as f:
    f.write(options_str)

with open(os.path.join(output_directory, 'train_loss.csv'), 'w') as f:
    writer = csv.writer(f)
    writer.writerow(model.loss_names)


psnr = 0.0
duration = 0
# training loop
for epoch in range(ep0, opts.n_epochs + 1):

    train_bar = tqdm(train_loader)
    model.train()
    model.set_epoch(epoch)
    for it, data in enumerate(train_bar):
        total_iter += 1
        model.set_input(data)
        model.optimize()
        train_bar.set_description(desc='[Epoch {}]'.format(epoch) + model.loss_summary())

        if it % opts.log_freq == 0:
            with open(os.path.join(output_directory, 'train_loss.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow(model.get_current_losses().values())

        
    model.update_learning_rate()
        
    pred = os.path.join(image_directory, 'pred_{:03d}.png'.format(epoch))
    gt = os.path.join(image_directory, 'gt_{:03d}.png'.format(epoch))
    input_sub = os.path.join(image_directory, 'input_{:03d}.png'.format(epoch))

    # save checkpoint
    if (epoch+1) % opts.snapshot_epochs == 0:
        checkpoint_name = os.path.join(checkpoint_directory, 'model_{}.pt'.format(epoch))
        model.save(checkpoint_name, epoch, total_iter)
        
    tar_hr = tensor_transform_reverse(model.recon)
    save_image(tar_hr, pred, normalize=True, scale_each=True)
    vis_gt = tensor_transform_reverse(model.tag_image_full)
    save_image(vis_gt, gt, normalize=True, scale_each=True)
    
    vis_input = tensor_transform_reverse(model.tag_image_sub)
    save_image(vis_input, input_sub, normalize=True, scale_each=True)

    # evaluation
    print('Validation Evaluation ......')
    if (epoch+1) % opts.eval_epochs == 0:
        
        # if opts.wr_L1 > 0:
            
        model.eval()
        with torch.no_grad():
            model.evaluate(val_loader)

        with open(os.path.join(output_directory, 'metrics.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, model.psnr_recon, model.ssim_recon])

        if model.psnr_recon > psnr+0.05:
            psnr = model.psnr_recon
            checkpoint_name = os.path.join(checkpoint_directory, 'best_{}.pt'.format(epoch))
            model.save(checkpoint_name, epoch, total_iter)
            duration = 0
            
        else:
            duration += 1
            if duration > 10:
                print('bbbb: ', duration)
                break
            
    if (epoch+1) % opts.save_epochs == 0:
        sio.savemat(os.path.join(image_directory, 'eval.mat'), model.results)
ƒ

from collections import OrderedDict
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import sys
sys.path.append('..')
sys.path.append('.')
from networks import get_network
from networks.pro_D import gaussian_weights_init
from models.utils import AverageMeter, get_scheduler, psnr, DataConsistencyInKspace_I, DataConsistencyInKspace_K, fft2_net, complex_abs_eval
from networks.network_tgpmcmr import *
from componets.fastsurfer.set_para import args2cfg
from componets.fastsurfer.dice import criterion
from losses.gan_loss import GANLoss
import cv2
import torch
import itertools
from .basicSR import restorer
from losses.preceptual_loss import PerceptualLoss
_MODELS = {
    "FastSurferCNN": FastSurferCNN,
    "FastSurferVINN": FastSurferVINN,
}


def build_model(cfg):
    assert (cfg.MODEL.MODEL_NAME in _MODELS.keys()),\
        f"Model {cfg.MODEL.MODEL_NAME} not supported"
    params = {k.lower(): v for k, v in dict(cfg.MODEL).items()}
    model = _MODELS[cfg.MODEL.MODEL_NAME](params, padded_size=cfg.DATA.PADDED_SIZE)
    return model

class RecurrentModel(restorer):
    def __init__(self, opts):
        super(RecurrentModel, self).__init__(opts)
        self.opts = opts
        # if self.is_train: 
    def setgpu(self, gpu_ids):
        
        if gpu_ids[0] != -1:
            self.device = torch.device('cuda')

        else:
            self.device = torch.device('cpu')
            
        #content loss 2,4,7,10
        self.contentLoss = PerceptualLoss(layer_weights={'2': 1., '4': 1., '7': 1., },use_input_norm=False,
                 norm_img=False, device=self.device)
        
        self.styleLoss = PerceptualLoss(layer_weights={'1': 1., '3': 1., '5': 1., '9':1., '13': 1.},use_input_norm=False,
                 norm_img=False, device=self.device)
        
    def forward(self, input=None):
        # self.set_require_grad(self.discriminator, require_grad=False)        self.optimizer_G.zero_grad()
        self.tarlr = self.tag_image_sub
        
        out = self.net_G_I(self.tarlr)  # fake hr
        self.recon = out
    
    def fft2(self, data):
        if data.ndim == 4:
            data = torch.mean(data, dim=1, keepdim=False)
        x_uncentered = torch.fft.ifftshift(data)
        fft_uncentered = torch.fft.fft2(x_uncentered)
        fft_center = torch.fft.fftshift(fft_uncentered)
        amplitute = 10*torch.log(torch.abs(fft_center)) 
        return amplitute 
    
    def update_G(self):
        self.optimizer_G.zero_grad()
        targe_hr = self.recon
        self.loss = self.criterion(targe_hr, self.tag_image_full)
        self.content_loss, _ = self.contentLoss(targe_hr, self.tag_image_full, perceptual_weight=1.0, style_weight=0)
        _, self.style_loss = self.styleLoss(targe_hr, self.tag_image_full, perceptual_weight=0, style_weight=1.0)
        # self.kspac = self.mse(self.fft2(targe_hr), self.fft2(self.tag_image_full))
        self.kspac = torch.tensor(0)
        # self.style_loss = self.criterion(targe_hr, self.ref_image_full, perceptual_weight=0.5, style_weight=1.0)
        self.total_loss = self.loss + (self.content_loss + self.style_loss)*0.5
        
        
        self.total_loss.backward()
        self.optimizer_G.step()
    
    
    
    def loss_summary(self):
        message = ''
        if self.opts.wr_L1 > 0:
            message += 'G_L1: {:.4f} perp_loss: {:.4f} content_loss: {:.4f} kspac_loss: {:4f} total_loss: {:.4f}'.format(self.loss.item(), self.content_loss.item(), self.style_loss.item(), self.kspac.item(), self.total_loss.item())

        return message
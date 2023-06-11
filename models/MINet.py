
from collections import OrderedDict
import os
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import sys
from networks import get_network
from networks.pro_D import gaussian_weights_init
from models.utils import AverageMeter, get_scheduler, psnr, tensor_transform_reverse
from PIL import Image
from losses.preceptual_loss import PerceptualLoss

from .basicSR import restorer
class RecurrentModel(restorer):
    def __init__(self, opts):

        super(RecurrentModel, self).__init__(opts)
        self.opts = opts

    def forward(self):
        I = self.tag_image_sub
        I.requires_grad_(True)
        I_T1 = self.ref_image_sub
        I_T1.requires_grad_(True)
        T1 = self.ref_image_full
        T1.requires_grad_(True)

        net = {}
        tarlr = I
        reflr = I_T1
        ref = T1
        out = self.net_G_I(ref, tarlr)  # output recon image [b,c,w,h]
        self.recon = out

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
        
    def update_G(self):
        self.optimizer_G.zero_grad()
        targe_hr = self.recon
        self.loss = self.criterion(targe_hr, self.tag_image_full)
        self.content_loss, _ = self.contentLoss(targe_hr, self.tag_image_full, perceptual_weight=1.0, style_weight=0)
        _, self.style_loss = self.styleLoss(targe_hr, self.tag_image_full, perceptual_weight=0, style_weight=1.0)
        # self.style_loss = self.criterion(targe_hr, self.ref_image_full, perceptual_weight=0.5, style_weight=1.0)
        self.total_loss = self.loss + (self.content_loss + self.style_loss)*0.5
        self.total_loss.backward()
        self.optimizer_G.step()
        
        
    def loss_summary(self):
        message = ''
        if self.opts.wr_L1 > 0:
            message += 'G_L1: {:.4f} perp_loss: {:.4f} content_loss: {:.4f} kspac_loss: {:4f} total_loss: {:.4f}'.format(self.loss.item(), self.content_loss.item(), self.style_loss.item(), self.kspac.item(), self.total_loss.item())

        return message

   

    # def loss_summary(self):
    #     message = ''
    #     if self.opts.wr_L1 > 0:
    #         message += 'G_L1: {:.4f} Img_L1: {:.4f}'.format(self.loss_G_L1, self.loss_img_l1)

    #     return message

    # def update_learning_rate(self):
    #     for scheduler in self.schedulers:
    #         scheduler.step()
    #     lr = self.optimizers[0].param_groups[0]['lr']
    #     print('learning rate = {:7f}'.format(lr))

    # def save(self, filename, epoch, total_iter):

    #     state = {}
    #     if self.opts.wr_L1 > 0:
    #         state['net_G_I'] = self.net_G_I.state_dict()
    #         state['opt_G'] = self.optimizer_G.state_dict()

    #     state['epoch'] = epoch
    #     state['total_iter'] = total_iter

    #     torch.save(state, filename)
    #     print('Saved {}'.format(filename))

    # def resume(self, checkpoint_file, train=True):
    #     if self.opts.gpu_ids[0] == -1:
    #         checkpoint = torch.load(checkpoint_file, map_location='cpu')
    #     else:
    #         checkpoint = torch.load(checkpoint_file)

    #     # if self.opts.wr_L1 > 0:
    #     weight = {k[7:]:v for k, v in checkpoint['net_G_I'].items()}
            
    #     self.net_G_I.load_state_dict(weight)
    #     if self.is_train:
    #         self.optimizer_G.load_state_dict(checkpoint['opt_G'])

    #     print('Loaded {}'.format(checkpoint_file))

    #     return checkpoint['epoch'], checkpoint['total_iter']

    # def evaluate(self, loader, save_folder=None):
    #     val_bar = tqdm(loader)
    #     avg_psnr = AverageMeter()
    #     avg_ssim = AverageMeter()

    #     recon_images = []
    #     gt_images = []
    #     input_images = []
        

    #     j = 1
    #     for data in val_bar:
    #         self.set_input(data)
    #         self.forward()
    #         _, self.rec = self.recon
    #         self.recs = tensor_transform_reverse(self.rec)
    #         self.gts = tensor_transform_reverse(self.tag_image_full)
    #         self.lr_imags = tensor_transform_reverse(self.tag_image_sub)
    #         self.results = {}
    #         self.recs = self.recs.mul(255).add_(0.5).clamp_(0, 255)
    #         self.gts = self.gts.mul(255).add_(0.5).clamp_(0, 255)
    #         self.lr_imags = self.lr_imags.mul(255).add_(0.5).clamp_(0, 255)
    #         if len(data['tag_image_sub'].shape)==4:
    #             for i in range(data['tag_image_sub'].shape[0]):
    #                 self.rec = self.recs[i]
    #                 self.gt = self.gts[i]
    #                 self.lr_img = self.lr_imags[i]
    #                 ############# back to [0,1] #############
    #                 if self.opts.wr_L1 > 0:
    #                     psnr_recon = psnr(self.rec, self.gt)
    #                     avg_psnr.update(psnr_recon)
    #                     r = self.rec.squeeze().cpu().numpy()
    #                     ssim_recon = ssim(self.rec.squeeze().detach().cpu().numpy(), self.gt.squeeze().detach().cpu().numpy(), channel_axis=0)

    #                     avg_ssim.update(ssim_recon)

    #                     recon_images.append(self.rec.cpu())
    #                     gt_images.append(self.gt.cpu())
    #                     # input_images.append(self.lr_imag.cpu())
    #                     if save_folder:
    #                         ndarr = self.rec.permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    #                         im = Image.fromarray(ndarr)
    #                         im.save(os.path.join(save_folder, 'predict_'+str(j)+'.jpg'))
                            
    #                         ndarr = self.gt.permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    #                         im = Image.fromarray(ndarr)
    #                         im.save(os.path.join(save_folder, 'gt_'+str(j)+'.jpg'))
                            
    #                         ndarr = self.lr_img.permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    #                         im = Image.fromarray(ndarr)
    #                         im.save(os.path.join(save_folder, 'lr_'+str(j)+'.jpg'))
    #                         j += 1
    #         message = 'PSNR: {:.4f} '.format(avg_psnr.avg)
    #         message += 'SSIM: {:.4f} '.format(avg_ssim.avg)
    #         val_bar.set_description(desc=message)
        
    #     print('PSNR: {:.4f}'.format(avg_psnr.avg))
    #     print('SSIM: {:.4f}'.format(avg_ssim.avg))
    #     # message = 'PSNR: {:4f} '.format(avg_psnr.avg)
    #     #     message += 'SSIM: {:4f} '.format(avg_ssim.avg)
    #     self.psnr_recon = avg_psnr.avg
    #     self.ssim_recon = avg_ssim.avg

        
        

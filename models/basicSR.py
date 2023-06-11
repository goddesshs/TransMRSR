
from collections import OrderedDict
import os
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
# from skimage.metrics import structural_similarity as ssim
import sys
from networks import get_network
from networks.pro_D import gaussian_weights_init
from models.utils import AverageMeter, get_scheduler, psnr, tensor_transform_reverse, ssim, rmse
from PIL import Image
import time
print('now: ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

class restorer(nn.Module):
    def __init__(self, opts=None):
        super(restorer, self).__init__()
        if opts != None:
            self.loss_names = []
            self.networks = []
            self.optimizers = []

            # self.n_recurrent = opts.n_recurrent
            self.upscale = opts.upscale


            # set default loss flags
            loss_flags = ("w_img_L1")
            for flag in loss_flags:
                if not hasattr(opts, flag): setattr(opts, flag, 0)

            self.is_train = True if hasattr(opts, 'lr') else False

            self.net_G_I = get_network(opts)
            self.networks.append(self.net_G_I)

            if self.is_train:
                self.loss_names += ['total_loss']
                param = [p for p in self.net_G_I.parameters() if p.requires_grad]

                self.optimizer_G = torch.optim.Adam(param,
                                                    lr=opts.lr,
                                                    betas=(opts.beta1, opts.beta2),
                                                    weight_decay=opts.weight_decay)
                self.optimizers.append(self.optimizer_G)

            self.criterion = nn.L1Loss()
            self.mse = nn.MSELoss()

            self.opts = opts

    def setgpu(self, gpu_ids):
        
        if gpu_ids[0] != -1:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def initialize(self):
        [net.apply(gaussian_weights_init) for net in self.networks]

    def set_scheduler(self, opts, epoch=-1):
        self.schedulers = [get_scheduler(optimizer, opts, last_epoch=epoch) for optimizer in self.optimizers]

    def set_input(self, data):
        #[b,2,w,h]
        self.ref_image_full = data['ref_image_full'].to(self.device)
        self.ref_image_sub = data['ref_image_sub'].to(self.device)
        # self.tag_kspace_full = data['tag_kspace_full'].to(self.device)
        self.tag_image_full = data['tag_image_full'].to(self.device)
        self.tag_image_sub = data['tag_image_sub'].to(self.device)
        self.img_names = data['img_name']
        
        # self.tag_kspace_mask2d = data['tag_kspace_mask2d'].to(self.device)

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, name))
        return errors_ret

    def set_epoch(self, epoch):
        self.curr_epoch = epoch

    def forward(self, input=None):
        I = self.tag_image_sub
        I.requires_grad_(True)
        I_T1 = self.ref_image_sub
        I_T1.requires_grad_(True)
        T1 = self.ref_image_full
        T1.requires_grad_(True)


        net = {}
        # for i in range(1, self.n_recurrent + 1):
        tarlr = I
        reflr = I_T1
        ref = T1
        # if 
        out = self.net_G_I(tarlr)  # output recon image [b,c,w,h]
        self.recon = out

    def update_G(self):
        self.optimizer_G.zero_grad()
        targe_hr = self.recon
        loss = self.criterion(targe_hr, self.tag_image_full)
        self.loss_G_L1 = loss
        self.total_loss = loss
        self.total_loss.backward()
        self.optimizer_G.step()

    def optimize(self):

        self.forward()
        self.update_G()

    def loss_summary(self):
        message = ''
        if self.opts.wr_L1 > 0:
            message += 'G_L1: {:.4f}'.format(self.total_loss)

        return message

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = {:7f}'.format(lr))

    def save(self, filename, epoch, total_iter):

        state = {}
        if self.opts.wr_L1 > 0:
            state['net_G_I'] = self.net_G_I.state_dict()
            state['opt_G'] = self.optimizer_G.state_dict()

        state['epoch'] = epoch
        state['total_iter'] = total_iter

        torch.save(state, filename)
        print('Saved {}'.format(filename))

    # def resume(self, checkpoint_file, train=True):
    def resume(self, checkpoint_file, train=True):
        if self.opts.gpu_ids[0] == -1:
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
        else:
            checkpoint = torch.load(checkpoint_file)
        net_dict = self.net_G_I.state_dict()
        weight = dict()
        # if self.opts.wr_L1 > 0:
        # weight = {k[7:]:v for k, v in checkpoint['net_G_I'].items() if 'module' in k}
        for k, v in net_dict.items():
            print('net: ', k)
        for k, v in checkpoint['net_G_I'].items():
            if 'module' in k:
                k = k[7:]
            key = k
            try:
                m = net_dict[k].numel()
            except:
                key = 'module.' + k
            print('kkkk: ', key)
            if net_dict[key].numel() == v.numel():
                print('eeeeekkkk: ', key)
                
                weight[key] = v
        
        self.net_G_I.load_state_dict(weight)
        if self.is_train:
            self.optimizer_G.load_state_dict(checkpoint['opt_G'])

        print('Loaded {}'.format(checkpoint_file))

        return checkpoint['epoch'], checkpoint['total_iter']

    def evaluate(self, loader, save_folder=None):
        val_bar = tqdm(loader)
        avg_psnr = AverageMeter()
        avg_ssim = AverageMeter()
        avg_rmse = AverageMeter()

        recon_images = []
        gt_images = []
        input_images = []
        

        j = 1
        for data in val_bar:
            self.set_input(data)
            self.forward()
            self.rec = self.recon
            self.recs = tensor_transform_reverse(self.rec)
            self.gts = tensor_transform_reverse(self.tag_image_full)
            self.lr_imags = tensor_transform_reverse(self.tag_image_sub)
            self.results = {}
            self.recs = self.recs.mul(255).add_(0.5).clamp_(0, 255)
            self.gts = self.gts.mul(255).add_(0.5).clamp_(0, 255)
            self.lr_imags = self.lr_imags.mul(255).add_(0.5).clamp_(0, 255)
            if len(data['tag_image_sub'].shape)==4:
                for i in range(data['tag_image_sub'].shape[0]):
                    self.img_name = self.img_names[i]
                    self.rec = self.recs[i]
                    self.gt = self.gts[i]
                    self.lr_img = self.lr_imags[i]
                    ############# back to [0,1] #############
                    if self.opts.wr_L1 > 0:
                        psnr_recon = psnr(self.rec, self.gt)
                        if psnr_recon!=0:
                            avg_psnr.update(psnr_recon)
                        r = self.rec.squeeze().cpu().numpy()
                        # ssim_recon = ssim(self.rec.squeeze().detach().cpu().numpy(), self.gt.squeeze().detach().cpu().numpy(), channel_axis=0)
                        #mmcv ssim
                        ssim_recon = ssim(self.rec.squeeze().detach().cpu().numpy(), self.gt.squeeze().detach().cpu().numpy(), input_order='CHW')

                        avg_ssim.update(ssim_recon)

                        rmse_rcon = rmse(self.rec.squeeze().detach().cpu().numpy(), self.gt.squeeze().detach().cpu().numpy())
                        avg_rmse.update(rmse_rcon)
                        recon_images.append(self.rec.cpu())
                        gt_images.append(self.gt.cpu())
                        # input_images.append(self.lr_imag.cpu())
                        if save_folder:

                            ndarr = self.rec.permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                            im = Image.fromarray(ndarr)
                            im.save(os.path.join(save_folder, 'predict_'+self.img_name))
                            
                            ndarr = self.gt.permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                            im = Image.fromarray(ndarr)
                            im.save(os.path.join(save_folder, 'gt_'+self.img_name))
                            
                            ndarr = self.lr_img.permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                            im = Image.fromarray(ndarr)
                            im.save(os.path.join(save_folder, 'lr_'+self.img_name))
                            j += 1
            message = 'PSNR: {:.4f} '.format(avg_psnr.avg)
            message += 'SSIM: {:.4f} '.format(avg_ssim.avg)
            message += 'RMSE: {:.4f} '.format(avg_rmse.avg)
            
            val_bar.set_description(desc=message)
        
        print('PSNR_all: {:.4f}'.format(avg_psnr.avg))
        print('SSIM_all: {:.4f}'.format(avg_ssim.avg))
        print('RMSE_all: {:.4f}'.format(avg_ssim.avg))
        
        # message = 'PSNR: {:4f} '.format(avg_psnr.avg)
        #     message += 'SSIM: {:4f} '.format(avg_ssim.avg)
        self.psnr_recon = avg_psnr.avg
        self.ssim_recon = avg_ssim.avg

        
        

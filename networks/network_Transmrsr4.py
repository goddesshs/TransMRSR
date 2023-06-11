
import os
import sys
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import functools
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from componets import *
from mmcv.ops import FusedBiasLeakyReLU

from mmcv.ops import upfirdn2d

from torch.nn import functional as F
from torch.nn.utils import spectral_norm
import componets.fastsurfer.sub_module as sm
# import FastSurferCNN.models.sub_module as sm
import componets.fastsurfer.interpolation_layer as il
# from componets.swim import *

# class FAB
#     R"""
#         Feature extraction:,没有SC
#******没有hr 
#     """
class TGP4(nn.Module):
    r""" 

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 2
        embed_dim (int): Patch embedding dimension. Default: 60
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 8
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bo=512ol): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/4 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
    """

    def __init__(self,
        size,
        style_dim=512,
        in_chans=3,
        patch_size=1,
        channel_multiplier=1,
        lr_mlp=0.01,
        enable_full_resolution=8,
        mlp_ratio=4,
        use_checkpoint=False,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0,
        attn_drop_rate=0,
        num_center=8,
        norm_layer=nn.LayerNorm,
        unet_narrow = 1,
        drop_path_rate=0.1,
        patch_norm=True,
        truncation = 0.9,
        truncation_path=None,
        ape = False, 
        img_range=1., sft_half = False, fix_decoder=True, pretrained_path=None):
        
        
        super(TGP4, self).__init__()
        num_in_ch = in_chans
        start = 2
        depths = [2, 2, 2, 2, 2, 2, 2, 2, 2]
        channels = {
            '4': int(512 * unet_narrow),
            '8': int(512 * unet_narrow),
            '16': int(512 * unet_narrow),
            '32': int(512 * unet_narrow),
            '64': int(256 * channel_multiplier * unet_narrow),
            '128': int(128 * channel_multiplier * unet_narrow),
            '256': int(64 * channel_multiplier * unet_narrow),
            '512': int(32 * channel_multiplier * unet_narrow),
            '1024': int(16 * channel_multiplier * unet_narrow)
        }
        
        print('fff', fix_decoder)
        embed_dim = channels[f'{size}']
        self.size = size
        self.style_dim = style_dim
        self.patch_norm = patch_norm
        self.ape = ape
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        # self.upscale = upscale
        # self.window_size = window_size
        n_blks = [2, 2, 2]
        n_blks_dec = [2, 2, 2, 12, 8, 4]
        self.ref_down_block_size = 1.5
        #scale
        self.dilations = [1,2,3]
        self.num_nbr = 1
        self.psize = 3

        self.MAB = MAB(embed_dim, num_in_ch, n_blks=n_blks_dec, upscale=1)

        #####################################################################################################
        ################################### 1, Tar/Ref LR feature extraction ###################################
        self.conv2d = Conv2D(in_chl=num_in_ch, nf=embed_dim, n_blks=n_blks)

        #####################################################################################################
        ################################### 2, deep feature extraction (STG) ######################################
        self.start=2
        # self.log_size = 
        self.end = int(math.log(size, 2))
        self.mlp_ratio = mlp_ratio
        num_heads = [max(int(c) // 32, 4) for c in channels.values()]
        full_resolution_index = int(math.log(enable_full_resolution, 2))
        window_sizes = [2 ** i if i <= full_resolution_index else 8 for i in range(self.end, self.start, -1)]
        self.lr_block_sizes = [2 ** (i-2) for i in range(self.end, self.start-1, -1)]
        #split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        # build Residual Swin Transformer blocks (RSTB)
        self.fusion_convs = nn.ModuleList()
        
        in_channels = channels[f'{size}']
        self.layers = nn.ModuleList()
        self.num_layers = 0
        for i_layer in range(self.end, self.start, -1):
            # size = 2**(i_layer-1)
            out_channels = channels[f'{2**(i_layer-1)}']
            layer = RSTB(dim=in_channels,
                         input_resolution=(2** i_layer,
                                           2**i_layer),
                         depth=depths[i_layer],
                         num_heads=num_heads[self.end-i_layer],
                         out_dim=out_channels,
                         window_size=window_sizes[self.end-i_layer],
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate, # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=PatchMerging,
                         use_checkpoint=use_checkpoint,
                         img_size=size,
                         patch_size=patch_size,
                         )
            in_channels = out_channels
            self.layers.append(layer)
            self.fusion_convs.append(EqualConv2d(out_channels*2, out_channels, kernel_size=3, stride=1, padding=1, bias=True))
            self.num_layers += 2
        
        # self.final_fusion = EqualConv2d(out_channels*2, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.norm = norm_layer(in_channels)
        self.num_layers += 2
        self.final_layers = EqualLinear(channels['4']*4*4, style_dim*self.num_layers, bias=True, lr_mul=lr_mlp, activation='fused_lrelu')
        #####################################################################################################
        ################################### 3, Style Swin ######################################
        self.sft_half = sft_half
        self.num_center = num_center
        
        self.truncation_path = truncation_path
        self.truncation = truncation
        centers = None
        self.centers = None
        centers = None
        print('ttt: ', self.truncation_path)
        if self.truncation_path is not None:
            for file in os.listdir(truncation_path):
                if 'npy' in file:
                    center = np.load(os.path.join(truncation_path, file))
                    center = center[None, :]
                    if centers is None:
                        centers = center
                    else:
                        centers = np.concatenate((centers, center), axis=0)
        # self.centers = torch.Tensor(centers) if centers is not 
            self.centers = torch.Tensor(centers)
        
        self.condition_scale = nn.ModuleList()
        self.condition_shift = nn.ModuleList()
        for i in range(self.end-1, self.start-1, -1):
            out_channels = channels[f'{2**i}']
            if sft_half:
                sft_out_channels = out_channels//2
            else:
                sft_out_channels = out_channels 
            self.condition_scale.append(
                (
                    nn.Sequential(
                        EqualConv2d(out_channels, out_channel=out_channels, kernel_size=3, stride=1, padding=1, bias=True, bias_init_val=0),
                        ScaledLeakyReLU(0.2),
                        EqualConv2d(out_channels, sft_out_channels, kernel_size=3, stride=1, padding=1, bias=True, bias_init_val=0),               
                    )
                )
            )
            
            self.condition_shift.append(
                (
                    nn.Sequential(
                        EqualConv2d(out_channels, out_channel=out_channels, kernel_size=3, stride=1, padding=1, bias=True, bias_init_val=0),
                        ScaledLeakyReLU(0.2),
                        EqualConv2d(out_channels, sft_out_channels, kernel_size=3, stride=1, padding=1, bias=True, bias_init_val=0),
                    )
                )
            )
        self.apply(self._init_weights)
        
        self.generator = Generator1(size, style_dim, channel_multiplier=channel_multiplier, lr_mlp=lr_mlp, enable_full_resolution=enable_full_resolution, sft_half=sft_half)
        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path, map_location='cpu')['g_ema']
            self.generator.load_state_dict(state_dict, strict=False)
        
        if fix_decoder:
            for param in self.generator.parameters():
                param.requires_grad = False
        ################################### 4, deep feature ######################################
        self.conv_deep =  Conv2D(embed_dim, embed_dim, n_blks=n_blks)
        self.conv_last = nn.Conv2d(embed_dim, 3, 1,1)
        
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
            
    def get_nearest(self, latent):
            latent = latent.unsqueeze(2).repeat(1,1,self.num_center,1)
            dist = torch.sum(torch.square(latent-self.centers), dim=-1, keepdim=False)
            idx = torch.argmax(dist,dim=-1)
            b, num_latents = idx.shape
            idx = idx.view(-1).type(torch.LongTensor)
            nearest_center = self.centers[idx]
            nearest_center = nearest_center.view(b, num_latents, -1, self.style_dim).squeeze()
            return nearest_center
    
    def forward(self, tar):
        
        #### tar_lr #####
        #### Conv2D #####
        tar_lr = self.conv2d(tar)
        
        x_size = (tar_lr.shape[2], tar_lr.shape[3])
        tar_lr = self.patch_embed(tar_lr)
        if self.ape:
            tar_lr = tar_lr + self.absolute_pos_embed
            
        tar_lr = self.pos_drop(tar_lr)
        
        conditions = []
        #feature merging
        #256, 128, 64
        for i_layer in range(self.end-2):
            #set block size
            layer = self.layers[i_layer]
            
            tar_lr, out_size = layer(tar_lr)
            b = tar_lr.size()[0]
            tar_lr = tar_lr.view(b, out_size[0], out_size[1], -1).permute(0,3,1,2).contiguous()#B,H,W,C->B,C,H,W

            #
            shift = self.condition_shift[i_layer](tar_lr)#B,C,H,W
            conditions.insert(0,shift)
            
            scale = self.condition_scale[i_layer](tar_lr) #B,C,H,W
            conditions.insert(0, scale)
            tar_lr = tar_lr.flatten(2).transpose(1,2)
            
            # tar
        #B,4*4,C
        tar_lr = self.norm(tar_lr)
        #B,L,C->B,C,H,W
        style = tar_lr.view(tar_lr.shape[0], -1)
        
        
        latent = self.final_layers(style)
        latent = latent.view(latent.size(0), -1, self.style_dim)
        
        if self.truncation_path is not None:
        
            self.centers = self.centers.to(tar_lr.device)
            nearest_latents = self.get_nearest(latent)
            
            latent = nearest_latents + self.truncation * (latent - nearest_latents)
        
        x = self.generator.input(latent)
        count = 0
        for layer in self.generator.layers:

            if count < len(conditions):
                x = layer(x, latent[:,count,:], latent[:,count+1,:], conditions[count], conditions[count+1], sft_half=self.sft_half)
            else:
                x = layer(x, latent[:,count,:], latent[:,count+1,:], sft_half=self.sft_half)
            
            if x.dim() == 3:
                b, n, c = x.shape
                h, w = int(math.sqrt(n)), int(math.sqrt(n))
                x = x.transpose(-1, -2).reshape(b, c, h, w)
                
            count += 2
               
        out = self.conv_deep(x)
        out = self.conv_last(out)
    
        return out

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops



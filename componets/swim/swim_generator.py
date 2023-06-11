# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math

import torch
import torch.utils.checkpoint as checkpoint
from timm.models.layers import to_2tuple, trunc_normal_
from torch import nn
# from mmcv.ops import FusedBiasLeakyReLU
from .swim import *


class StyleSwinTransformerBlock(nn.Module):
    r""" StyleSwin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        style_dim (int): Dimension of style vector.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, style_dim=512):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.shift_size = self.window_size // 2
        self.style_dim = style_dim
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = AdaptiveInstanceNorm(dim, style_dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn = nn.ModuleList([
            WindowAttentionDual(
                dim // 2, window_size=to_2tuple(self.window_size), num_heads=num_heads // 2,
                qk_scale=qk_scale, attn_drop=attn_drop),
            WindowAttentionDual(
                dim // 2, window_size=to_2tuple(self.window_size), num_heads=num_heads // 2,
                qk_scale=qk_scale, attn_drop=attn_drop),
        ])
        
        attn_mask1 = None
        attn_mask2 = None
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1,
                                            self.window_size * self.window_size)
            attn_mask2 = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask2 = attn_mask2.masked_fill(
                attn_mask2 != 0, float(-100.0)).masked_fill(attn_mask2 == 0, float(0.0))
        
        self.register_buffer("attn_mask1", attn_mask1)
        self.register_buffer("attn_mask2", attn_mask2)

        self.norm2 = AdaptiveInstanceNorm(dim, style_dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, style):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        # Double Attn
        shortcut = x
        x = self.norm1(x.transpose(-1, -2), style).transpose(-1, -2)
        
        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3).reshape(3 * B, H, W, C)
        qkv_1 = qkv[:, :, :, : C // 2].reshape(3, B, H, W, C // 2)
        if self.shift_size > 0:
            qkv_2 = torch.roll(qkv[:, :, :, C // 2:], shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)).reshape(3, B, H, W, C // 2)
        else:
            qkv_2 = qkv[:, :, :, C // 2:].reshape(3, B, H, W, C // 2)
        
        q1_windows, k1_windows, v1_windows = self.get_window_qkv(qkv_1)
        q2_windows, k2_windows, v2_windows = self.get_window_qkv(qkv_2)

        x1 = self.attn[0](q1_windows, k1_windows, v1_windows, self.attn_mask1)
        x2 = self.attn[1](q2_windows, k2_windows, v2_windows, self.attn_mask2)
        
        x1 = window_reverse(x1.view(-1, self.window_size * self.window_size, C // 2), self.window_size, H, W)
        x2 = window_reverse(x2.view(-1, self.window_size * self.window_size, C // 2), self.window_size, H, W)

        if self.shift_size > 0:
            x2 = torch.roll(x2, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x2 = x2

        x = torch.cat([x1.reshape(B, H * W, C // 2), x2.reshape(B, H * W, C // 2)], dim=2)
        x = self.proj(x)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x.transpose(-1, -2), style).transpose(-1, -2))

        return x
    
    def get_window_qkv(self, qkv):
        q, k, v = qkv[0], qkv[1], qkv[2]   # B, H, W, C
        C = q.shape[-1]
        q_windows = window_partition(q, self.window_size).view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        k_windows = window_partition(k, self.window_size).view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        v_windows = window_partition(v, self.window_size).view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        return q_windows, k_windows, v_windows

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += 1 * self.style_dim * self.dim * 2
        flops += 2 * (H * W) * self.dim
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        for attn in self.attn:
            flops += nW * (attn.flops(self.window_size * self.window_size))
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += 1 * self.style_dim * self.dim * 2
        flops += 2 * (H * W) * self.dim
        return flops


class StyleBasicLayer(nn.Module):
    """ A basic StyleSwin layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        out_dim (int): Number of output channels.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        upsample (nn.Module | None, optional): Upsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        style_dim (int): Dimension of style vector.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, out_dim=None,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., upsample=None, 
                 use_checkpoint=False, style_dim=512):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            StyleSwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop, style_dim=style_dim)
            for _ in range(depth)])

        if upsample is not None:
            self.upsample = upsample(input_resolution, dim=dim, out_dim=out_dim)
        else:
            self.upsample = None

    def forward(self, x, latent1, latent2, scale=None, shift=None, sft_half=False):
        B, C, H, W = x.shape
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.blocks[0], x, latent1)
            x = checkpoint.checkpoint(self.blocks[1], x, latent2)
        else:
            # print('xxx: ', x.shape)
            
            x = x.permute(0, 2, 3, 1).contiguous().view(B, H*W, C)
            x = self.blocks[0](x, latent1)
            
            # 
            # if sft_half
            
            # x = x.view(x.size()[0], self.input_resolution[0], self.input_resolution[1], -1).permute(0,3,1,2).contiguous()
            # x = x*scale+shift
            # x = x.flatten(2).transpose(0,2,1)
            x = self.blocks[1](x, latent2)

        if self.upsample is not None:
            x = self.upsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.upsample is not None:
            flops += self.upsample.flops()
        return flops


class BilinearUpsample(nn.Module):
    """ BilinearUpsample Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        out_dim (int): Number of output channels.
    """

    def __init__(self, input_resolution, dim, out_dim=None):
        print('inpur: ', input_resolution)
        print('dim: ', dim)
        print('out dim: ', out_dim)
        super().__init__()
        assert dim % 2 == 0, f"x dim are not even."
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.norm = nn.LayerNorm(dim)
        self.reduction = nn.Linear(dim, out_dim, bias=False)
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.alpha = nn.Parameter(torch.zeros(1))
        self.sin_pos_embed = SinusoidalPositionalEmbedding(embedding_dim=out_dim // 2, padding_idx=0, init_size=out_dim // 2)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert C == self.dim, "wrong in PatchMerging"
        # out_size
        x = x.view(B, H, W, -1)
        x = x.permute(0, 3, 1, 2).contiguous()   # B,C,H,W
        x = self.upsample(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(B, L*4, C)   # B,H,W,C
        x = self.norm(x)
        x = self.reduction(x)

        # Add SPE    
        x = x.reshape(B, H * 2, W * 2, self.out_dim).permute(0, 3, 1, 2)
        x += self.sin_pos_embed.make_grid2d(H * 2, W * 2, B) * self.alpha#B,C,H/2,W/2
        # print('bili: ', x.shape)
        # x = x.permute(0, 2, 3, 1).contiguous().view(B, H * 2 * W * 2, self.out_dim)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        # LN
        flops = 4 * H * W * self.dim
        # proj
        flops += 4 * H * W * self.dim * (self.out_dim)
        # SPE
        flops += 4 * H * W * 2
        # bilinear
        flops += 4 * self.input_resolution[0] * self.input_resolution[1] * self.dim * 5
        return flops


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class Generator1(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        channel_multiplier=2,
        lr_mlp=0.01,
        enable_full_resolution=8,
        mlp_ratio=4,
        use_checkpoint=False,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0,
        attn_drop_rate=0,
        sft_half=False,
        train=True,
        n_mlp=8,
    ):
        super().__init__()
        self.style_dim = style_dim
        self.size = size
        self.mlp_ratio = mlp_ratio
        if not train:
            layers = [PixelNorm()]
            for _ in range(n_mlp):
                layers.append(
                    EqualLinear(
                        style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                    )
                )
            self.style = nn.Sequential(*layers)
        start = 2
        depths = [2, 2, 2, 2, 2, 2, 2, 2, 2]
        in_channels = [
            512, 
            512, 
            512, 
            512, 
            256 * channel_multiplier, 
            128 * channel_multiplier, 
            64 * channel_multiplier, 
            32 * channel_multiplier, 
            16 * channel_multiplier
        ]  
        
        # in_channels = [
        #     32,
        #     32,
        #     32,
        #     32,
        #     32,
        #     32,
        #     32,
        #     32,
        #     32
        # ]

        end = int(math.log(size, 2))
        num_heads = [max(c // 32, 4) for c in in_channels]
        full_resolution_index = int(math.log(enable_full_resolution, 2))
        window_sizes = [2 ** i if i <= full_resolution_index else 8 for i in range(start, end + 1)]

        self.input = ConstantInput(in_channels[0])
        self.layers = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        num_layers = 0
        
        for i_layer in range(start, end + 1):
            in_channel = in_channels[i_layer - start]
            layer = StyleBasicLayer(dim=in_channel,
                               input_resolution=(2 ** i_layer,2 ** i_layer),
                               depth=depths[i_layer - start],
                               num_heads=num_heads[i_layer - start],
                               window_size=window_sizes[i_layer - start],
                               out_dim=in_channels[i_layer - start + 1] if (i_layer < end) else None,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               upsample=BilinearUpsample if (i_layer < end) else None,
                               use_checkpoint=use_checkpoint, style_dim=style_dim)
            self.layers.append(layer)

            out_dim = in_channels[i_layer - start + 1] if (i_layer < end) else in_channels[i_layer - start]
            upsample = True if (i_layer < end) else False
            to_rgb = ToRGB(out_dim, upsample=upsample, resolution=(2 ** i_layer))
            self.to_rgbs.append(to_rgb)
            num_layers += 2

        self.n_latent = num_layers
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight, gain=.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        noise,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
    ):
        styles = self.style(noise)
        inject_index = self.n_latent

        if truncation < 1:
            style_t = []
            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = torch.cat(style_t, dim=0)
        
        if styles.ndim < 3:
            latent = styles.unsqueeze(1).repeat(1, inject_index, 1)
        else:
            latent = styles

        x = self.input(latent)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        count = 0
        skip = None
        for layer, to_rgb in zip(self.layers, self.to_rgbs):
            x = layer(x, latent[:,count,:], latent[:,count+1,:])
            b, n, c = x.shape
            h, w = int(math.sqrt(n)), int(math.sqrt(n))
            skip = to_rgb(x.transpose(-1, -2).reshape(b, c, h, w), skip)
            count = count + 2

        B, L, C = x.shape
        assert L == self.size * self.size
        x = x.reshape(B, self.size, self.size, C).permute(0, 3, 1, 2).contiguous()
        image = skip

        if return_latents:
            return image, latent
        else:
            return image, None

    def flops(self):
        flops = 0
        for _, layer in enumerate(self.layers):
            flops += layer.flops()
        for _, layer in enumerate(self.to_rgbs):
            flops += layer.flops()
        # 8 FC + PixelNorm
        flops += 1 * 10 * self.style_dim * self.style_dim
        return flops

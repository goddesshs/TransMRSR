
import sys
import os
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn as nn
from networks.MINET import MINet
from networks.network_swimIR import SwinIR
from networks.network_Transmrsr4 import TGP4
from networks.network_restormer import Restormer
from networks.network_Tramsmrsr6 import TGP6
from networks.network_Transmrsr7 import TGP7

from networks.gfpgan import GFPGANv1

def set_gpu(network, gpu_ids):
    if gpu_ids[0] == -1:
        return network
    else:
        
        network = nn.DataParallel(network)
        network.to('cuda')
        
        

    return network

def get_network(opts):
    opts.net_G = opts.net_G.lower()
        
    if opts.net_G == 'minet':
        network = MINet(n_resgroups=opts.n_resgroups, n_resblocks=opts.n_resblocks, n_feats=opts.n_feats)
        
    elif opts.net_G == 'gfpgan':
        network = GFPGANv1(out_size=opts.img_size,
            num_style_feat=512,
            channel_multiplier=2, pretrained=opts.pretrained_path)        
        
    elif opts.net_G == 'swinir':
        network = SwinIR(img_size=opts.img_size, window_size=8, embed_dim=opts.embed_dim, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6])
        
    elif opts.net_G == 'restormer':
        
        network = Restormer()
        
    #wo SC
    elif opts.net_G == 'tgp4':
        network = TGP4(size=opts.img_size, channel_multiplier=1, style_dim=opts.style_dim, pretrained_path=opts.pretrained_path, fix_decoder=opts.fix_decoder, sft_half=opts.sft_half, truncation_path=opts.truncation_path)
        
    #wo MREF
    elif opts.net_G == 'transmrsr6':
        print('111111111111', opts.fix_decoder)
        
        network = TGP6(size=opts.img_size, channel_multiplier=1, style_dim=opts.style_dim, pretrained_path=opts.pretrained_path, fix_decoder=opts.fix_decoder, sft_half=opts.sft_half, truncation_path=opts.truncation_path)
    
    #transmrsr
    elif opts.net_G == 'transmrsr7':
        network = TGP7(size=opts.img_size, channel_multiplier=1, style_dim=opts.style_dim, pretrained_path=opts.pretrained_path, fix_decoder=opts.fix_decoder, sft_half=opts.sft_half, truncation_path=opts.truncation_path)
    

    return network



    # return network

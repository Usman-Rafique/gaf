# A function that returns the desired network

import torch
from config import cfg
import os

from torch import nn

def count_trainable_parameters(model):  
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_network(net_name):
    if net_name == 'AF_plus':
        from models.AF_plus import AF_plus
        net = AF_plus(cfg.data.image_size, cfg.data.border_size, cfg.data.vector_size)
    elif net_name == 'FDS':
        from models.FDS import FDS 
        net = FDS(cfg.data.image_size, cfg.data.border_size, )
    elif net_name == 'GAF':
        from models.GAF import UNet
        net = UNet(in_channels=11, out_channels=1)
    elif net_name == 'feature_loss':
        from models.feature_loss import feature_loss
        net = feature_loss()
    elif net_name == 'discriminator':
        from models.discriminator import NLayerDiscriminator
        net = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3)
    elif net_name == 'gan_loss':
        from models.discriminator import GANLoss
        net = GANLoss(use_lsgan=True)
        
    # set to CUDA and GPUs
    net.cuda()
    net = nn.DataParallel(net, device_ids=cfg.model.device_ids)
    
    return net
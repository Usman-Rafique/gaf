# FDS network
# Inputs: Image, flow field (from AF++)
# Outputs: Flow field

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import cfg
import functools

def apply_warp(source, flow, grid):
    full_flow = grid + flow
    prediction = torch.nn.functional.grid_sample(source, full_flow)
    return prediction

# ResNet generator and ResNet block from PyTorch Cycle GAN and pix2pix by Jun-Yan Zhu
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# FDS
class FDS(nn.Module):
    def __init__(self, image_size, border_size, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(FDS, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # Encoder
        encoder = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            encoder += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        
        for i in range(int(n_blocks/2)):
            encoder += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            
        # 1x1 conv to reduce dimensionality
        encoder += [nn.Conv2d(256, 16, kernel_size=1, padding=0),
                   nn.BatchNorm2d(16)]
        
        self.encoder = nn.Sequential(*encoder)
        
        # Decoder
        decoder = []
        decoder += [nn.Conv2d(16, 256, kernel_size=1, padding=0),
                   nn.BatchNorm2d(256)]
        
        for i in range(int(n_blocks/2)):
            decoder += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            decoder += [#nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),kernel_size=3, stride=2,padding=1, output_padding=1,bias=use_bias),
                      nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1,
                                    bias=use_bias), # output_padding=1,
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        decoder += [nn.ReflectionPad2d(3)]
        decoder += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        decoder += [nn.Tanh()]

        self.decoder = nn.Sequential(*decoder)

        self.drop_out = nn.Dropout(p=0.1)
        self.drop_out50 = nn.Dropout(p=0.5)

        # small grid to apply warp on the feature maps
        yy = torch.linspace(-1, 1, int(image_size[0]/4))
        xx = torch.linspace(-1, 1, int((image_size[1] + (border_size * 2))/4))
        grid_x, grid_y = torch.meshgrid(yy, xx)
        self.grid_small = torch.cat([grid_y.unsqueeze(2), grid_x.unsqueeze(2)], dim=2).cuda()
        
    def forward(self, image_flow, flow):
        # Encoder
        features = self.encoder(image_flow)
        
        # downsample flow
        flow_small = F.interpolate(flow.permute(0, 3, 1, 2), scale_factor=(1 / 4), mode='nearest').permute(0, 2, 3, 1)

        # warp feature maps
        #print('flow:', flow_small.shape)
        #print('grid:', self.grid_small.shape)
        features_warped = apply_warp(features.cuda(), flow_small.cuda(), grid=self.grid_small.unsqueeze(0).cuda())

        image_gen = self.decoder(features_warped)

        return image_gen
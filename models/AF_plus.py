# AF ++ network
# Inputs: Image and motion vector
# Outputs: Flow field

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import cfg
from .resnet import ResNet18_OS8

class AF_plus(nn.Module):
    def __init__(self, image_size, border_size, vector_size):
        # image_size: size of image, needed to construct a grid for Coord Conv layers
        # border_size: needed only for panoramic images
        # vector_size: number of elements of the transformation vector
        super(AF_plus, self).__init__()
        
        # image encoder
        self.resNet = ResNet18_OS8()
        self.resNet.train()

        # decoder
        self.conv1 = nn.Conv2d(in_channels=512 + 2 + vector_size, out_channels=512, padding=(1, 1), kernel_size=(3, 3)) # coordconv
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.bnConv1 = nn.BatchNorm2d(512)

        self.conv1b = nn.Conv2d(in_channels=512, out_channels=256, padding=(1, 1), kernel_size=(3, 3))
        torch.nn.init.xavier_uniform_(self.conv1b.weight)
        self.bnConv1b = nn.BatchNorm2d(256)

        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, padding=1, kernel_size=(3, 3))
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.bnConv2 = nn.BatchNorm2d(256)

        self.conv2b = nn.Conv2d(in_channels=256, out_channels=128, padding=1, kernel_size=(3, 3))
        torch.nn.init.xavier_uniform_(self.conv2b.weight)
        self.bnConv2b = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, padding=(1, 1), kernel_size=(3, 3))
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        self.bnConv3 = nn.BatchNorm2d(64)

        self.conv3b = nn.Conv2d(in_channels=64, out_channels=32, padding=(1, 1), kernel_size=(3, 3))
        torch.nn.init.xavier_uniform_(self.conv3b.weight)
        self.bnConv3b = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(in_channels=34, out_channels=8, padding=(1, 1), kernel_size=(3, 3)) # coordconv
        torch.nn.init.xavier_uniform_(self.conv4.weight)
        self.bnConv4 = nn.BatchNorm2d(8)

        self.conv5 = nn.Conv2d(in_channels=8, out_channels=8, padding=(1, 1), kernel_size=(3, 3))
        torch.nn.init.xavier_uniform_(self.conv5.weight)
        self.bnConv5 = nn.BatchNorm2d(8)

        self.conv6 = nn.Conv2d(in_channels=8, out_channels=2, padding=(1, 1),
                               kernel_size=(3, 3), bias=False)
        torch.nn.init.uniform_(self.conv6.weight, a=-0.01, b=0.01)


        # Grid for bilinear sampling and coordConv
        yy = torch.linspace(-1, 1, image_size[0])
        xx = torch.linspace(-1, 1, image_size[1] + (border_size * 2))
        grid_x, grid_y = torch.meshgrid(yy, xx)
        self.grid = torch.cat([grid_y.unsqueeze(2), grid_x.unsqueeze(2)], dim=2).cuda()
        
        # small grid for coordconv, in the first layer of decoder
        yy = torch.linspace(-1, 1, int(image_size[0]/8))
        xx = torch.linspace(-1, 1, int((image_size[1] + (border_size * 2))/8))
        grid_x, grid_y = torch.meshgrid(yy, xx)
        self.grid_small = torch.cat([grid_y.unsqueeze(2), grid_x.unsqueeze(2)], dim=2).cuda()
        
    def forward(self, image_in, v):  

        # encoder
        y = self.resNet(image_in)

        # tile the motion vector v
        z = v.view(v.shape[0], v.shape[1], 1, 1)
        z = z.expand(z.shape[0], z.shape[1], y.shape[2], y.shape[3])

        # coordinates for coordconv
        coord_small = self.grid_small.permute(2, 0, 1).expand(image_in.shape[0], self.grid_small.shape[2], self.grid_small.shape[0], self.grid_small.shape[1])
        
        u = torch.cat((y, z, coord_small), dim=1)

        u = self.conv1(u)
        u = self.bnConv1(u)
        u = F.relu(u)
        u = self.conv1b(u)
        u = self.bnConv1b(u)
        features = F.relu(u)

        u = F.interpolate(features, scale_factor=2, mode='nearest')
        u = self.conv2(u)
        u = self.bnConv2(u)
        u = F.relu(u)

        u = self.conv2b(u)
        u = self.bnConv2b(u)
        u = F.relu(u)

        u = F.interpolate(u, scale_factor=2, mode='nearest')
        u = self.conv3(u)
        u = self.bnConv3(u)
        u = F.relu(u)

        u = self.conv3b(u)
        u = self.bnConv3b(u)
        u = F.relu(u)

        u = F.interpolate(u, scale_factor=2, mode='nearest')
        
        # coord conv
        coord = self.grid.permute(2, 0, 1).expand(image_in.shape[0], self.grid.shape[2], self.grid.shape[0], self.grid.shape[1])
        u = torch.cat((u, coord), dim=1)

        u = self.conv4(u)
        u = self.bnConv4(u)
        u = F.relu(u)

        u = self.conv5(u)
        u = self.bnConv5(u)
        u = F.relu(u)

        u = self.conv6(u)

        u = torch.tanh(u)  # both channels scaled from -1 to 1

        flow = u.permute(0, 2, 3, 1)

        # expanding the grid to the batch size
        grid2 = self.grid.expand(image_in.shape[0], self.grid.shape[0], self.grid.shape[1],
                                 self.grid.shape[2]).float().cuda() + flow

        image_out = F.grid_sample(input=image_in, grid=grid2)

        return image_out, flow, features
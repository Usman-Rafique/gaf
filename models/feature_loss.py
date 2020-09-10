# Feature loss, also known as perceptual loss
# Inputs: syntheiszed image and GT target image
# Outputs: feature loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import ResNet18_OS8

class feature_loss(nn.Module):
    def __init__(self):
        super(feature_loss, self).__init__()
        # ResNet for feature loss
        self.resNet_feature = nn.Sequential(*list(ResNet18_OS8().children())[:-2]) # not using last 2 layers.

        for param in self.resNet_feature.parameters():
            param.requires_grad = False
        #self.L2_loss = nn.MSELoss(reduction='none')
        self.L2_loss = nn.MSELoss()

    def forward(self, output, true_img):   # x is disp_vector
        #out_feature = self.resNet_feature(output.permute(0, 3, 1, 2))
        #true_feature = self.resNet_feature(true_img.permute(0, 3, 1, 2))
        #L2 = self.L2_loss(out_feature, true_feature)  # new
        #loss_final = torch.mean(torch.mean(torch.mean(L2, dim=3), dim=2), dim=1)
        
        out_feature = self.resNet_feature(output)
        true_feature = self.resNet_feature(true_img)
        loss_final = self.L2_loss(out_feature, true_feature)  
        return loss_final
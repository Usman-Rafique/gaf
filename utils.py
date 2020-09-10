# Utility functions
import torch
import torch.nn as nn
import numbers
import math

# Gaussian blurring in PyTorch:
class Gaussian_Conv_Update(nn.Module):
    def forward(self, sigma):
        # intialize the Gaussian Conv layer...
        kernel_size = 5
        spatial_d = 2
        channels = 3
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * spatial_d
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * spatial_d

        sigma_f = [sigma] * spatial_d  # 2 represents 2D (spatial) data
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma_f, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        # gauss_weight = kernel.cuda()
        return kernel.cuda()

def apply_warp(source, flow, grid):
    full_flow = grid + flow
    prediction = torch.nn.functional.grid_sample(source, full_flow)
    return prediction

# AF++ loss
def af_plus_loss(output, outputs_not_blurred, target, discrim, gan_loss, feature_loss, l1_loss):
    weight_L1 = 1.0
    gan_weight = 0.01
    
    border = cfg.data.border_size
    # remove border
    output = output[:, :, border:-border, :]
    outputs_not_blurred = outputs_not_blurred[:, :, :, border:-border]
    target = target[:, :, :, border:-border]

    target_V = Variable(target)
    discrim_labels_real = discrim(target_V)
    discrim_err_real = gan_loss(discrim_labels_real, target_is_real=True)

    # discriminator forward for generated images
    outputs_V = Variable(outputs_not_blurred)
    discrim_labels_fake = discrim(outputs_V)
    discrim_err_fake = gan_loss(discrim_labels_fake, target_is_real=False)

    discrim_loss = discrim_err_fake + discrim_err_real

    # Network loss for not fooling discriminator
    discrim_labels_fake = discrim(outputs_V)

    loss_net2 = gan_weight * gan_loss(discrim_labels_fake, target_is_real=True)
        

    # Reconstruction loss
    loss_net1 = l1_loss(output, target) * weight_L1

    # perceptual loss
    loss_net3 = weight_L1 * torch.mean(feature_loss(output, target))

    net_loss = loss_net1 + loss_net2 + loss_net3

    return net_loss, discrim_loss

def get_grid(grid_size, border_size):
    yy = torch.linspace(-1, 1, grid_size[0])
    xx = torch.linspace(-1, 1, grid_size[1] + (border_size* 2))
    grid_x, grid_y = torch.meshgrid(yy, xx)
    grid = torch.cat([grid_y.unsqueeze(2), grid_x.unsqueeze(2)], dim=2).cuda()
    
    return grid
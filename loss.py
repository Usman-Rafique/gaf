# Loss function used for training
import torch
from torch.autograd import Variable
from config import cfg


def loss_function(image_out, target, discrim, gan_loss_function, feature_loss, l1_loss, need_gan_loss=True, image_out_blurred=None):
    
    
    cfg.train.weight_l1 = 1.0
    cfg.train.weight_gan = 0.01
    weight_L1 = 1.0
    gan_weight = 0.01

    # remove border
    image_out = image_out[:, :, :, 48:-48]
    target = target[:, :, :, 48:-48]
    if image_out_blurred is not None:
        image_out_blurred = image_out_blurred[:, :, :, 48:-48]
    
    if need_gan_loss:
        target_V = Variable(target)
        discrim_labels_real = discrim(target_V)
        discrim_err_real = gan_loss_function(discrim_labels_real, target_is_real=True)

        # discriminator forward for generated images
        image_out_V = Variable(image_out)
        discrim_labels_fake = discrim(image_out_V)
        discrim_err_fake = gan_loss_function(discrim_labels_fake, target_is_real=False)

        discrim_loss = discrim_err_fake + discrim_err_real

        # Network loss for not fooling discriminator
        discrim_labels_fake = discrim(image_out_V)

        gan_loss = cfg.train.weight_gan * gan_loss_function(discrim_labels_fake, target_is_real=True)
    else:
        gan_loss = 0.0

    if image_out_blurred is not None:
        reconstruction_loss = cfg.train.weight_l1 * l1_loss(image_out_blurred, target)
        perceptual_loss = cfg.train.weight_l1 * torch.mean(feature_loss(image_out_blurred, target))
    else:
        reconstruction_loss = cfg.train.weight_l1 * l1_loss(image_out, target)
        perceptual_loss = cfg.train.weight_l1 * torch.mean(feature_loss(image_out, target))

    # total loss
    net_loss = reconstruction_loss + gan_loss + perceptual_loss

    if need_gan_loss:
        return net_loss, discrim_loss
    else:
        return net_loss
# Train AF++

# Before starting training, make sure that correct model name is set in config.py i.e. (cfg.model.name should be 'AF_plus') 

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from config import cfg
from data_factory import get_dataset
from net_factory import get_network
from torch.autograd import Variable

from loss import loss_function
from utils import Gaussian_Conv_Update, apply_warp, af_plus_loss, get_grid

# setup checkpoint directory
out_dir = cfg.train.out_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
else:
    print('Folder already exists. Are you sure you want to overwrite results?')

print('Configuration: \n', cfg) 

assert cfg.model.name=='AF_plus', 'To train AF++, set cfg.model.name  to \'AF_plus\''

# dataloaders
train_loader = get_dataset(dataset_name=cfg.data.name, mode='train')
test_loader = get_dataset(dataset_name=cfg.data.name, mode = 'test')
print('Data loaders have been prepared!')

# networks
af_plus = get_network('AF_plus')
feature_loss = get_network('feature_loss')
discrim = get_network('discriminator')
gan_loss = get_network('gan_loss') 

# loss function
l1_loss = torch.nn.L1Loss()

# adaptive scale space during training
sigma = Variable(2.0 * torch.ones(1), requires_grad=True).float()  # scale
gauss_conv = F.conv2d
gauss_update_method = Gaussian_Conv_Update()

# grid for bilinear sampling
grid = get_grid(cfg.data.image_size, cfg.data.border_size)

# optimizer for the network
param1 = list(af_plus.parameters())
param2 = [sigma]

optim = torch.optim.Adam([{'params':param2, 'lr':0.001, 'weight_decay':0.0},{'params':param1}],
                         lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)

# optimizer for the discriminator
optim_d = torch.optim.Adam(discrim.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)

ep = cfg.train.num_epochs # number of epochs

# initialize logging variables
train_loss = [] 
test_loss = []
best_val_loss = 999.0

train_loss_d = []  # for discriminator
test_loss_d = []

sigma_log = []

print('Starting training...')


for epoch in range(cfg.train.num_epochs):
    # training loop
    loss_train = 0
    loss_train_d = 0
    af_plus.train()
    discrim.train()
    for i, data in enumerate(train_loader, 0):
        optim.zero_grad()
        optim_d.zero_grad()

        source = data['source'].float().cuda()
        target = data['target'].float().cuda()
        vec = data['vec'].float().cuda()
        
        image_flow, flow, _  = af_plus(source, vec)
        
        # apply the scale space
        source_pad = F.pad(source, (1, 1, 1, 1))  
        gauss_weight = gauss_update_method(sigma)
        source_blurred = gauss_conv(source_pad, weight=gauss_weight, groups=3)

        image_flow_blurred = apply_warp(source_blurred, flow, grid)
        
        loss_net1, discrim_loss = loss_function(image_flow, target, discrim, gan_loss, feature_loss,
                                             l1_loss, need_gan_loss=True, image_out_blurred=image_flow_blurred)
        
        # discriminator optimization
        discrim_loss.mean().backward()
        optim_d.step()

        blur_regularization = 1e-3*blur_regularization.cuda()
        
        net_loss = loss_net1 + blur_regularization
        net_loss.mean().backward()
        
        # main network optimization
        optim.step()
        
        loss_train += net_loss.mean().item()
        loss_train_d += discrim_loss.mean().item()
        
        if (i+1)%100==0:
            print('[Ep',(epoch+1), ': ', (i+1), 'of', len(train_loader), ']', 'loss: ', loss_train/(i+1), 'discrim loss:', loss_train_d/(i+1))
            print('sigma:', sigma.item())
            
    loss_train /= len(train_loader)
    loss_train_d /= len(train_loader)
    
    # optional validation loop
    af_plus.eval()
    discrim.eval()
    loss_val = 0
    loss_val_d = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            if i > 49:  # save time
                break
                
            source = data['source'].float().cuda()
            target = data['target'].float().cuda()
            vec = data['vec'].float().cuda()

            image_flow, flow, _  = af_plus(source, vec)
        
            # apply the scale space
            source_pad = F.pad(source, (1, 1, 1, 1))  
            gauss_weight = gauss_update_method(sigma)
            source_blurred = gauss_conv(source_pad, weight=gauss_weight, groups=3)

            image_flow_blurred = apply_warp(source_blurred, flow, grid)

            loss_net1, discrim_loss = loss_function(image_flow, target, discrim, gan_loss, feature_loss,
                                                 l1_loss, need_gan_loss=True, image_out_blurred=image_flow_blurred)

            
            blur_regularization = 1e-3*sigma.cuda()

            net_loss = loss_net1 + blur_regularization    
            
            loss_val += net_loss.mean().item()
            loss_val_d += discrim_loss.mean().item()
            
    loss_val /= 50
    loss_val_d /= 50
    
    # end of epoch printing
    print('End of epoch ', epoch+1, ' , Train loss: ', loss_train, ', val loss: ', loss_val)
    print('sigma:', sigma.item())
    
    # save logs
    train_loss.append(loss_train)
    train_loss_d.append(loss_train_d)
    
    test_loss.append(loss_val)
    test_loss_d.append(loss_val_d)
    
    sigma_log.append(sigma.detach().item())
    
    # model checkpoint
    if loss_val < best_val_loss:
        best_val_loss = loss_val
        fname = 'af_plus_dict.pth'
        torch.save(af_plus.state_dict(), os.path.join(out_dir, fname))
        print('=========== model saved at epoch: ', epoch+1, ' =================')
        
# end of training
print('Training finished')
fname = 'model_dict_end.pth'
torch.save(af_plus.state_dict(), os.path.join(out_dir, fname))
print('=========== model saved at the end of training =================')

fname = os.path.join(out_dir, 'logging.txt')
with open(fname, 'w') as result_file:
    result_file.write('Training log:')
    result_file.write('Validation loss')
    result_file.write(str(test_loss))
    result_file.write('Training loss')
    result_file.write(str(train_loss))
    result_file.write('\nSigma \n')
    result_file.write(str(sigma_log))

# save loss curves        
plt.figure()
plt.plot(train_loss)
plt.plot(test_loss)
plt.legend(['train loss', 'test loss'])
plt.title('Network loss')
fname = os.path.join(out_dir,'loss.png')
plt.savefig(fname)
plt.close()


plt.figure()
plt.plot(train_loss_d)
plt.plot(test_loss_d)
plt.legend(['train loss', 'test loss'])
plt.title('Discriminator loss')
fname = os.path.join(out_dir,'discrim_loss.png')
plt.savefig(fname)
plt.close()


plt.figure()
plt.plot(sigma_log)
plt.title('sigma')
fname = os.path.join(out_dir,'sigma.png')
plt.savefig(fname)
plt.close()

print('All done...')


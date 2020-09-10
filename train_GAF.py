# Train direct GAF.

# GAF combines the flow-based output (from AF++) and generative output (from FDS). A fusion network (a U-Net) is trained to combine flow and generative outputs.

# Inputs: source image, flow-based image and flow (from AF++), generative image

# Before starting training, make sure that correct model name is set in config.py i.e. (cfg.model.name should be 'GAF') 

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

# dataloaders
train_loader = get_dataset(dataset_name=cfg.data.name, mode='train')
test_loader = get_dataset(dataset_name=cfg.data.name, mode = 'test')
print('Data loaders have been prepared!')

# networks
af_plus = get_network('AF_plus')
fds = get_network('FDS')
fusion = get_network('GAF')
feature_loss = get_network('feature_loss')
discrim = get_network('discriminator')
gan_loss = get_network('gan_loss') 

# load weights of a trained AF++ 
af_plus.load_state_dict(torch.load(os.path.join('checkpoints', 'af_plus_dict.pth')))
af_plus.eval()

# load weights of a trained FDS network 
fds.load_state_dict(torch.load(os.path.join('checkpoints', 'fds_dict.pth')))
fds.eval()

# loss function
l1_loss = torch.nn.L1Loss()


# optimizer for the network
optim = torch.optim.Adam(fusion.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)

# optimizer for the discriminator
optim_d = torch.optim.Adam(discrim.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)

ep = cfg.train.num_epochs # number of epochs

# initialize logging variables
train_loss = []  # for network
test_loss = []
best_val_loss = 999.0

train_loss_d = []  # for discriminator
test_loss_d = []


print('Starting training...')


for epoch in range(cfg.train.num_epochs):
    # training loop
    loss_train = 0
    loss_train_d = 0
    fusion.train()
    discrim.train()
    for i, data in enumerate(train_loader, 0):
        optim.zero_grad()
        optim_d.zero_grad()

        source = data['source'].float().cuda()
        target = data['target'].float().cuda()
        vec = data['vec'].float().cuda()
        
        image_flow, flow, _  = af_plus(source, vec)
        
        image_gen = fds(source, flow)
                
        # prepare input for fusion
        input_to_fusion = torch.cat([ source, image_flow, image_gen,  flow.permute(0, 3, 1, 2)], dim=1)
        
        # get fusion scores
        fusion_scores = torch.sigmoid(fusion(input_to_fusion))
        
        # final GAF synthesis
        image_gaf = fusion_scores*image_flow + (1 - fusion_scores)*image_gen
        
        # loss
        loss_net1, discrim_loss = loss_function(image_gaf, target, discrim, gan_loss, feature_loss, l1_loss)
        
        # discriminator optimization
        discrim_loss.mean().backward()
        optim_d.step()
        
        net_loss = loss_net1 
        net_loss.mean().backward()
        
        # main network optimization
        optim.step()
        
        loss_train += net_loss.mean().item()
        loss_train_d += discrim_loss.mean().item()
        
        if (i+1)%100==0:
            print('[Ep',(epoch+1), ': ', (i+1), 'of', len(train_loader), ']', 'loss: ', loss_train/(i+1), 'discrim loss:', loss_train_d/(i+1))
            
    loss_train /= len(train_loader)
    loss_train_d /= len(train_loader)
    
    # optional validation loop
    fusion.eval()
    discrim.eval()
    loss_val = 0
    loss_val_d = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            if i >49:  # save time
                break
                
            source = data['source'].float().cuda()
            target = data['target'].float().cuda()
            vec = data['vec'].float().cuda()

            image_flow, flow, _  = af_plus(source, vec)
        
            image_gen = fds(source, flow)

            # prepare input for fusion
            input_to_fusion = torch.cat([ source, image_flow, image_gen,  flow.permute(0, 3, 1, 2)], dim=1)
            
            # get fusion scores
            fusion_scores = torch.sigmoid(fusion(input_to_fusion))

            # final GAF synthesis
            image_gaf = fusion_scores*image_flow + (1 - fusion_scores)*image_gen

            # loss
            loss_net1, discrim_loss = loss_function(image_gaf, target, discrim, gan_loss, feature_loss, l1_loss)
            
            net_loss = loss_net1
            
            loss_val += net_loss.mean().item()
            loss_val_d += discrim_loss.mean().item()
            
    loss_val /= 50
    loss_val_d /= 50
    
    # end of epoch printing
    print('End of epoch ', epoch+1, ' , Train loss: ', loss_train, ', val loss: ', loss_val)

    # save logs
    train_loss.append(loss_train)
    train_loss_d.append(loss_train_d)
    
    test_loss.append(loss_val)
    test_loss_d.append(loss_val_d)
    
    # model checkpoint
    if loss_val < best_val_loss:
        best_val_loss = loss_val
        fname = 'fds_dict.pth'
        torch.save(fds.state_dict(), os.path.join(out_dir, fname))
        print('=========== model saved at epoch: ', epoch+1, ' =================')
        
# end of training
print('Training finished')
fname = 'fds_dict_end.pth'
torch.save(fds.state_dict(), os.path.join(out_dir, fname))
print('=========== model saved at the end of training =================')

fname = os.path.join(out_dir, 'logging.txt')
with open(fname, 'w') as result_file:
    result_file.write('Training log:')
    result_file.write('Validation loss')
    result_file.write(str(test_loss))
    result_file.write('Training loss')
    result_file.write(str(train_loss))
    
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


print('All done...')


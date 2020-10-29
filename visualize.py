# Generate results from a trained model.

# Before starting visualization, make sure that correct model name is set in config.py

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from config import cfg
from data_factory import get_dataset
from net_factory import get_network
from torch.autograd import Variable

# setup checkpoint directory
checkpoint_dir = 'checkpoints'

# make a directory for image results
out_dir = os.path.join(cfg.train.out_dir, 'image_results_'+cfg.model.name)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
else:
    print('overwriting results')

print('Configuration: \n', cfg) 

# dataloaders
cfg.train.shuffle = False
test_loader = get_dataset(dataset_name=cfg.data.name, mode = 'test')
print('Data loaders have been prepared!')

# network
af_plus = get_network('AF_plus')

# load pretrained model
if cfg.model.name == 'AF_plus':
    af_plus.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'af_plus_dict.pth')))
elif cfg.model.name == 'FDS':
    af_plus.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'af_plus_dict.pth')))
    
    fds = get_network('FDS')
    fds.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'fds_dict.pth')))
    fds.eval()
elif cfg.model.name == 'GAF':
    af_plus.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'af_plus_dict.pth')))
    
    fds = get_network('FDS')
    fds.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'fds_dict.pth')))
    fds.eval()
    
    fusion = get_network('GAF')
    fusion.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'fusion_dict.pth')))
    fusion.eval()
        
af_plus.eval()    
print('Starting visualization...')


with torch.no_grad():
    ctr = 0
    for i, data in enumerate(test_loader, 0):
        if i>1:
            break
        source = data['source'].float().cuda()
        target = data['target'].float().cuda()
        vec = data['vec'].float().cuda()

        image_out, flow, _  = af_plus(source, vec)
        
        if cfg.model.name == 'FDS':
            image_out = fds(source, flow)
        
        elif cfg.model.name == 'GAF':
            image_gen = fds(source, flow)
            input_to_fusion = torch.cat([ source, image_out, image_gen,  flow.permute(0, 3, 1, 2)], dim=1)
            fusion_scores = torch.sigmoid(fusion(input_to_fusion)) # get fusion scores

            image_out = fusion_scores*image_out + (1 - fusion_scores)*image_gen # final GAF synthesis
        
        # remove border
        source = source[:, :, :, 48:-48].permute(0,2,3,1).detach().cpu().numpy()
        target = target[:, :, :, 48:-48].permute(0,2,3,1).detach().cpu().numpy()
        image_out = image_out[:, :, :, 48:-48].permute(0,2,3,1).detach().cpu().numpy()

        plt.ioff()
        my_dpi = 300 
        
        for k in range(source.shape[0]):
                        
            plt.figure(figsize=(160 / my_dpi, 960 / my_dpi), dpi=8.0 * my_dpi)
            plt.imshow(source[k, :, :, :])
            plt.axis('off')
            fname1 = str(str(ctr) + '_source' + '.png')
            plt.savefig(os.path.join(out_dir, fname1), bbox_inches='tight', dpi=8.0*my_dpi, pad_inches=0.0)
            plt.close()

            plt.figure(figsize=(160 / my_dpi, 960 / my_dpi), dpi=8.0 * my_dpi)
            plt.imshow(target[k, :, :, :])
            plt.axis('off')
            fname1 = str(str(ctr) + '_target' + '.png')
            plt.savefig(os.path.join(out_dir, fname1), bbox_inches='tight', dpi=8.0*my_dpi, pad_inches=0.0)
            plt.close()

            plt.figure(figsize=(160 / my_dpi, 960 / my_dpi), dpi=8.0 * my_dpi)
            plt.axis('off')
            plt.imshow(image_out[k, :, :, :])
            fname1 = str(str(ctr) + '_output' + '.png')  # naming ans saving
            plt.savefig(os.path.join(out_dir, fname1), bbox_inches='tight', dpi=8.0*my_dpi, pad_inches=0.0)
            plt.close()
            
            plt.figure(dpi=150)
            plt.subplot(3, 1, 1)
            plt.imshow(source[k, :, :, :])
            plt.axis('off')
            plt.title('source')
            
            plt.subplot(3, 1, 2)
            plt.imshow(image_out[k, :, :, :])
            plt.axis('off')
            plt.title(cfg.model.name)
            
            plt.subplot(3, 1, 3)
            plt.imshow(target[k, :, :, :])
            plt.axis('off')
            plt.title('target')
            fname1 = str(str(ctr) + '_combined' + '.png')  # naming ans saving
            plt.savefig(os.path.join(out_dir, fname1), bbox_inches='tight', pad_inches=0.0)
            plt.close()
            
            ctr += 1


print('All done saving images')
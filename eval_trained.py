# Quantitative evaluation of a trained model

# Before starting evaluation, make sure that correct model name is set in config.py 

import torch
import numpy as np
import os
from config import cfg
from data_factory import get_dataset
from net_factory import get_network
import skimage.measure as measure
import skimage.metrics as metrics


def PSNR_single(prediction, target):
    diff = prediction-target
    #diff = diff.view(1,-1)
    diff = np.reshape(diff,(1,-1))
    diff = np.mean(diff**2,1)
    psnr = 10*np.log10(1/diff)
    return (psnr)

# setup checkpoint directory
checkpoint_dir = 'checkpoints'

# make a directory for image results
out_dir = os.path.join(cfg.train.out_dir, 'eval_results_'+cfg.model.name)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
else:
    print('overwriting results')

print('Configuration: \n', cfg) 

# dataloaders
cfg.train.shuffle = True
test_loader = get_dataset(dataset_name=cfg.data.name, mode = 'test')
print('Data loaders have been prepared!')

# network
af_plus = get_network('AF_plus')

# load pretrained model
if cfg.model.name == 'AF_plus':
    af_plus.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'af_plus_dict.pth')))
elif cfg.model.name == 'FDS':
    fds = get_network('FDS')
    fds.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'fds_dict.pth')))
    af_plus.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'af_plus_dict.pth')))
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

l1_function = torch.nn.L1Loss()  # l1 error

print('Starting evaluation...')

l1 = 0.0
ssim = 0.0
psnr = 0.0
psnr2 = 0.0

with torch.no_grad():
    ctr = 0
    for i, data in enumerate(test_loader, 0):
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
            
        source = source[:, :, :, cfg.data.border_size:-cfg.data.border_size]
        target = target[:, :, :, cfg.data.border_size:-cfg.data.border_size]
        image_out = image_out[:, :, :, cfg.data.border_size:-cfg.data.border_size]
        
        l1 += l1_function(image_out, target).item()
        
        source = source.permute(0,2,3,1).detach().cpu().numpy()
        target = target.permute(0,2,3,1).detach().cpu().numpy()
        image_out = image_out.permute(0,2,3,1).detach().cpu().numpy()
        
        # SSIM and PSNR are computed separately for each image
        ssim_batch = 0
        psnr_batch = 0
        psnr_batch2 = 0
        for k in range(source.shape[0]):
            ssim_batch += metrics.structural_similarity(target[k,:,:,:], image_out[k,:,:,:], multichannel=True)
            
            psnr_batch2 += PSNR_single(image_out[k,:,:,:], target[k,:,:,:])
            psnr_batch += metrics.peak_signal_noise_ratio(target[k,:,:,:], image_out[k,:,:,:], data_range=1.0)
            
        # average of batch
        ssim_batch /= float(source.shape[0])
        psnr_batch /= float(source.shape[0])
        psnr_batch2 /= float(source.shape[0])
        
        ssim += ssim_batch
        psnr += psnr_batch
        psnr2 += psnr_batch2
    
    l1 /= float(len(test_loader))
    ssim /= float(len(test_loader))
    psnr /= float(len(test_loader))
    psnr2 /= float(len(test_loader))

    
print('l1', l1)
print('ssim', ssim)
print('psnr', psnr)
print('psnr2', psnr2)

result_name = os.path.join(out_dir, "test_eval_result.txt")

with open(result_name, 'w') as result_file:
    result_file.write('Results on the test set \n')
    result_file.write('\nL1 error = ')
    result_file.write(str(l1))
    result_file.write('\nSSIM = ')
    result_file.write(str(ssim))
    result_file.write('\nPSNR = ')
    result_file.write(str(psnr))
    result_file.write('\nPSNR2 = ')
    result_file.write(str(psnr2))

print('Finished evaluation')

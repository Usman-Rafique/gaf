import torch
import torchvision
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from config import cfg
import csv
import utm
import PIL
from PIL import Image

def GetLatLongFromNameAndPath(file_name):  
    a = os.path.split(file_name)
    lat_st, long_st = a[1].split('_')
    latt = np.float64(lat_st)
    longg = np.float64(long_st.replace('.jpg',''))
    return latt, longg

def AddBorder(image, num_pixels):       
    left_border = image[:,-num_pixels:,:]
    right_border = image[:,:num_pixels,:]
    return np.hstack((left_border, image, right_border))

def AddBorder_tensor(image, num_pixels):       # tensor version
    left_border = image[:,:,-num_pixels:]
    right_border = image[:,:,:num_pixels]
    return torch.cat((left_border, image, right_border), dim=2)

# Normal distribution
def get_gauss_value(x, mu, sig):
    val = (1 / (2*np.pi*sig*sig)**0.5) * (np.exp(- (x-mu)**2 / (2*sig*sig) ) )
    return val

def make_gaussian_vector(x, y, sigma=0.75):
    inp = np.arange(start=-12, stop=13, step=1)
    x_vec = get_gauss_value(inp, mu=x, sig = sigma)
    y_vec = get_gauss_value(inp, mu=y, sig=sigma)
    return np.concatenate((x_vec, y_vec), axis=0)

class Dataset_BPS(Dataset):    
    def __init__(self, split_file, root_dir ):

        with open(split_file, 'r') as f:
            self.rows = [i[:-1] for i in f]

        self.root_dir = os.path.join(root_dir, 'images')
        
        self.resize = transforms.Resize((cfg.data.image_size[0], cfg.data.image_size[1]))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        # reading source
        #print('raw row', self.rows[idx])
        source_name = os.path.join(self.root_dir, self.rows[idx].split(',')[0])
        source_image = Image.open(source_name)
        
        # reading target
        target_name = os.path.join(self.root_dir, self.rows[idx].split(',')[1])
        target_image = Image.open(target_name)
        
        # displacement vector
        lat_source, long_source = GetLatLongFromNameAndPath(source_name)
        source_loc = utm.from_latlon(lat_source, long_source)
        
        lat_target, long_target = GetLatLongFromNameAndPath(target_name)
        target_loc = utm.from_latlon(lat_target, long_target)

        # vector from source to target
        disp_vec =  np.asarray([target_loc[0]-source_loc[0], target_loc[1]-source_loc[1]]) #

        # Gaussian displacement vetor
        vec = make_gaussian_vector(disp_vec[0], disp_vec[1])

        # convert to tensor
        source_image = self.to_tensor(source_image)
        target_image = self.to_tensor(target_image)
        
        # Check if alignment is needed
        if cfg.data.align:
            # angle from x-axis (or east direction)
            theta_x = (180.0 / np.pi) * np.arctan2(disp_vec[1], disp_vec[0])  # first y and then x i.e. arctand (y/x)

            # angle from y-axis or north
            theta_y = 90 + theta_x

            if theta_y < 0:  # fixing negative
                theta_y += 360

            column_shift = np.int(
                theta_y * (cfg.data.image_size[1]/360.0) )   

            source_image = torch.roll(source_image, column_shift, dims=2)  # rotate columns
            target_image = torch.roll(target_image, column_shift, dims=2)  # rotate columns
            
        source_image = AddBorder_tensor(source_image, cfg.data.border_size)
        target_image = AddBorder_tensor(target_image, cfg.data.border_size)

        sample = {'source': source_image, 'target': target_image, 'vec':vec}
        return sample
    

def get_dataset(dataset_name, mode):
    #if dataset_name == 'BPS':
         #data_folder = os.path.join(cfg.data.root_dir, 'BPS')
    
    if dataset_name == 'BPS':
        split_file = os.path.join(os.path.join(cfg.data.root_dir, cfg.data.name), 'splits', mode+'_list.txt')
    
    # making dataSet class 
    ds = Dataset_BPS(split_file, os.path.join(cfg.data.root_dir, cfg.data.name) )
    
    # preparing pytorch data loader
    ds_final = torch.utils.data.DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=cfg.train.shuffle,
                                           num_workers=cfg.train.num_workers, drop_last=True)
        
    return ds_final
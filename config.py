# All the configurations are stored in this file

from easydict import EasyDict as edict
import os

cfg = edict()

# model
cfg.model = edict()

# network name
cfg.model.name = 'GAF'        # options are 'AF_plus', 'FDS', or 'GAF'
cfg.model.device_ids = [0, 1, 2, 3]

# data
cfg.data = edict()
cfg.data.name = 'BPS'
cfg.data.root_dir = 'data'
cfg.data.image_size = (160, 960)
cfg.data.border_size = 48
cfg.data.vector_size = 50
cfg.data.align = True   # aligns according to the motion direction

# training details
cfg.train = edict()
cfg.train.batch_size = 16
cfg.train.learning_rate = 2.0*1e-5    
cfg.train.shuffle = True              
cfg.train.num_epochs = 10 
cfg.train.num_workers = 2 
cfg.train.weight_decay = 1.0*1e-6      # weight decay

cfg.train.weight_l1 = 1.0
cfg.train.weight_gan = 0.01

cfg.train.out_dir = './outputs/1'    # This is the directory where the trained models are saved (or read from).
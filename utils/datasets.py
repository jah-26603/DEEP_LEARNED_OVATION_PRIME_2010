# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 10:43:33 2025

@author: JDawg
"""

import os
import numpy as np
from torch.utils.data import Dataset
import torch
import yaml
import matplotlib.pyplot as plt
import cv2

#%%
#classical model with deterministic DL methods

class OP_dataset(Dataset):
    '''Ovation prime dataset'''
    def __init__(self, spwd_dir =r'E:\ml_aurora\paired_data\swdata', image_dir = r'E:\ml_aurora\paired_data\images' , transform=None):

        self.spwd_dir = spwd_dir
        self.image_dir = image_dir
        self.transform = transform
        self.titles = os.listdir(spwd_dir)
        
        with open("config.yaml") as stream:
            stat_dict =yaml.safe_load(stream)
        
        #values determined across entire dataset
        self.input_mean = np.array(stat_dict['data_statistics']['input_mean'])[:4]
        self.input_std = np.array(stat_dict['data_statistics']['input_std'])[:4]
        self.image_mean = np.array(stat_dict['data_statistics']['op_img_mean'])
        self.image_std = np.array(stat_dict['data_statistics']['op_img_std'])

        #remove entirely 0 datapoints during loading since this case doesn't matter, or I could calculate it at the end with the loss
        # self.valid_titles = []
        # for t in titles:
        #     inputs = np.load(os.path.join(spwd_dir, t))
        #     mask = np.isnan(inputs)
        #     if mask.mean() <= 0.5:
        #         self.valid_titles.append(t)
                
                
    def __len__(self):
        return len(os.listdir(self.spwd_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        #here idx is the title name
        img = np.load(os.path.join(self.image_dir, self.titles[idx]))
        inputs = np.load(os.path.join(self.spwd_dir, self.titles[idx]))
        
        mask = np.isnan(inputs)
        
        if len(mask[mask == True])/len(inputs) > .5:
            inputs = np.zeros_like(inputs)
            img = np.zeros_like(img)
            
        
        # # img = median_filter(img,size = 3) #smooth
        # for c in range(img.shape[0]):
        #     img[c] = median_filter(img[c], size=3)
        
        inputs[:,:-1] = (inputs[:,:-1] - self.input_mean)/self.input_std
        inputs[:,-1] = inputs[:,-1]/365 #see if there is maybe an issue here...
        img = (img - self.image_mean[:,None,None])/self.image_std[:,None,None]
        sample = {'image': img, 'inputs': inputs}

        inputs[mask] = 0
        return sample


# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 23:23:43 2025

@author: JDawg
"""

import utils
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pytorch_msssim import ssim
from pathlib import Path
import sys

sys.path.insert(0, str(Path("OvationPyme-master").resolve()))
device = "cuda" if torch.cuda.is_available() else "cpu"


#do this for the first run, this will download all of the training data
fp_pd = r'E:\ml_aurora' #parent directory
if False:                  
    utils.download_solar_wind_data(out_dir = os.path.join(fp_pd, 'solar_wind')) #downloads files
    utils.collate_solar_wind(fp = os.path.join(fp_pd, 'solar_wind'), out = os.path.join(fp_pd, 'organized_solar_wind.csv')) #creates the solar wind dataframe  
    utils.OP_training_data(solar_wind_df = os.path.join(fp_pd, 'organized_solar_wind.csv'), out = os.path.join(fp_pd, 'ovation_paired_data'))


#%%

train = True

device = "cuda" if torch.cuda.is_available() else "cpu"
model = utils.FC_to_Conv().to(device)

# dataset = OP_dataset(spwd_dir =r'D:\ml_aurora\paired_data\swdata', image_dir = r'D:\ml_aurora\paired_data\images')
dataset = utils.OP_dataset(spwd_dir =r'E:\ml_aurora\ovation_paired_data\swdata', image_dir = r'E:\ml_aurora\ovation_paired_data\images')

subset_indices = list(range(128 *400)) # first 12800 samples
dataset = Subset(dataset, subset_indices)

train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size

train_loss_hist, val_loss_hist= [],[]

train_set, val_set = random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_set,
                            batch_size=32,
                            shuffle=True
                        )
val_dataloader = DataLoader(val_set, 
                            batch_size=32,
                            shuffle=True,
                        )


# Better optimizer settings
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay = 1e-3)
criterion = nn.HuberLoss()
num_epochs = 10
best_val = 100
loss = []
if train:

    for epoch in range(num_epochs):
        # ---------- Training ----------
        model.train()
        train_running_loss = 0.0
        val_running_loss = 0
        loop = tqdm(train_dataloader)
        
        loss_list = []
        for batch in loop:
            x = batch["inputs"].unsqueeze(1).to(device, dtype=torch.float32)
            y = batch["image"].unsqueeze(1).to(device, dtype=torch.float32)
    
            
            optimizer.zero_grad()
            y_hat = model(x)
            y_hat = (y_hat).unsqueeze(1)
            
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            
            loop.set_description(f"Train Loss: {loss.item():.4f}")
            loss_list.append(loss.item())
        
        train_loss = np.mean(np.array(loss_list))
        loss.append(train_loss)
        # scheduler.step()
        print(f'Epoch {epoch + 1} Loss: {loss[-1]:.4f}')
    
        
        model.eval()
        with torch.no_grad():
            loop = tqdm(val_dataloader)
            loss_list = []
            for batch in loop:
                x = batch["inputs"].unsqueeze(1).to(device, dtype=torch.float32)
                y = batch["image"].unsqueeze(1).to(device, dtype=torch.float32)
    
                
                y_hat = model(x)
                y_hat = (y_hat).unsqueeze(1)
                
                loss = criterion(y_hat, y)
                loss_list.append(loss.item())

                
            
            val_loss = np.mean(loss_list)
            
            print(f'Val Loss epoch {epoch +1}: {val_loss:.4f}')
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), f'weights/OP_weight.pth')
                print('saving new weights...')
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
            
model.load_state_dict(torch.load(r'weights/OP_weight.pth', weights_only = True))
model.eval()

batch = next(iter(val_dataloader))
x = batch["inputs"].unsqueeze(1).to(device, dtype=torch.float32)
y = batch["image"].unsqueeze(1).to(device, dtype=torch.float32)
y_hat = model(x)
y_hat = (y_hat).unsqueeze(1)
for i in range(8): #unique images
    for c in range(16): 
        # order goes as: North_energy_diff, North_energy_mono, North_energy_wave, North_energy_ions
        #                North_number_diff, North_number_mono, North_number_wave, North_number_ions
        #                South_energy_diff, South_energy_mono, South_energy_wave, South_energy_ions
        #                South_number_diff, South_number_mono, South_number_wave, South_number_ions

        plt.figure()
        gt = y[i,0,c].detach().cpu().numpy()
        pr = y_hat[i,0,c].detach().cpu().numpy()


        vmax = max(np.max(gt), np.max(pr))
        vmin = min(np.min(gt), np.min(pr))

        plt.subplot(1,2,1)
        plt.imshow(gt.T, vmin = vmin, vmax = vmax)
        plt.title('Model truth')
        plt.colorbar(orientation = 'horizontal')


        plt.subplot(1,2,2)
        plt.imshow(pr.T, vmin = vmin, vmax = vmax)
        plt.colorbar(orientation = 'horizontal')
        plt.title('Model prediction')

        plt.show()  
        




plt.figure()
plt.plot(train_loss, label = 'train loss')
plt.plot(val_loss, label = 'val loss')
plt.legend()
plt.show()

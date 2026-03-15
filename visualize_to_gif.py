# -*- coding: utf-8 -*-
"""
Created on Thu Dec 25 18:36:57 2025

@author: dogbl
"""

from dataloader import FC_to_Conv, OP_dataset
from tqdm import tqdm
from download_training_data import collate_solar_wind, download_solar_wind_data,download_OP_runs
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from dataloader import FC_to_Conv, OP_dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pytorch_msssim import ssim
import datetime





#need to create a function that will create a tiny dataset of solar wind -> then save

df = pd.read_csv(r'D:\ml_aurora\organized_solar_wind.csv')
df['datetime'] = pd.to_datetime(
    dict(
        year=df.year,
        month=df.month,
        day=df.day,
        hour=df.hhmm // 100,
        minute=df.hhmm % 100
    )
)

#%%


def get_time_series_inference(df, date = datetime.datetime(2015,3,17,0,0), cadence = 5, n_samples = 288*2):
    
    cadence = 5 # minutes between each inference
    n_samples = n_samples
    s_idx = df.loc[df.datetime == date].index[0]
    
    input_mean = np.load(r'input_mean.npy')
    input_std = np.load(r'input_std.npy')
    
    
    input_data = []
    timestamps = []
    for i in range(n_samples):
        swd = df.iloc[-240 + i*cadence + s_idx: i*cadence + s_idx + 1]
        
        dt = swd.iloc[-1]['datetime']# datetime at ACE
        delta_t = 1.5e6 / np.nanmedian(swd['vel'].iloc[-35:].to_numpy()) 
        if np.isnan(delta_t) is True:# L1 → bowshock propagation
            delta_t = 3600
        bowshock_dt = dt + datetime.timedelta(minutes=delta_t // 60)
        ts = bowshock_dt
        
        
        # Remove last row (used for timestamp)
        swd = swd.iloc[:-1]
        doy = bowshock_dt.timetuple().tm_yday
        inputs = swd[['Bx', 'By', 'Bz', 'vel']].to_numpy()
        inputs = np.hstack([inputs, np.array([doy] * len(inputs))[:,None]])
        
        
        #this is preprocessing steps
        mask = np.isnan(inputs)
        if len(mask[mask == True])/len(inputs) > .5:
            inputs = np.zeros_like(inputs)
            
        inputs[:,:-1] = (inputs[:,:-1] - input_mean)/input_std
        inputs[:,-1] = inputs[:,-1]/365 #see if there is maybe an issue here...
    
        
        inputs[mask] = 0
        timestamps.append(ts)
        input_data.append(torch.tensor(inputs, dtype = torch.float32))
    all_solar_wind = df.iloc[-240 + s_idx: n_samples*cadence + s_idx]
    all_solar_wind = all_solar_wind[['Bx', 'By', 'Bz', 'vel', 'datetime']]
    return input_data, timestamps, all_solar_wind




input_data, timestamps, all_solar_wind = get_time_series_inference(df)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = FC_to_Conv().to(device)
model.load_state_dict(torch.load(r"OP_weight.pth", weights_only = True))
model.eval()

with torch.no_grad():
    x = torch.stack(input_data).unsqueeze(1).to(device)
    y_hat = model(x).detach().cpu().numpy()
    

image_mean = np.load(r'image_mean.npy')
image_std = np.load(r'image_std.npy')    
y_hat = (y_hat * image_std[None,:,None,None]) + image_mean[None,:,None,None]    





#%%


def plot_polar(ax, img, hemisphere='N', vmin=None, vmax=None):
    """
    img: (lat, lon) array
    hemisphere: 'N' or 'S'
    """
    nlat, nlon = img.shape

    theta = np.linspace(-np.pi, np.pi, nlon)
    r = np.linspace(0, 1, nlat)

    Theta, R = np.meshgrid(theta, r)

    if hemisphere == 'S':
        img = img[::-1]  # flip radial direction

    im = ax.pcolormesh(Theta, R, img, shading='auto',
                       vmin=vmin, vmax=vmax)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    return im







import matplotlib.gridspec as gridspec

from PIL import Image
import io
import numpy as np
labels = [r'$B_x$ (nT)', r'$B_y$ (nT)', r'$B_z$ (nT)', r'$V_{sw}$ (km/s)']

figures_pil = []   # <-- this is what you want
df = all_solar_wind
for b in tqdm(range(y_hat.shape[0])):

    t0 = df.datetime.iloc[b*5]
    t1 = df.datetime.iloc[b*5 + 240]
    
    fig = plt.figure(figsize=(30, 7))
    gs = gridspec.GridSpec(
        nrows=4, ncols=3,
        width_ratios=[1.5, 1, 1],
        wspace=0.25, hspace=0.1
    )
    
    # ---- Solar wind panels (left) ----
    for i, (col, lab) in enumerate(zip(df.columns[:-1], labels)):
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(df.datetime, df[col], lw=1.1)
        ax.axvspan(t0, t1, color='orange', alpha=0.25)
        ax.set_ylabel(lab)
        ax.grid(alpha=0.25)
    
        if i < 3:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Time (UTC)')
    
    # ---- Polar image panels (right) ----
    ax_nh = fig.add_subplot(gs[:, 1], projection='polar')
    ax_sh = fig.add_subplot(gs[:, 2], projection='polar')
    
    nh_img = y_hat[b, :8:2].sum(axis=0)
    sh_img = y_hat[b, 8::2].sum(axis=0)
    

    im1 = plot_polar(ax_nh, nh_img, hemisphere='N', vmin=0, vmax=4)
    ax_nh.set_title('Northern Hemisphere', pad=12)
    
    im2 = plot_polar(ax_sh, sh_img, hemisphere='S', vmin=0, vmax=4)
    ax_sh.set_title('Southern Hemisphere', pad=12)
    
    # Create separate colorbars for each
    cbar_nh = fig.colorbar(im1, ax=ax_nh, shrink=0.8)
    cbar_nh.set_label(r'ergs / cm$^2$')
    
    cbar_sh = fig.colorbar(im2, ax=ax_sh, shrink=0.8)
    cbar_sh.set_label(r'ergs / cm$^2$')
    fig.suptitle(
        f'Solar Wind Context and Model Predictions\n'
        f'{t1} → {timestamps[b]}',
        fontsize=14
    )

    # ---- Save figure to PIL Image ----
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=75, bbox_inches='tight')
    buf.seek(0)
    figures_pil.append(Image.open(buf).copy())
    buf.close()
    # plt.show()
    plt.close(fig)



from PIL import Image

# figures_pil : list of PIL.Image objects
gif_path = "solarwind_predictions.gif"

figures_pil[0].save(
    gif_path,
    save_all=True,
    append_images=figures_pil[1:],
    duration=33,   # ms per frame
    loop=0          # 0 = infinite loop
)

print(f"Saved GIF to {gif_path}")










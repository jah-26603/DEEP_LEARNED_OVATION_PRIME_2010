# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 19:07:56 2025

@author: JDawg
"""

import os
from tqdm import tqdm
import glob
import numpy as np
import requests
import re
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed



def download_solar_wind_data(out_dir = r'E:\solar_wind'):
    '''Downloads solar wind data from ACE.'''

    #downloads 1m resolution solar wind data 
    url = r'https://sohoftp.nascom.nasa.gov/sdb/goes/ace/daily/'
    page = requests.get(url).text
    
    
    file_types = ['swepam', 'mag']
    pattern = re.compile(r'href="([^"]*(?:swepam|mag).*?\.txt)"')
    files = [url + m for m in pattern.findall(page)]
    
    os.makedirs(out_dir, exist_ok=True)
    session = requests.Session()

    for f in tqdm(files, desc = 'Downloading solar wind data...'):
        fname = os.path.join(out_dir, os.path.basename(f))
    
        if os.path.exists(fname):   # skip existing
            continue
    
        with session.get(f, stream=True) as r:
            r.raise_for_status()
            with open(fname, 'wb') as fp:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        fp.write(chunk)
    print(f'Solar Wind Data: Complete Download from: {files[0][-24:-15]} to {files[-4][-24:-15]}')
      


#transform solar wind data
def collate_solar_wind(fp = r'E:\ml_aurora\solar_wind', out = r'E:\ml_aurora\organized_solar_wind.csv'):

    '''Creates time history of solar wind from ACE files. Can be called directly after 
    downloading solar wind data.'''
    mag_data = []
    vel_data = []
    
    for f in tqdm(glob.glob(os.path.join(fp, '*.txt')), desc = 'Loading in Solar Wind data...'):
        
    
        if 'mag' in f:
            with open(f, 'r') as fh:
                for i,line in enumerate(fh):
                    if i < 20:
                        continue
                    mag_data.append(line[:-2]) #ignore new line character
        else:
            
            with open(f, 'r') as fh:
                for i,line in enumerate(fh):
                    if i < 18:
                        continue
                    vel_data.append(line[:-2]) #ignore new line character
                    
    

    
    cols = ['year', 'month', 'day', 'hhmm','mjd', 'sec', 'mag_dqi', 'Bx', 'By', 'Bz','Bt', 'lat', 'lon']
    df = pd.DataFrame(mag_data)[0].str.split(expand=True).set_axis(cols, axis = 1)
    df = df[['year','month','day','hhmm','mag_dqi','Bx', 'By', 'Bz','Bt']]
    
    cols = ['year', 'month', 'day', 'hhmm','mjd', 'sec', 'vel_dqi', 'Np', 'vel', 'T_ion']
    vel_df = pd.DataFrame(vel_data)[0].str.split(expand=True).set_axis(cols, axis = 1)
    vel_df = vel_df[['year','month','day','hhmm','vel_dqi','vel', 'Np']]
    
    

    df = pd.merge(df, vel_df, on = ['year', 'month', 'day', 'hhmm'])
    df = df.astype(float)
    mask = ((df.vel_dqi == 9) 
              |(df.mag_dqi == 9) 
              |(df.vel_dqi == 9)
              |(df.vel <= -9999)
              |(df.Bz <= -999))

    cols = ['Bx', 'By', 'Bz', 'Bt', 'vel', 'Np']
    df.loc[mask, cols] = np.nan
    df['clock_ang'] = np.arctan2(df.By, df.Bz) #radians
    
    
    def build_datetime(df):
        hh = df['hhmm'] // 100
        mm = df['hhmm'] % 100
        return pd.to_datetime(
            dict(year=df.year, month=df.month, day=df.day,
                 hour=hh, minute=mm),
            errors='coerce'
        )
    
    df['time'] = build_datetime(df)
    time_index = pd.date_range(df.time.min(), df.time.max(), freq='1min')
    time_df = pd.DataFrame({'time': time_index})
    
    df = time_df.merge(df, on='time', how='left')
    df = df.drop_duplicates(subset='time', keep='first')
    df.to_csv(out)
    
#%% OVATION PRIME SPECIFIC DOWNLOAD FUNCTIONS

def OP_training_data(solar_wind_df = r"E:\ml_aurora\organized_solar_wind.csv", out = r'E:\ml_aurora\ovation_paired_data'):

    '''This is going to give runs for every 30 minutes of the solar wind data frame that you feed
    into this function. For about 11 years worth of solar wind data, this will return 165k label pairs.
    Each image is simply the 4 aurora per 2 hemisphere per 2 flux types (16 channels).
    
    This function will return the entire training dataset necessary to train the model. With images
    and input solar wind data being paired.'''
    

    from ovationpyme.ovation_prime import SeasonalFluxEstimator
    from ovationpyme.ovation_utilities import calc_avg_solarwind, calc_coupling
    from collections import OrderedDict
    from joblib import Parallel, delayed

    
    df = pd.read_csv(solar_wind_df)
    df['datetime'] = pd.to_datetime(df['time'])
    
    # ==========================
    # OVATION PYME PARAMETERS, DO NOT CHANGE
    # ==========================
    cadence = 0.5            # hours between each OP run
    prior_hours = 4
    prev_hour_weight = 0.65
    
    atypes = ['diff', 'mono', 'wave', 'ions'] #aurora types
    jtypes = ['energy', 'number'] # flux types
    
    
    os.makedirs(os.path.join(out, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out, 'swdata'), exist_ok=True)
    
    # ==========================
    # DIRECTORY SETUP
    # ==========================
    os.makedirs(os.path.join(out, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out, 'swdata'), exist_ok=True)
    
    # ==========================
    # SEASON FUNCTION
    # ==========================
    def doy_to_season(doy):
        if 80 <= doy <= 171:
            return 'spring'
        elif 172 <= doy <= 263:
            return 'summer'
        elif 264 <= doy <= 354:
            return 'fall'
        else:
            return 'winter'
    
    # ==========================
    # OPTIONAL: CACHE ESTIMATORS
    # ==========================
    SEASONS = ['spring', 'summer', 'fall', 'winter']
    ESTIMATORS = {
        (season, a, j): SeasonalFluxEstimator(season, a, j)
        for season in SEASONS
        for a in atypes
        for j in jtypes
    }
    
    # ==========================
    # ONE PARALLEL JOB
    # ==========================
    def process_one(i, df, out):
        
        swd = df.loc[(-prior_hours * 60 + (i + 1) * cadence * 60): (i + 1) * cadence * 60
        ]
    
        # datetime at ACE
        dt = swd.iloc[-1]['datetime']
    
        # L1 → bowshock propagation
        delta_t = 1.5e6 / np.nanmedian(swd['vel'].iloc[-35:].to_numpy()) 
        if np.isnan(delta_t) is True:# L1 → bowshock propagation
            delta_t = 3600
        bowshock_dt = dt + timedelta(minutes=delta_t // 60)
        
        ts = bowshock_dt.strftime('%Y%m%d_%H%M%S')
    
        img_fp = os.path.join(out, 'images', f'{ts}.npy')
        sw_fp  = os.path.join(out, 'swdata', f'{ts}.npy')
        
        if os.path.exists(img_fp) and os.path.exists(sw_fp):
            return   # skip this run
        
        
        # Remove last row (used for timestamp)
        swd = swd.iloc[:-1]
    
        Ec = calc_coupling(
            swd['Bx'].to_numpy(copy=True),
            swd['By'].to_numpy(copy=True),
            swd['Bz'].to_numpy(copy=True),
            swd['vel'].to_numpy(copy=True),
        )
        
        swd = swd.copy()        # make slice writable
        swd['Ec'] = Ec
    
        # Hourly averages
        group_idx = np.arange(len(swd)) // 60
        sw4avg = swd.groupby(group_idx).mean(numeric_only=True)
    
        weights = np.array([
            prev_hour_weight ** p for p in range(prior_hours)[::-1]
        ])
        wsum = weights.sum()
    
        avgsw = OrderedDict(
            Bx=np.nansum(sw4avg['Bx'] * weights) / wsum,
            By=np.nansum(sw4avg['By'] * weights) / wsum,
            Bz=np.nansum(sw4avg['Bz'] * weights) / wsum,
            V =np.nansum(sw4avg['vel'] * weights) / wsum,
            Ec=np.nansum(sw4avg['Ec'] * weights) / wsum,
        )
    
        # Seasons
        doy = bowshock_dt.timetuple().tm_yday
        seasonN = doy_to_season(doy)
        seasonS = doy_to_season((doy + 183) % 366)
    
        # ==========================
        # FLUX IMAGES
        # ==========================
        imgs = []
        #determine whether im grabbing the correct flux grids
        for a in atypes:
            for j in jtypes:
                estimatorN = ESTIMATORS[(seasonN, a, j)]
                fluxN = estimatorN.get_gridded_flux(
                    avgsw['Ec'], combined_N_and_S=False
                )[2]
                imgs.append(fluxN)
    
        for a in atypes:
            for j in jtypes:
                estimatorS = ESTIMATORS[(seasonS, a, j)]
                fluxS = estimatorS.get_gridded_flux(
                    avgsw['Ec'], combined_N_and_S=False
                )[5]
                imgs.append(fluxS)
    
        imgs = np.stack(imgs)
    
        # ==========================
        # SAVE
        # ==========================
        ts = bowshock_dt.strftime('%Y%m%d_%H%M%S')
        np.save(
            os.path.join(out, 'images', f'{ts}.npy'),
            imgs
        )
        input_data = swd[['Bx', 'By', 'Bz', 'vel']].to_numpy()
        input_data = np.hstack([input_data, np.array([doy] * len(input_data))[:,None]])
        np.save(
            os.path.join(out, 'swdata', f'{ts}.npy'),
            input_data
            
        )
    
    N = int((len(df)/30))
    
    Parallel(
        n_jobs=os.cpu_count() - 1,   # leave one core free
        backend='loky',
        verbose=10
    )(
        delayed(process_one)(i, df, out)
        for i in range(8,N)
    )
    
    

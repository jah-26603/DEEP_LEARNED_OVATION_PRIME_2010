# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 10:32:23 2026

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
from torch.utils.data import Dataset, DataLoader
import torch
from scipy.ndimage import median_filter

import torch
import torch.nn as nn

class FC_to_Conv(nn.Module):
    def __init__(self, num_in = 240*5, c_out = 16 ):
        super().__init__()
        p = 0.5
        
        # Less aggressive compression: 240×5 → 20×24
        self.fc = nn.Sequential(
            nn.Linear(num_in, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            
            # nn.Linear(1024, 128 * 20 ),  # Larger spatial resolution
            # nn.BatchNorm1d(128 * 20),
            # nn.ReLU(inplace=True),
            
            nn.Linear(1024, 128 * 20 * 24),  # Larger spatial resolution
            nn.BatchNorm1d(128 * 20 * 24),
            nn.ReLU(inplace=True),
        )
        
        # Simpler architecture with less downsampling
        self.enc1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)  # 20×24 → 10×12
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(64, 64, 4, 2, 1)  # 10×12 → 20×24
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1),  # skip connection
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.up2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)  # 20×24 → 40×48
        self.dec2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        self.up3 = nn.ConvTranspose2d(32, 16, 4, 2, 1)  # 40×48 → 80×96
        self.final = nn.Conv2d(16, c_out, 3, padding=1)
        
    def forward(self, x):
        B = x.size(0)
        x = x.view(B, -1)
        x = self.fc(x)
        x = x.view(B, 128, 20, 24)
        
        # Encoder
        e1 = self.enc1(x)  # 64 @ 20×24
        p1 = self.pool1(e1)  # 64 @ 10×12
        e2 = self.enc2(p1)  # 64 @ 10×12
        
        # Decoder
        d1 = self.up1(e2)  # 64 @ 20×24
        d1 = torch.cat([d1, e1], dim=1)  # 128 @ 20×24
        d1 = self.dec1(d1)  # 64 @ 20×24
        
        d2 = self.up2(d1)  # 32 @ 40×48
        d2 = self.dec2(d2)  # 32 @ 40×48
        
        d3 = self.up3(d2)  # 16 @ 80×96
        d3 = self.final(d3)
        
        return d3
    
#%%
#

import torch
import torch.nn as nn
import torch.nn.functional as F

class FC_to_VAE(nn.Module):
    def __init__(self, num_in=240*5, c_out=2, latent_dim=256):
        super().__init__()
        p = 0.5

        # ---------- FC stem ----------
        self.fc = nn.Sequential(
            nn.Linear(num_in, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p),

            nn.Linear(1024, 128 * 20 * 24),
            nn.BatchNorm1d(128 * 20 * 24),
            nn.ReLU(inplace=True),
        )

        # ---------- Encoder ----------
        self.enc1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)  # 20×24 → 10×12

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        enc_out_dim = 64 * 10 * 12

        # ---------- VAE bottleneck ----------
        self.fc_mu     = nn.Linear(enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_out_dim, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, enc_out_dim)

        # ---------- Decoder ----------
        self.up1 = nn.ConvTranspose2d(64, 64, 4, 2, 1)  # 10×12 → 20×24
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)  # 20×24 → 40×48
        self.dec2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.ConvTranspose2d(32, 16, 4, 2, 1)  # 40×48 → 80×96
        self.final = nn.Conv2d(16, c_out, 3, padding=1)

    # ---------- Reparameterization trick ----------
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        B = x.size(0)

        # FC stem
        x = x.view(B, -1)
        x = self.fc(x)
        x = x.view(B, 128, 20, 24)

        # Encoder
        e1 = self.enc1(x)          # 64 @ 20×24
        p1 = self.pool1(e1)        # 64 @ 10×12
        e2 = self.enc2(p1)         # 64 @ 10×12

        # VAE bottleneck
        h = e2.view(B, -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        z = self.reparameterize(mu, logvar)

        # Decode latent
        h_dec = self.fc_decode(z)
        h_dec = h_dec.view(B, 64, 10, 12)

        # Decoder
        d1 = self.up1(h_dec)               # 64 @ 20×24
        d1 = torch.cat([d1, e1], dim=1)    # skip
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = self.dec2(d2)

        d3 = self.up3(d2)
        out = self.final(d3)

        return out, mu, logvar

    def encode(self, x):
        B = x.size(0)
        x = self.fc(x.view(B, -1))
        x = x.view(B, 128, 20, 24)
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        h = e2.view(B, -1)
        return self.fc_mu(h), self.fc_logvar(h), e1

    def decode(self, z, e1):
        B = z.size(0)
        h = self.fc_decode(z).view(B, 64, 10, 12)
        d1 = self.up1(h)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        d2 = self.up2(d1)
        d2 = self.dec2(d2)
        d3 = self.up3(d2)
        return self.final(d3)

    
    
#%%
#Diffusion based architecture


class Conv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.Conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                                  nn.BatchNorm2d(out_channels),
                                  nn.GELU())


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.Conv(x)


class DownConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.Conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 2, 1),
                                nn.BatchNorm2d(out_channels),
                                nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.Conv(x)

class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.Conv = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
                                nn.BatchNorm2d(out_channels),
                                nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.Conv(x)


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.Sequential(nn.AvgPool2d(7),
                                  nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


class Unflatten(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.Conv = nn.Sequential(nn.ConvTranspose2d(in_channels, in_channels, 7, 7, 0),
                        nn.BatchNorm2d(in_channels),
                        nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.Conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv_block = nn.Sequential(Conv(in_channels, out_channels),
                                        Conv(out_channels, out_channels))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)



class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_block = nn.Sequential(DownConv(in_channels, out_channels),
                                        ConvBlock(out_channels, out_channels))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_block = nn.Sequential(UpConv(in_channels, out_channels),
                                        ConvBlock(out_channels, out_channels))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)
    
    
class FCBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.model = nn.Sequential(
                                    nn.Linear(in_channels, out_channels),
                                    nn.GELU(),
                                    nn.Linear(out_channels, out_channels)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
class DeepFCBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.model = nn.Sequential(
                                    nn.Linear(in_channels, out_channels),
                                    nn.GELU(),
                                    nn.Linear(out_channels, out_channels),
                                    nn.GELU(),
                                    nn.Linear(out_channels, out_channels)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class ClassConditionalUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_hiddens: int
    ):
        super().__init__()
        self.num_hiddens = num_hiddens

        self.x1 = ConvBlock(in_channels, num_hiddens)
        self.x2 = DownBlock(num_hiddens, num_hiddens)
        self.x3 = DownBlock(num_hiddens, num_hiddens*2)
        self.x4 = Flatten()
        self.x5 = Unflatten(num_hiddens*2)

        self.x6 = UpBlock(num_hiddens*4, num_hiddens)
        self.x7 = UpBlock(num_hiddens*2, num_hiddens)
        self.x8 = ConvBlock(num_hiddens*2, num_hiddens)
        self.x9 = nn.Conv2d(num_hiddens, in_channels, 3, 1, 1)

        self.fc1 = FCBlock(1, self.num_hiddens*2)
        self.fc2 = FCBlock(1, self.num_hiddens)

        #conditoning the image generation
        self.class_fc_a1 = DeepFCBlock(num_classes, self.num_hiddens*4)
        self.class_fc_a2 = DeepFCBlock(self.num_hiddens*4, self.num_hiddens*2)
        
        self.class_fc_b1 = DeepFCBlock(num_classes, self.num_hiddens*2)
        self.class_fc_b2 = DeepFCBlock(self.num_hiddens*2, self.num_hiddens)
    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (N, C, H, W) input tensor.
            c: (N,) int64 condition tensor.
            t: (N,) normalized time tensor.
            mask: (N,) mask tensor. If not None, mask out condition when mask == 0.

        Returns:
            (N, C, H, W) output tensor.
        """

        # assert x.shape[-2:] == (100, 96), "Expect input shape to be (28, 28)."
        x1 = self.x1(x)
        x2 = self.x2(x1)
        x3 = self.x3(x2)
        x4 = self.x4(x3)
        x5 = self.x5(x4)

        #these may need to be normalized, move to self
        t1 = self.fc1(t)[:,:,None,None]
        t2 = self.fc2(t)[:,:,None,None]

        
        dc = c.reshape(x.shape[0],-1).to('cuda')
        if self.training:
            p_drop = 0.1
            mask = (torch.rand(x.shape[0], device=x.device) > p_drop).float()
        else:
            mask = torch.ones(x.shape[0], device=x.device)
        
        dc = dc * mask[:, None]
            
        c1 = self.class_fc_a2(self.class_fc_a1(dc))[:,:,None,None]
        c2 = self.class_fc_b2(self.class_fc_b1(dc))[:,:,None,None]


        #differential padding for the unflattening section...
        diff_h = x3.shape[-2] - x5.shape[-2]
        diff_w = x3.shape[-1] - x5.shape[-1]
        x5 = torch.nn.functional.pad(x5, (diff_w//2, diff_w - diff_w//2, 
                                           diff_h//2, diff_h - diff_h//2))
        
        
        x6 = torch.concatenate([x3, c1 * x5 + t1], dim = 1) 

        x7 = torch.concatenate([x2, c2 * self.x6(x6) + t2], dim = 1) 
        x8 = torch.concatenate([x1, self.x7(x7)], dim = 1)
        x9 = self.x8(x8)
        xf = self.x9(x9)

        return xf
    
    
    

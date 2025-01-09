import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

class Generator(nn.Module):
    def __init__(self, nz, ngf):
        super().__init__()

        # generator network for generating images (64x64 pixels)
        self.network = nn.Sequential(
            # neccesary for 256x256 pixels
            #nn.ConvTranspose2d(nz, ngf*32, 4, 1, 0),
            #nn.BatchNorm2d(ngf*32),
            #nn.ReLU(True),

            #nn.ConvTranspose2d(ngf*32, ngf*16, 4, 2, 1),
            #nn.BatchNorm2d(ngf*16),
            #nn.ReLU(True),

            nn.ConvTranspose2d(nz, ngf*8, 4, 2, 0),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.network(x)
    
class Discriminator(nn.Module):
    def __init__(self, ndf):
        super().__init__()

        # discriminator network to evalueate whether the images are real or fake
        self.network = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf*2, 4, 2, 1),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # neccesary for 256x256 pixels
            #nn.Conv2d(ndf*8, ndf*16, 4, 2, 1),
            #nn.BatchNorm2d(ndf*16),
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout(0.3),

            #nn.Conv2d(ndf*16, ndf*32, 4, 2, 1),
            #nn.BatchNorm2d(ndf*32),
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout(0.3),

            nn.Conv2d(ndf*8, 1, 4, 1, 0)   #ndf*32 for 256x256 pixels
        )
    
    def forward(self, x):
        return self.network(x)
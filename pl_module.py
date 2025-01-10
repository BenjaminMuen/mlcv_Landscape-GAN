import mlflow

import pytorch_lightning as pl

import torch
import torchvision

from torch import nn
from torch.nn import functional as F

class PL_Module(pl.LightningModule):
    def __init__(self, generator, discriminator, lr, betas, nz, n_valid):
        super().__init__()

        self.automatic_optimization = False

        self.generator = generator
        self.discriminator = discriminator

        self.lr = lr
        self.betas = betas

        self.nz = nz

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.n_valid = n_valid

        # create a fixed set of validation samples
        self.z_valid = torch.randn((self.n_valid, self.nz, 1, 1), dtype=self.dtype, device=self.device)

    def training_step(self, batch):
        real_images = batch
        
        optim_gen, optim_dis = self.optimizers()

        # Generate noise vector z
        z = torch.randn((real_images.shape[0], self.nz, 1, 1), dtype=self.dtype)
        z = z.type_as(real_images)

        
        # train discriminator
        self.toggle_optimizer(optim_dis)

        gen_images = self.generator(z).detach()

        real_labels = torch.ones((real_images.size(0), 1), dtype=self.dtype, device=self.device)
        real_labels = real_labels.type_as(real_images)

        real_preds = self.discriminator(real_images)
        real_loss = self.loss_fn(real_preds, real_labels)

        fake_labels = torch.zeros((real_images.size(0), 1), dtype=self.dtype, device=self.device)
        fake_labels = fake_labels.type_as(real_images)

        fake_preds = self.discriminator(gen_images)
        fake_loss = self.loss_fn(fake_preds, fake_labels)

        # combined loss
        dis_loss = (real_loss + fake_loss) / 2

        # mlflow logging
        self.log('train_step/dis_loss', dis_loss, prog_bar=True)

        # optimizers
        optim_dis.zero_grad()
        self.manual_backward(dis_loss)
        optim_dis.step()

        self.untoggle_optimizer(optim_dis)

        # train generator
        self.toggle_optimizer(optim_gen)

        gen_images = self.generator(z)

        gen_labels = torch.ones((real_images.size(0), 1), dtype=self.dtype, device=self.device)
        gen_labels = gen_labels.type_as(real_images)

        gen_preds = self.discriminator(gen_images)

        # generator loss
        gen_loss = self.loss_fn(gen_preds, gen_labels)

        # mlflow logging
        self.log('train_step/gen_loss', gen_loss, prog_bar=True)

        # optimizers
        optim_gen.zero_grad()
        self.manual_backward(gen_loss)
        optim_gen.step()

        self.untoggle_optimizer(optim_gen)

    def on_train_epoch_end(self):
        # generate validation images
        gen_images = self(self.z_valid.to(self.device))

        image_grid = torchvision.utils.make_grid(gen_images, nrow=4, normalize=True).swapaxes(0, 2).swapaxes(0, 1)

        mlflow.log_image(image_grid.detach().cpu().numpy(), f"generated_images/{self.current_epoch}.png")
    
    def configure_optimizers(self):
        optim_gen = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=self.betas)
        optim_dis = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=self.betas)

        return [optim_gen, optim_dis], []
    
    def forward(self, z):
        return self.generator(z)

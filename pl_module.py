import mlflow

import pytorch_lightning as pl

import torch
import torchvision

from torch import nn
from torch.nn import functional as F

class PL_Module(pl.LightningModule):
    def __init__(self, generator, critic, lr, betas, lambda_gp, nz, n_valid):
        super().__init__()

        self.generator = generator
        self.critic = critic

        self.lr = lr
        self.betas = betas

        self.lambda_gp = lambda_gp

        self.nz = nz

        self.n_valid = n_valid

        # create a fixed set of validation samples
        self.z_valid = torch.randn((self.n_valid, self.nz, 1, 1), dtype=self.dtype, device=self.device)

    def training_step(self, batch):
        real_images = batch

        optim_gen, optim_crit = self.optimizers()

        # create a random noise vector
        z = torch.randn((real_images.size(0), self.nz, 1, 1), dtype=self.dtype, device=self.device)

        # train the critic
        self.toggle_optimizer(optim_crit)

        gen_images = self.generator(z).detach()

        real_scores = self.critic(real_images)
        fake_scores = self.critic(gen_images)

        # wasserstein loss
        wasserstein_loss = fake_scores.mean() - real_scores.mean()

        # gradient penalty
        gp = self.gradient_penalty(real_images, gen_images)

        # combined loss
        critic_loss = wasserstein_loss + self.lambda_gp * gp

        # mlflow logging
        self.log('train_step/crit_loss', critic_loss, prog_bar=True)

        # optimizers
        optim_crit.zero_grad()
        self.manual_backward(critic_loss)
        optim_crit.step()

        self.untoggle_optimizer(optim_crit)

        # train the generator
        self.toggle_optimizer(optim_gen)

        gen_images = self.generator(z)

        fake_scores = self.critic(gen_images)

        # generator loss
        generator_loss = -fake_scores.mean()

        # optimizers
        optim_gen.zero_grad()
        self.manual_backward(generator_loss)
        optim_gen.step()

        # mlflow logging
        self.log('train_step/gen_loss', generator_loss, prog_bar=True)

        self.untoggle_optimizer(optim_gen)

    def gradient_penalty(self, real_images, gen_images):
        batch_size = real_images.size(0)

        alpha = torch.rand((batch_size, 1, 1, 1), dtype=self.dtype, device=self.device)
        diff = gen_images - real_images

        interpolated = real_images + alpha * diff
        interpolated.requires_grad_(True)

        interpolated_scores = self.critic(interpolated)

        gradients = torch.autograd.grad(
            outputs=interpolated_scores,
            inputs=interpolated,
            grad_outputs=torch.ones_like(interpolated_scores),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradients_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradients_norm - 1) ** 2).mean()

        return gradient_penalty

    def on_train_epoch_end(self):
        # generate validation images
        gen_images = self(self.z_valid.to(self.device))

        image_grid = torchvision.utils.make_grid(gen_images, nrow=4, normalize=True).swapaxes(0, 2).swapaxes(0, 1)

        mlflow.log_image(image_grid.detach().cpu().numpy(), f"generated_images/{self.current_epoch}.png")
    
    def configure_optimizers(self):
        optim_gen = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=self.betas)
        optim_crit = torch.optim.Adam(self.critic.parameters(), lr=self.lr, betas=self.betas)

        return [optim_gen, optim_crit], []
    
    def forward(self, z):
        return self.generator(z)


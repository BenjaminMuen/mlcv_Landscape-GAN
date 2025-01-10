import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import pytorch_lightning as pl
import mlflow

from diffusers import DDPMScheduler, UNet2DModel, DDPMPipeline


class PL_Module(pl.LightningModule):
    def __init__(self, model: UNet2DModel=None, noise_scheduler: DDPMScheduler=None, lr=None, betas=None, num_valid=None, num_inference_timesteps=None):
        super().__init__()
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.lr = lr
        self.betas = betas
        self.num_valid = num_valid
        self.num_inference_timesteps = num_inference_timesteps


    def training_step(self, batch):
        if isinstance(batch, tuple) or isinstance(batch, list):
            clean_img = batch[0]
        else:
            clean_img = batch

        # Forward step of diffusion
        batch_size = clean_img.shape[0]
        noise = torch.randn_like(clean_img)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (batch_size, ), device=clean_img.device).long()
        noisy_img = self.noise_scheduler.add_noise(clean_img, noise, timesteps)

        # Predict noise and calculate loss
        noise_pred = self.model(noisy_img, timesteps).sample
        loss = F.mse_loss(noise_pred, noise)
        self.log("train_loss", loss, prog_bar=True)

        return loss
                

    def on_train_epoch_end(self):
        if (self.current_epoch + 1) % 10 == 0:
            # Setup inference pipeline
            pipeline = DDPMPipeline(self.model, self.noise_scheduler)
            generator = torch.Generator(device=pipeline.device).manual_seed(0)

            # Generate images
            images = pipeline(self.num_valid, generator, self.num_inference_timesteps, "numpy").images

            # Log images
            img_grid = torchvision.utils.make_grid(torch.from_numpy(images.transpose((0, 3, 1, 2))), nrow=4, normalize=True).swapaxes(0, 2).swapaxes(0, 1)
            mlflow.log_image(img_grid.detach().cpu().numpy(), artifact_file=f"generated_images/{self.current_epoch}_grid.png")
    

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=self.betas)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.95)
        return [optim], [lr_scheduler]
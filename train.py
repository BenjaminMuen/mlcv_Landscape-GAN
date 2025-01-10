import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch
from torch import nn
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from torchvision import datasets, transforms

from diffusers import UNet2DModel, DDPMScheduler

from pl_module import PL_Module
from dataloader import get_dataloader


torch.set_float32_matmul_precision("medium")



if __name__ == "__main__":
    mlflow.pytorch.autolog(
        checkpoint_save_best_only=False,
        checkpoint_save_weights_only=True
    )
    mlflow.set_experiment("Diffuser")
    mlflow.start_run(run_name="DDPM_UNET")

    # Parameters
    lr = 1e-4
    batch_size = 64
    epochs = 100
    workers = 6
    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_train_timesteps = 1000
    beta_schedule = "linear"
    nc = 3

    b1 = 0.95
    b2 = 0.999

    num_inference_timesteps = 1000
    num_valid = 16

    data_path = "./data/train/dataset"
    image_size = (64, 64)


    # DataLoader
    train_dataloader: DataLoader = get_dataloader(data_path, batch_size, workers)


    # Log params
    mlflow.log_param("data_path", data_path)
    mlflow.log_param("length_data", len(train_dataloader.dataset))
    mlflow.log_param("image_size", image_size)
    mlflow.log_param("nc", nc)
    mlflow.log_param("num_train_timesteps", num_train_timesteps)
    mlflow.log_param("beta_schedule", beta_schedule)
    mlflow.log_param("b1", b1)
    mlflow.log_param("b2", b2)
    mlflow.log_param("num_valid", num_valid)

    
    # Load model
    model = UNet2DModel(
        sample_size=image_size,
        in_channels=nc,
        out_channels=nc,
        layers_per_block=2,
        block_out_channels=(128, 128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule=beta_schedule,
    )
    module = PL_Module(
        model=model,
        noise_scheduler=noise_scheduler,
        lr = lr,
        betas=(b1, b2),
        num_valid=num_valid,
        num_inference_timesteps=num_inference_timesteps,
    )


    # Callbacks
    callbacks = []
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)


    # Create training
    trainer = pl.Trainer(
        accelerator = device,
        devices = 1,
        max_epochs = epochs,
        precision = "bf16",
        logger=True,
        callbacks=callbacks,
    )

    
    # Train
    trainer.fit(module, train_dataloaders=train_dataloader)
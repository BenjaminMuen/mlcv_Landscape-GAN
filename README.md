# Landscape Image Generation Using GANs and Diffusion Models

This repository contains implementations of multiple approaches for generating high-quality landscape images using various Generative Adversarial Networks (GANs) and a Diffusion Model.

---

## Branches
- **`main`**: Implementation of a Diffusion Model.
- **`DCGAN`**: Implementation of a Deep Convolutional GAN (DCGAN).
- **`WGAN`**: Implementation of a Wasserstein GAN with Gradient Penalty and Feature Matching Loss.

All models are trained on the **Landscapes HQ (LHQ) 1024×1024** dataset, which contains **90,000 images** of diverse landscapes.

---

## Table of Contents
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Training](#training)
- [How to Use](#how-to-use)
- [Samples](#samples)
- [Acknowledgments](#acknowledgments)

---

## Project Structure

```plaintext
.
├── data/           # folder containing the dataset
├── dataloader.py   # module containing the dataset and dataloader
├── model.py        # module containing the model architecture
├── pl_module.py    # module containing the train-logic (trainer)
└── train.ipynb     # jupyter notebook containing the settings and hyperparameters for training
```

---

# Dataset
The Landscapes HQ (LHQ) dataset, originally created by Ivan Skorokhodov, Grigorii Sotnikov and Mohamed Elhoseiny, is used to train models.
It consists of 90.000 high-resolution landscape images at a resolution of 1024x1024 pixels.

- **Features:** The dataset includes a wide range of landscapes, such as mountains, forests, beaches and more.
- **Augumentation:** Images are resized and cropped to a resolution of 256x256 pixels.
- **Source:** The dataset is available [here](https://github.com/universome/alis)

To use the dataset: 
1. Download it and place the images in the **`./data/`** directory.
2. The folder structure should be as followed:

([Kaggle](https://www.kaggle.com/datasets/dimensi0n/lhq-1024))
```plaintext
./data/
└── train/
    └── dataset/
        ├── 00000.jpg
        ├── 00001.jpg
        └── ...
```

---

# Training
Each branch includes a dedicated **`train.ipynb`** notebook for training the respective model.


### Hyperparameters

#### General
| Parameter      | Description                              | Default       |
|----------------|------------------------------------------|---------------|
| `n_workers`    | Number of Threads for loading the data   | 4             |       
| `epochs`       | Number of training epochs                | 100           |
| `batch_size`   | Batch size for training                  | 16            |
| `lr`           | Learning rate                            | 0.0002        |
| `betas`        | Adam optimizer betas                     | (0.5, 0.999)  |
| `image_size`   | Resolution of generated images           | (256, 256)    |
| `num_valid`    | Number of images for validation          | 16            |

#### Diffusion Model specific
| Parameter      | Description                              | Default       |
|----------------|------------------------------------------|---------------|

#### DCGAN specific
| Parameter      | Description                              | Default       |
|----------------|------------------------------------------|---------------|
| `nz`           | Length of the input noise vector         | 100           |
| `ngf`          | Feature map size for the generator       | 64            |
| `ndf`          | Feature map size for the discriminator   | 64            |

#### WGAN specific (DCGAN + ...)
| Parameter      | Description                              | Default       |
|----------------|------------------------------------------|---------------|
| `lambda_gp`    | Factor to weigh gradient penalty         | 10            |

---

TODO:

# How To Use
## Train the Models
## Generate Images
## Pre-trained Models

---

# Samples
TODO

---

# Acknowledgments

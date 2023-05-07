# AstroNoiseNet
This is a Neural Network designed to denoise Astro images. The architecture of the Neural Network itself is based on 
[PRIDNet](https://github.com/491506870/PRIDNet). Additionally, inspired by [Starnet V1](https://github.com/nekitmm/starnet), 
a discriminator is used in training to retain high-frequency details.

For training data, pairs of real astro images with short and longer integration times are used.

## Installation for Developers
Clone repository
```
git clone https://github.com/Steffenhir/AstroNoiseNet
cd AstroNoiseNet
```

Create new venv and install required packages
```
conda create --name AstroNoiseNet python=3.8.15
conda activate AstroNoiseNet
conda install --file requirements.txt
```

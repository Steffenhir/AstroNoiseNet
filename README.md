# AstroNoiseNet
This is a Neural Network designed to denoise Astro images. The architecture of the Neural Network itself is based on 
[PRIDNet](https://github.com/491506870/PRIDNet). Additionally, inspired by [Starnet V1](https://github.com/nekitmm/starnet), 
a discriminator is used in training to retain high-frequency details.

For training data, pairs of real astro images with short and longer integration times are used.

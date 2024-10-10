# PyTorch Implementation of Original U-Net Model

This repo implements the original U-Net model from "U-Net: Convolutional Networks 
for Biomedical Image Segmentation" in Pytorch. This implementation was mainly
for learning purposes.  

The model is implemented in the class UNet. Additionally, I coded a smaller version
of the U-Net architecture that also allows for images of any size. The original
architecture is quite restrictive regarding image size where it really only expects
images of size 1 x 572 x 572 (C x H x W).
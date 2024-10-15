# PyTorch Implementation of Original U-Net Model

This repo implements the original U-Net model from "U-Net: Convolutional Networks 
for Biomedical Image Segmentation" in Pytorch. This implementation was mainly
for learning purposes.  

NOTE: There are 2 slight differences between this model and the original:

1. I added padding=1 to each convolution layer making it easier to use with
more common image sizes (i.e. 256 or 224).
2. I added the option to either use: 1) use a ConvTranspose layer or 2) use 
UpSample with a 1x1 Conv layer to increase the size of the features.
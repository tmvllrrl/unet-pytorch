# PyTorch Implementation of Original U-Net Model

This repo implements the original U-Net model from "U-Net: Convolutional Networks 
for Biomedical Image Segmentation" in Pytorch. This implementation was mainly
for learning purposes. The model is located in models.py.

NOTE: There are 2 slight differences between this model and the original:

1. I added padding=1 to each convolution layer making it easier to use with
more common image sizes (i.e. 256 or 224).
2. I added the option to either use: 1) use a ConvTranspose layer or 2) use 
UpSample with a 1x1 Conv layer to increase the size of the features.

Additionally, I have implemented a ResNet-18-based U-Net where the first half (or the encoder) is a ResNet-18 model. This model is also located in models.py.

I have included a file, seg_trainer.py, to train a segmentation model using either the original U-Net or the ResNet-18 U-Net. There are config classes within seg_trainer.py that one can change to meet their needs. Additionally, I have written two dataset classes to be instantiated within the config classes. The first is called SegDataset, which is just a standard dataset to be used for training segmentation models. SegDataset expects your images to be RGB images and your masks to be grayscale/binary images. SegDataset also expects you to pass a CSV file containing the names of your images where the names for the images and the masks are an exact match, but are stored in separate directory. I have also included a dataset called CarvanaDataset to be used with the Carvana Kaggle dataset located [here](https://www.kaggle.com/datasets/ipythonx/carvana-image-masking-png).
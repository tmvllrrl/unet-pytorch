import torch
import torch.nn as nn
import torchvision.transforms as transforms

class CenterCrop(nn.Module):
    def __init__(self, size):
        super().__init__()

        self.crop = transforms.CenterCrop(size)
    
    def forward(self, x):
        return self.crop(x)
    
class UNetConvBlock(nn.Module):
    def __init__(self, crop_size, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.crop = CenterCrop(crop_size)

    def forward(self, x):
        x = self.block(x)
        crop = self.crop(x)

        return x, crop
    
class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_mode="upconv"):
        super().__init__()

        if up_mode == "upconv":
            self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=2, stride=2)
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
            )

        self.block = UNetConvBlock(crop_size=64, in_channels=in_channels,
                                   out_channels=out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        x, _ = self.block(x)

        return x

class UNet(nn.Module):
    def __init__(self, img_size, num_channel=1, first_out=64, up_mode="deconv"):
        super().__init__()

        second_out = first_out * 2
        third_out = second_out * 2
        fourth_out = third_out * 2
        fifth_out = fourth_out * 2

        # Reused layers
        self.relu = nn.ReLU()
        self.down_sample = nn.MaxPool2d(kernel_size=(2,2))

        # Contracting Path
        self.down1 = UNetConvBlock(crop_size=img_size, in_channels=num_channel,
                                   out_channels=first_out)
        self.down2 = UNetConvBlock(crop_size=img_size // 2, in_channels=first_out,
                                   out_channels=second_out)
        self.down3 = UNetConvBlock(crop_size=img_size // 4, in_channels=second_out,
                                   out_channels=third_out)
        self.down4 = UNetConvBlock(crop_size=img_size // 8, in_channels=third_out,
                                   out_channels=fourth_out)
        self.down5 = UNetConvBlock(crop_size=img_size // 16, in_channels=fourth_out,
                                   out_channels=fifth_out)
        
        # Expansive Path
        self.up1 = UNetUpBlock(in_channels=fifth_out, out_channels=fourth_out, up_mode=up_mode)
        self.up2 = UNetUpBlock(in_channels=fourth_out, out_channels=third_out, up_mode=up_mode)
        self.up3 = UNetUpBlock(in_channels=third_out, out_channels=second_out, up_mode=up_mode)
        self.up4 = UNetUpBlock(in_channels=second_out, out_channels=first_out, up_mode=up_mode)

        # There are 23 convolutions total in original model
        self.conv23 = nn.Conv2d(in_channels=first_out, out_channels=num_channel, kernel_size=1)

    def forward(self, x):
        # Contracting Path
        skip = []

        x, crop = self.down1(x)
        skip.append(crop)
        x = self.down_sample(x)
        
        x, crop = self.down2(x)
        skip.append(crop)
        x = self.down_sample(x)
        
        x, crop = self.down3(x)
        skip.append(crop)
        x = self.down_sample(x)
        
        x, crop = self.down4(x)
        skip.append(crop)

        # Point where the features have lowest dimension
        z = self.down_sample(x)
        z_flat = z.view(z.shape[0], -1)

        x, _ = self.down5(z)

        # Expansive Path
        x = self.up1(x, skip[3])
        x = self.up2(x, skip[2])
        x = self.up3(x, skip[1])
        x = self.up4(x, skip[0])

        x = self.conv23(x)
    
        return x, z, z_flat
    
# img_size = 224
# model = UNet(img_size=img_size, num_channel=1, first_out=2, up_mode="upsamp")
# print(model)
# dummy = torch.randn((2, 1, img_size, img_size))
# x, z, z_flat = model(dummy)
# print(x.shape, z.shape, z_flat.shape)

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torchvision.models import resnet18


'''
    Ordinary U-Net Model
'''
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

        # Crop size doesn't matter for upsampling portion of U-Net
        self.block = UNetConvBlock(crop_size=64, in_channels=in_channels,
                                   out_channels=out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        x, _ = self.block(x)

        return x


class UNet(nn.Module):
    def __init__(
            self, 
            img_size: int, 
            input_ch: int = 3, 
            output_ch: int = 1, 
            first_out: int = 2, 
            up_mode: str = "deconv"
        ) -> None:
        super().__init__()

        second_out = first_out * 2
        third_out = second_out * 2
        fourth_out = third_out * 2
        fifth_out = fourth_out * 2

        # Reused layers
        self.relu = nn.ReLU()
        self.down_sample = nn.MaxPool2d(kernel_size=(2,2))

        # Contracting Path
        self.down1 = UNetConvBlock(crop_size=img_size, in_channels=input_ch,
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
        self.conv23 = nn.Conv2d(in_channels=first_out, out_channels=output_ch, kernel_size=1)

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
    
        return x #, z, z_flat    


'''
    ResNet-18 U-Net Model
'''
class CenterCropRN(nn.Module):
    def __init__(self, size):
        super().__init__()

        self.crop = transforms.CenterCrop(size)
    
    def forward(self, x):
        return self.crop(x)
    

class UNetConvBlockRN(nn.Module):
    def __init__(self, crop_size, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.crop = CenterCropRN(crop_size)

    def forward(self, x):
        x = self.block(x)
        crop = self.crop(x)

        return x, crop
    

class UNetUpBlockRN(nn.Module):
    def __init__(self, up_in, up_out, block_in, block_out, up_mode="upconv"):
        super().__init__()            

        if up_mode == "upconv":
            self.up = nn.ConvTranspose2d(in_channels=up_in, out_channels=up_out,
                                    kernel_size=2, stride=2)
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(in_channels=up_in, out_channels=up_out, kernel_size=1)
            )

        # Crop size doesn't matter for upsampling portion of U-Net
        self.block = UNetConvBlockRN(crop_size=64, in_channels=block_in,
                                   out_channels=block_out)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        x, _ = self.block(x)

        return x


class ResNet18UNet(nn.Module):
    def __init__(self, input_ch: int = 3, target_ch: int = 3, up_mode: str = "deconv"):
        super().__init__()

        self.rn18_model = resnet18(weights=None)

        self.down1 = nn.Sequential(
            self.rn18_model.conv1,
        )

        self.down2 = nn.Sequential(
            self.rn18_model.bn1,
            self.rn18_model.relu,
            self.rn18_model.maxpool,
            self.rn18_model.layer1
        )
        self.down3 = self.rn18_model.layer2
        self.down4 = self.rn18_model.layer3
        self.down5 = self.rn18_model.layer4

        self.down_path = [
            self.down1,
            self.down2,
            self.down3,
            self.down4,
            self.down5
        ]
        
        # # Expansive Path
        self.up1 = UNetUpBlockRN(up_in=512, up_out=256, block_in=512, block_out=256, up_mode=up_mode)
        self.up2 = UNetUpBlockRN(up_in=256, up_out=128, block_in=256, block_out=128, up_mode=up_mode)
        self.up3 = UNetUpBlockRN(up_in=128, up_out=64, block_in=128, block_out=64, up_mode=up_mode)
        self.up4 = UNetUpBlockRN(up_in=64, up_out=64, block_in=128, block_out=64, up_mode=up_mode)
        self.up5 = UNetUpBlockRN(up_in=64, up_out=3, block_in=6, block_out=3, up_mode=up_mode)

        # Get number of channels to desired
        self.end_conv = nn.Conv2d(input_ch, target_ch, kernel_size=1)

    def forward(self, x):
        # Contracting Path
        skip = [x]

        for down in self.down_path:
            x = down(x)
            skip.append(x)

        # Expansive Path
        x = self.up1(x, skip[4])
        x = self.up2(x, skip[3])
        x = self.up3(x, skip[2])
        x = self.up4(x, skip[1])
        x = self.up5(x, skip[0])

        x = self.end_conv(x)
    
        return x
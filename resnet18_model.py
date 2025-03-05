import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torchvision.models import resnet18

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
        self.block = UNetConvBlock(crop_size=64, in_channels=block_in,
                                   out_channels=block_out)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        x, _ = self.block(x)

        return x

class ResNet18UNet(nn.Module):
    def __init__(self, input_ch: int = 3, target_ch: int = 3, up_mode: str ="deconv"):
        super().__init__()

        rn18_model = resnet18(weights=None)

        self.down1 = nn.Sequential(
            rn18_model.conv1,
        )

        self.down2 = nn.Sequential(
            rn18_model.bn1,
            rn18_model.relu,
            rn18_model.maxpool,
            rn18_model.layer1
        )
        self.down3 = rn18_model.layer2
        self.down4 = rn18_model.layer3
        self.down5 = rn18_model.layer4

        self.down_path = [
            self.down1,
            self.down2,
            self.down3,
            self.down4,
            self.down5
        ]
        
        # # Expansive Path
        self.up1 = UNetUpBlock(up_in=512, up_out=256, block_in=512, block_out=256, up_mode=up_mode)
        self.up2 = UNetUpBlock(up_in=256, up_out=128, block_in=256, block_out=128, up_mode=up_mode)
        self.up3 = UNetUpBlock(up_in=128, up_out=64, block_in=128, block_out=64, up_mode=up_mode)
        self.up4 = UNetUpBlock(up_in=64, up_out=64, block_in=128, block_out=64, up_mode=up_mode)
        self.up5 = UNetUpBlock(up_in=64, up_out=3, block_in=6, block_out=3, up_mode=up_mode)

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
    
dummy_data = torch.randn(2, 3, 224, 224)
model = ResNet18UNet(target_ch=1)
logits = model(dummy_data)
print(logits.shape)


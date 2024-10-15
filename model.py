import torch
import torch.nn as nn
import torchvision.transforms as transforms

class CenterCrop(nn.Module):
    def __init__(self, size):
        super().__init__()

        self.crop = transforms.CenterCrop(size)
    
    def forward(self, x):
        cropped_x = self.crop(x)
        return cropped_x

class UNetOriginal(nn.Module):
    def __init__(self):
        super().__init__()

        # Reused layers
        self.relu = nn.ReLU()
        self.down_sample = nn.MaxPool2d(kernel_size=(2,2))
        self.up_sample = nn.Upsample(scale_factor=2)

        # Contracting Path
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.crop1 = CenterCrop(392)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.crop2 = CenterCrop(200)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        self.crop3 = CenterCrop(104)

        self.conv7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)
        self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)
        self.crop4 = CenterCrop(56)

        self.conv9 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3)
        self.conv10 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3)

        # Expansive Path
        self.conv11 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)
        self.conv12 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)

        self.conv14 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.conv15 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3)
        self.conv16 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)

        self.conv17 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        self.conv18 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3)
        self.conv19 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)

        self.conv20 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.conv21 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3)
        self.conv22 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)

        self.conv23 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)

    def forward(self, x):
        # Contracting Path
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        across1 = self.crop1(x)
        x = self.down_sample(x)
        
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        across2 = self.crop2(x)
        x = self.down_sample(x)
        
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        across3 = self.crop3(x)
        x = self.down_sample(x)
        
        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        across4 = self.crop4(x)
        z = self.down_sample(x)

        x = self.relu(self.conv9(z))
        x = self.relu(self.conv10(x))
        
        # Expansive Path
        x = self.up_sample(x)
        x = self.conv11(x)
        x = torch.cat([across4, x], dim=1)
        x = self.relu(self.conv12(x))
        x = self.relu(self.conv13(x))
    
        x = self.up_sample(x)
        x = self.conv14(x)
        x = torch.cat([across3, x], dim=1)
        x = self.relu(self.conv15(x))
        x = self.relu(self.conv16(x))
    
        x = self.up_sample(x)    
        x = self.conv17(x)
        x = torch.cat([across2, x], dim=1)
        x = self.relu(self.conv18(x))
        x = self.relu(self.conv19(x))

        x = self.up_sample(x)
        x = self.conv20(x)
        x = torch.cat([across1, x], dim=1)
        x = self.relu(self.conv21(x))
        x = self.relu(self.conv22(x))
        
        x = self.conv23(x)
    
        return x, z

# model = UNet()
# dummy_data = torch.randn((2, 1, 572, 572))
# x, z = model(dummy_data)
# print(x.shape, z.shape)

# model = UNet()
# works = []
# for i in range(100, 601):
#     print(i)
#     try:
#         dummy_data = torch.randn((2, 1, i, i))
#         x, z = model(dummy_data)
#         print(x.shape, z.shape)
#         works.append(i)
#     except:
#         pass

# print(works)

class UNet(nn.Module):
    def __init__(self, img_size, num_channel=1, first_out=64):
        super().__init__()

        second_out = first_out * 2
        third_out = second_out * 2
        fourth_out = third_out * 2
        fifth_out = fourth_out * 2

        # Reused layers
        self.relu = nn.ReLU()
        self.down_sample = nn.MaxPool2d(kernel_size=(2,2))
        self.up_sample = nn.Upsample(scale_factor=2)

        # Contracting Path
        self.conv1 = nn.Conv2d(in_channels=num_channel, out_channels=first_out, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=first_out, out_channels=first_out, kernel_size=3, padding=1)
        self.crop1 = CenterCrop(img_size)

        self.conv3 = nn.Conv2d(in_channels=first_out, out_channels=second_out, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=second_out, out_channels=second_out, kernel_size=3, padding=1)
        self.crop2 = CenterCrop(img_size // 2)

        self.conv5 = nn.Conv2d(in_channels=second_out, out_channels=third_out, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=third_out, out_channels=third_out, kernel_size=3, padding=1)
        self.crop3 = CenterCrop(img_size // 4)

        self.conv7 = nn.Conv2d(in_channels=third_out, out_channels=fourth_out, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(in_channels=fourth_out, out_channels=fourth_out, kernel_size=3, padding=1)
        self.crop4 = CenterCrop(img_size // 8)

        self.conv9 = nn.Conv2d(in_channels=fourth_out, out_channels=fifth_out, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(in_channels=fifth_out, out_channels=fifth_out, kernel_size=3, padding=1)

        # Expansive Path
        self.conv11 = nn.Conv2d(in_channels=fifth_out, out_channels=fourth_out, kernel_size=1)
        self.conv12 = nn.Conv2d(in_channels=fifth_out, out_channels=fourth_out, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(in_channels=fourth_out, out_channels=fourth_out, kernel_size=3, padding=1)

        self.conv14 = nn.Conv2d(in_channels=fourth_out, out_channels=third_out, kernel_size=1)
        self.conv15 = nn.Conv2d(in_channels=fourth_out, out_channels=third_out, kernel_size=3, padding=1)
        self.conv16 = nn.Conv2d(in_channels=third_out, out_channels=third_out, kernel_size=3, padding=1)

        self.conv17 = nn.Conv2d(in_channels=third_out, out_channels=second_out, kernel_size=1)
        self.conv18 = nn.Conv2d(in_channels=third_out, out_channels=second_out, kernel_size=3, padding=1)
        self.conv19 = nn.Conv2d(in_channels=second_out, out_channels=second_out, kernel_size=3, padding=1)

        self.conv20 = nn.Conv2d(in_channels=second_out, out_channels=first_out, kernel_size=1)
        self.conv21 = nn.Conv2d(in_channels=second_out, out_channels=first_out, kernel_size=3, padding=1)
        self.conv22 = nn.Conv2d(in_channels=first_out, out_channels=first_out, kernel_size=3, padding=1)

        self.conv23 = nn.Conv2d(in_channels=first_out, out_channels=num_channel, kernel_size=1)

    def forward(self, x):
        # Contracting Path
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        across1 = self.crop1(x)
        x = self.down_sample(x)
        
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        across2 = self.crop2(x)
        x = self.down_sample(x)
        
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        across3 = self.crop3(x)
        x = self.down_sample(x)
        
        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        across4 = self.crop4(x)
        z = self.down_sample(x)
        z_flat = z.view(z.shape[0], -1)

        x = self.relu(self.conv9(z))
        x = self.relu(self.conv10(x))
        
        # Expansive Path
        x = self.up_sample(x)
        x = self.conv11(x)
        x = torch.cat([across4, x], dim=1)
        x = self.relu(self.conv12(x))
        x = self.relu(self.conv13(x))
    
        x = self.up_sample(x)
        x = self.conv14(x)
        x = torch.cat([across3, x], dim=1)
        x = self.relu(self.conv15(x))
        x = self.relu(self.conv16(x))

        x = self.up_sample(x)    
        x = self.conv17(x)
        x = torch.cat([across2, x], dim=1)
        x = self.relu(self.conv18(x))
        x = self.relu(self.conv19(x))

        x = self.up_sample(x)
        x = self.conv20(x)
        x = torch.cat([across1, x], dim=1)
        x = self.relu(self.conv21(x))
        x = self.relu(self.conv22(x))
        
        x = self.conv23(x)
    
        return x, z, z_flat
    
# img_size = 224
# model = UNet(img_size=img_size, num_channel=1, first_out=2)
# dummy = torch.randn((2, 1, img_size, img_size))
# x, z, z_flat = model(dummy)
# print(x.shape, z.shape, z_flat.shape)

class SmallUNet(nn.Module):
    def __init__(self, img_size, first_out=4):
        super().__init__()

        second_out = first_out * 2
        third_out = second_out * 2
        fourth_out = third_out * 2
        fifth_out = fourth_out * 2

        # Reused layers
        self.relu = nn.ReLU()
        self.down_sample = nn.MaxPool2d(kernel_size=(2,2))
        self.up_sample = nn.Upsample(scale_factor=2)

        # Contracting Path
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=first_out, kernel_size=3, padding=1)
        self.crop1 = CenterCrop(img_size)

        self.conv2 = nn.Conv2d(in_channels=first_out, out_channels=second_out, kernel_size=3, padding=1)
        self.crop2 = CenterCrop(img_size // 2)

        self.conv3 = nn.Conv2d(in_channels=second_out, out_channels=third_out, kernel_size=3, padding=1)
        self.crop3 = CenterCrop(img_size // 4)

        self.conv4 = nn.Conv2d(in_channels=third_out, out_channels=fourth_out, kernel_size=3, padding=1)
        self.crop4 = CenterCrop(img_size // 8)

        self.conv5 = nn.Conv2d(in_channels=fourth_out, out_channels=fifth_out, kernel_size=3, padding=1)

        # Expansive Path
        self.conv6 = nn.Conv2d(in_channels=fifth_out, out_channels=fourth_out, kernel_size=1)
        self.conv7 = nn.Conv2d(in_channels=fifth_out, out_channels=fourth_out, kernel_size=3, padding=1)

        self.conv8 = nn.Conv2d(in_channels=fourth_out, out_channels=third_out, kernel_size=1)
        self.conv9 = nn.Conv2d(in_channels=fourth_out, out_channels=third_out, kernel_size=3, padding=1)

        self.conv10 = nn.Conv2d(in_channels=third_out, out_channels=second_out, kernel_size=1)
        self.conv11 = nn.Conv2d(in_channels=third_out, out_channels=second_out, kernel_size=3, padding=1)
        
        self.conv12 = nn.Conv2d(in_channels=second_out, out_channels=first_out, kernel_size=1)
        self.conv13 = nn.Conv2d(in_channels=second_out, out_channels=first_out, kernel_size=3, padding=1)

        self.conv14 = nn.Conv2d(in_channels=first_out, out_channels=1, kernel_size=1)

    def forward(self, x):
        # Contracting Path
        x = self.relu(self.conv1(x))
        across1 = self.crop1(x)
        x = self.down_sample(x)
        
        x = self.relu(self.conv2(x))
        across2 = self.crop2(x)
        x = self.down_sample(x)

        x = self.relu(self.conv3(x))
        across3 = self.crop3(x)
        x = self.down_sample(x)

        x = self.relu(self.conv4(x))
        across4 = self.crop4(x)
        z = self.down_sample(x)
        z_flat = z.view(z.shape[0], -1)

        x = self.relu(self.conv5(z))

        # Expansive Path
        x = self.up_sample(x)
        x = self.conv6(x)
        x = torch.cat([across4, x], dim=1)
        x = self.relu(self.conv7(x))

        x = self.up_sample(x)
        x = self.conv8(x)
        x = torch.cat([across3, x], dim=1)
        x = self.relu(self.conv9(x))
    
        x = self.up_sample(x)    
        x = self.conv10(x)
        x = torch.cat([across2, x], dim=1)
        x = self.relu(self.conv11(x))

        x = self.up_sample(x)
        x = self.conv12(x)
        x = torch.cat([across1, x], dim=1)
        x = self.relu(self.conv13(x))

        x = self.conv14(x)

        return x, z, z_flat

# img_size = 128
# model = SmallUNet(img_size=img_size)
# dummy_data = torch.randn((2, 1, img_size, img_size))
# x, z, z_flat = model(dummy_data)
# print(x.shape, z.shape, z_flat.shape)
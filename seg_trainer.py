import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as v2

from PIL import Image
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import InterpolationMode

from losses import *
from models import UNet, ResNet18UNet


class SegDataset(Dataset):
    def __init__(self, csv_path: str, img_dir: str, mask_dir: str, 
                 img_transform=None, mask_transform=None):
        self.csv = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = os.path.join(self.img_dir, self.csv.iloc[index, 0])
        mask_path = os.path.join(self.mask_dir, self.csv.iloc[index, 0])

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.img_transform:
            img = self.img_transform(img)
            
        if self.mask_transform:
            mask = self.mask_transform(mask)

        mask = (mask > 0.5).float()

        # print(torch.unique(mask))

        return (img, mask)
    

class CarvanaDataset(Dataset):
    def __init__(self, csv_path: str, img_dir: str, mask_dir: str, 
                 img_transform=None, mask_transform=None):
        self.csv = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        file_name = self.csv.iloc[index, 0]
        file_name = file_name[:-4]
        img_path = os.path.join(self.img_dir, f'{file_name}.jpg')
        mask_path = os.path.join(self.mask_dir, f'{file_name}_mask.gif')

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.img_transform:
            img = self.img_transform(img)
            
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return (img, mask)


class RN18UnetConfig():
    def __init__(self):
        # Training hyperparameters
        self.epochs = 200
        self.batch_size = 64
        self.num_workers = 8
        self.lr = 1e-4

        # Data parameters for Carvana dataset
        # self.train_csv = 'data/Carvana/train.csv'
        # self.valid_csv = 'data/Carvana/valid.csv'

        # self.img_dir = 'data/Carvana/images'
        # self.mask_dir = 'data/Carvana/masks'

        self.init_classes()

    def init_classes(self):
        # Model
        self.model = ResNet18UNet(target_ch=1, up_mode="upsample")
        
        # Loss function/Optimizer
        self.criterion = DiceLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Datasets and Dataloaders
        img_train_transforms = v2.Compose([
            v2.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
            # v2.RandomHorizontalFlip(),
            # v2.RandomRotation(degrees=20),
            # v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

        mask_train_transforms = v2.Compose([
            v2.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

        img_valid_transforms = v2.Compose([
            v2.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

        mask_valid_transforms = v2.Compose([
            v2.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

        self.train_set = CarvanaDataset(
            csv_path=self.train_csv,
            img_dir=self.img_dir,
            mask_dir=self.mask_dir,
            img_transform=img_train_transforms,
            mask_transform=mask_train_transforms
        )

        self.valid_set = CarvanaDataset(
            csv_path=self.valid_csv,
            img_dir=self.img_dir,
            mask_dir=self.mask_dir,
            img_transform=img_valid_transforms,
            mask_transform=mask_valid_transforms
        )

        self.train_dataloader = DataLoader(self.train_set, batch_size=self.batch_size, 
                                           shuffle=False, num_workers=self.num_workers)
        self.valid_dataloader = DataLoader(self.valid_set, batch_size=self.batch_size, 
                                           shuffle=True, num_workers=8)
        

class UnetConfig():
    def __init__(self):
        # Training hyperparameters
        self.epochs = 400
        self.batch_size = 64
        self.num_workers = 8
        self.lr = 1e-4

        # Data parameters for Carvana dataset
        # self.train_csv = 'data/Carvana/train.csv'
        # self.valid_csv = 'data/Carvana/valid.csv'

        # self.img_dir = 'data/Carvana/images'
        # self.mask_dir = 'data/Carvana/masks'

        self.init_classes()

    def init_classes(self):
        # Model
        self.model = UNet(img_size=224, first_out=64)
        
        # Loss function/Optimizer
        self.criterion = DiceLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Datasets and Dataloaders
        img_train_transforms = v2.Compose([
            v2.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
            # v2.RandomHorizontalFlip(),
            # v2.RandomRotation(degrees=20),
            # v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            # v2.GaussianNoise()
        ])

        mask_train_transforms = v2.Compose([
            v2.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

        img_valid_transforms = v2.Compose([
            v2.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

        mask_valid_transforms = v2.Compose([
            v2.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

        self.train_set = CarvanaDataset(
            csv_path=self.train_csv,
            img_dir=self.img_dir,
            mask_dir=self.mask_dir,
            img_transform=img_train_transforms,
            mask_transform=mask_train_transforms
        )

        self.valid_set = CarvanaDataset(
            csv_path=self.valid_csv,
            img_dir=self.img_dir,
            mask_dir=self.mask_dir,
            img_transform=img_valid_transforms,
            mask_transform=mask_valid_transforms
        )

        self.train_dataloader = DataLoader(self.train_set, batch_size=self.batch_size, 
                                           shuffle=False, num_workers=self.num_workers)
        self.valid_dataloader = DataLoader(self.valid_set, batch_size=self.batch_size, 
                                           shuffle=True, num_workers=8)


class Trainer():
    def __init__(
            self,
            model,
            criterion,
            optimizer,
            train_dataloader,
            valid_dataloader,
            epochs,
            save_dir
        ) -> None:
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = model
        self.model = model.to(self.device)

        self.criterion = criterion
        self.optimizer = optimizer

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.epochs = epochs
        self.save_dir = save_dir

        self.writer = SummaryWriter(self.save_dir)

        self.training_stats = os.path.join(self.save_dir, 'train_stats.csv')
        with open(self.training_stats, 'a') as stats_csv:
            header = f'epoch,train_loss,train_acc,valid_loss,valid_acc'
            stats_csv.write(header + '\n')

        self.ckpt_dir = os.path.join(self.save_dir, 'models')
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0

            num_correct, num_pixels = 0, 0
            
            for bi, (images, masks) in enumerate(self.train_dataloader):
                images = images.to(self.device)
                masks = masks.to(self.device)

                self.optimizer.zero_grad()
                
                logits = self.model(images)

                # Checking accuracy of model
                preds = torch.sigmoid(logits)
                preds = (preds > 0.5).float()
                num_correct += (preds == masks).sum()
                num_pixels += torch.numel(preds)

                loss = self.criterion(logits, masks)
                loss.backward()
                
                self.optimizer.step()
                
                epoch_loss += loss.item()

            train_loss = epoch_loss / len(self.train_dataloader)
            train_acc = num_correct / num_pixels
            valid_loss, valid_acc = self.valid()

            self.writer.add_scalar('Loss/train', train_loss, epoch+1)
            self.writer.add_scalar('Loss/valid', valid_loss, epoch+1)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch+1)
            self.writer.add_scalar('Accuracy/valid', valid_acc, epoch+1)

            with open(self.training_stats, 'a') as stats_csv:
                line = f'{epoch+1},{train_loss},{train_acc},{valid_loss},{valid_acc}'
                stats_csv.write(line + "\n")

            # torch.save(self.model.rn18_model.state_dict(), os.path.join(self.ckpt_dir, f'encoder_{epoch}.pt'))
            torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, f'unet.pt'))

            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")

    def valid(self):
        self.model.eval()

        epoch_loss = 0
        num_correct, num_pixels = 0, 0
        
        with torch.no_grad():
            for bi, (images, masks) in enumerate(self.valid_dataloader):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                logits = self.model(images)

                # Checking accuracy of model
                preds = torch.sigmoid(logits)
                preds = (preds > 0.5).float()

                preds_np = preds.cpu().numpy()
                count_ones = np.count_nonzero(preds_np == 1.) 
                unique_values = np.unique(preds_np)
                # print(count_ones, unique_values)

                num_correct += (preds == masks).sum()
                num_pixels += torch.numel(preds)
                
                loss = self.criterion(logits, masks)
                            
                epoch_loss += loss.item()

        valid_loss = epoch_loss / len(self.valid_dataloader)
        valid_acc = num_correct / num_pixels

        return valid_loss, valid_acc
            

def main() -> None:
    save_dir = os.path.join("./training_runs/", datetime.now().strftime("%Y%b%d_%H:%M:%S"))
    os.makedirs(save_dir, exist_ok=True)

    config = UnetConfig()

    trainer = Trainer(
        model=config.model,
        criterion=config.criterion,
        optimizer=config.optimizer,
        train_dataloader=config.train_dataloader,
        valid_dataloader=config.valid_dataloader,
        epochs=config.epochs,
        save_dir=save_dir
    )
    trainer.train()


if __name__ == "__main__":
    main()

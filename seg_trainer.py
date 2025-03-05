import os
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as v2

from PIL import Image
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from resnet18_model import ResNet18UNet


class SegDataset(Dataset):
    def __init__(self, csv_path: str, img_dir: str, mask_dir: str, transform=None):
        self.csv = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = os.path.join(self.img_dir, self.csv.iloc[index, 0])
        mask_path = os.path.join(self.mask_dir, self.csv.iloc[index, 0])

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return (img, mask)
    

train_csv = 'data/train.csv'
valid_csv = 'data/valid.csv'

img_dir = 'data/images'
mask_dir = 'data/masks'

train_transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.RandomHorizontalFlip(),
    v2.RandomRotation(degrees=20),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

valid_transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

train_set = SegDataset(csv_path=train_csv, img_dir=img_dir, mask_dir=mask_dir,
                       transform=train_transforms)
valid_set = SegDataset(csv_path=valid_csv, img_dir=img_dir, mask_dir=mask_dir,
                       transform=valid_transforms)

train_dataloader = DataLoader(train_set, batch_size=64, shuffle=False, num_workers=8)
valid_dataloader = DataLoader(valid_set, batch_size=64, shuffle=True, num_workers=8)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model
model = ResNet18UNet(target_ch=1)
model = model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

save_dir = os.path.join("./seg_results/", datetime.now().strftime("%Y%b%d_%H:%M:%S"))
os.makedirs(save_dir, exist_ok=True)

training_stats = os.path.join(save_dir, 'train_stats.csv')
with open(training_stats, 'a') as stats_csv:
    header = f'epoch,train_loss,train_acc,valid_loss,valid_acc'
    stats_csv.write(header + '\n')

ckpt_dir = os.path.join(save_dir, 'encoders')
os.makedirs(ckpt_dir, exist_ok=True)

writer = SummaryWriter(save_dir)

def train(num_epochs: int = 30):

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        num_correct, num_pixels = 0, 0
        
        for bi, (images, masks) in enumerate(train_dataloader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            
            logits = model(images)

            # Checking accuracy of model
            preds = torch.sigmoid(logits)
            preds = (preds > 0.5).float()
            num_correct += (preds == masks).sum()
            num_pixels += torch.numel(preds)

            loss = criterion(logits, masks)
            loss.backward()
            
            optimizer.step()
            
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_dataloader)
        train_acc = num_correct / num_pixels
        valid_loss, valid_acc = valid()

        writer.add_scalar('Loss/train', train_loss, epoch+1)
        writer.add_scalar('Loss/valid', valid_loss, epoch+1)
        writer.add_scalar('Accuracy/train', train_acc, epoch+1)
        writer.add_scalar('Accuracy/valid', valid_acc, epoch+1)

        with open(training_stats, 'a') as stats_csv:
            line = f'{epoch+1},{train_loss},{train_acc},{valid_loss},{valid_acc}'
            stats_csv.write(line + "\n")

        torch.save(model.rn18_model.state_dict(), os.path.join(ckpt_dir, f'encoder_{epoch}.pt'))

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")

def valid():
    model.eval()

    epoch_loss = 0
    num_correct, num_pixels = 0, 0
    
    with torch.no_grad():
        for bi, (images, masks) in enumerate(valid_dataloader):
            images = images.to(device)
            masks = masks.to(device)
            
            logits = model(images)

            # Checking accuracy of model
            preds = torch.sigmoid(logits)
            preds = (preds > 0.5).float()
            num_correct += (preds == masks).sum()
            num_pixels += torch.numel(preds)
            
            loss = criterion(logits, masks)
                        
            epoch_loss += loss.item()

    valid_loss = epoch_loss / len(valid_dataloader)
    valid_acc = num_correct / num_pixels

    return valid_loss, valid_acc
        
# Train the model
train()


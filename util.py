import os
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms.v2 as v2

from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode

from seg_trainer import RN18UnetConfig, UnetConfig, SegDataset


def save_generated_masks(
        config: RN18UnetConfig | UnetConfig,
        unet_path: str, 
        mask_img_dir: str, 
        num_imgs: int = 10
    ) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = config.model
    model.load_state_dict(torch.load(unet_path, weights_only=True))
    model = model.to(device)
    model.eval()

    valid_dataloader = DataLoader(config.valid_set, batch_size=4, 
                                  shuffle=False, num_workers=1)
    with torch.no_grad():
        for bi, (images, masks) in enumerate(valid_dataloader):
            if bi == num_imgs: break

            images = images.to(device)
            masks = masks.to(device)
            
            logits = model(images)
            predicted_masks = torch.sigmoid(logits)
            predicted_masks = (predicted_masks > 0.5).float()

            images = images.cpu().numpy()
            masks = masks.cpu().numpy()
            predicted_masks = predicted_masks.cpu().numpy()
            save_path = os.path.join(mask_img_dir, f'mask_img_{bi}.png')

            gen_mask_comparison(
                imgs=images,
                masks=masks,
                pred_masks=predicted_masks,
                save_path=save_path
            )


def gen_mask_comparison(imgs: torch.Tensor, masks: torch.Tensor, pred_masks: torch.Tensor, save_path: str) -> None:
    num_cols = masks.shape[0]

    fig, axes = plt.subplots(2, num_cols, figsize=(8, 4), dpi=500)

    for i, (img, mask, pred_mask) in enumerate(zip(imgs, masks, pred_masks)):
        img = np.moveaxis(img, 0, -1)
        mask = mask.squeeze()
        pred_mask = pred_mask.squeeze()

        mask_overlay = np.zeros((*mask.shape, 4))
        mask_overlay[mask == 1] = [1, 0, 0, 0.5]

        pred_mask_overlay = np.zeros((*pred_mask.shape, 4))
        pred_mask_overlay[pred_mask == 1] = [1, 0, 0, 0.5]

        # Top row is generated mask
        axes[0][i].set_title("Generated Mask")
        axes[0][i].imshow(img)
        axes[0][i].imshow(pred_mask_overlay)
        axes[0][i].axis('off')

        # Bot row is ground truth mask
        axes[1][i].set_title("GT Mask")
        axes[1][i].imshow(img)
        axes[1][i].imshow(mask_overlay)
        axes[1][i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)


def display_gen_masks(mask_dir: str, mask_csv: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    mask_csv = pd.read_csv(mask_csv)
    mask_filenames = list(mask_csv['filename'].to_numpy())

    for i in range(0, len(mask_filenames), 25):
        fig, axes = plt.subplots(5, 5, figsize=(20,20))
        axes = axes.flatten()

        for j in range(25):
            if i + j >= len(mask_filenames):
                break

            ax = axes[j]
            index = i + j
            mask_filename = mask_filenames[index]
            mask_img = Image.open(os.path.join(mask_dir, mask_filename))
            ax.imshow(mask_img)
            ax.set_title(mask_filename)
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"display_masks_{i}.png"))
        plt.close()

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.utils.data import DataLoader

from seg_trainer import RN18UnetConfig

def save_generated_masks(config: RN18UnetConfig, mask_img_dir: str, num_imgs: int = 10) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = config.model
    model.load_state_dict(torch.load('seg_results/2025Mar10_09:28:19/models/unet_29.pt', weights_only=True))
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

            masks = masks.cpu().numpy()
            predicted_masks = predicted_masks.cpu().numpy()
            save_path = os.path.join(mask_img_dir, f'mask_img_{bi}.png')

            gen_mask_comparison(
                masks=masks,
                pred_masks=predicted_masks,
                save_path=save_path
            )


def gen_mask_comparison(masks: torch.Tensor, pred_masks: torch.Tensor, save_path: str) -> None:
    num_cols = masks.shape[0]

    fig, axes = plt.subplots(2, 4, figsize=(8, 4), dpi=500)

    for i, (mask, pred_mask) in enumerate(zip(masks, pred_masks)):
        mask = mask.squeeze() * 255.
        pred_mask = pred_mask.squeeze() * 255.

        # Top row is generated mask
        axes[0][i].set_title("Generated Mask")
        axes[0][i].imshow(pred_mask, cmap='gray', vmin=0, vmax=255)
        axes[0][i].axis('off')

        # Bot row is ground truth mask
        axes[1][i].set_title("GT Mask")
        axes[1][i].imshow(mask, cmap='gray', vmin=0, vmax=255)
        axes[1][i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)


config = RN18UnetConfig()
mask_img_dir = 'seg_results/2025Mar10_09:28:19/mask_imgs'
os.makedirs(mask_img_dir, exist_ok=True)
save_generated_masks(config, mask_img_dir)


def dice_coefficient(pred, target, smooth=1e-6):
    """
    Compute the Dice Coefficient
    :param pred: Predicted tensor (batch_size, num_classes, H, W)
    :param target: Ground truth tensor (batch_size, num_classes, H, W)
    :param smooth: Smoothing factor to avoid division by zero
    :return: Dice coefficient
    """
    pred = pred.contiguous()
    target = target.contiguous()
    
    intersection = (pred * target).sum(dim=(2, 3))
    denominator = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    
    dice = (2. * intersection + smooth) / (denominator + smooth)
    return dice.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        """
        Compute Dice Loss
        :param logits: Raw logits from the model (before softmax/sigmoid)
        :param targets: Ground truth masks (one-hot encoded for multi-class)
        :return: Dice loss value
        """
        num_classes = logits.shape[1]
        
        # Convert logits to probabilities
        if num_classes == 1:
            probs = torch.sigmoid(logits)
            targets = targets.float()
        else:
            probs = F.softmax(logits, dim=1)
            targets = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        dice_loss = 1 - dice_coefficient(probs, targets, self.smooth)
        return dice_loss
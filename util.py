import os
import torch
import matplotlib.pyplot as plt

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
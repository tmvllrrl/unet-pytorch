import torch
import torch.nn as nn
import torch.nn.functional as F


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
    

class WeightedDiceLoss(nn.Module):
    def __init__(self, class_weights=None, smooth=1e-6):
        """
        Weighted Dice Loss for handling class imbalance.
        :param class_weights: Tensor of shape (num_classes,) assigning more weight to minority classes.
        :param smooth: Smoothing term to avoid division by zero.
        """
        super(WeightedDiceLoss, self).__init__()
        self.class_weights = class_weights
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]

        # Convert logits to probabilities
        if num_classes == 1:
            probs = torch.sigmoid(logits)
            targets = targets.float()
        else:
            probs = F.softmax(logits, dim=1)
            targets = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        # Compute per-class Dice loss
        intersection = (probs * targets).sum(dim=(2, 3))
        denominator = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

        dice_score = (2. * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1 - dice_score

        # Apply class weights if provided
        if self.class_weights is not None:
            dice_loss *= self.class_weights.view(1, -1, 1, 1)

        return dice_loss.mean()
    

class BinaryWeightedDiceLoss(nn.Module):
    def __init__(self, alpha=0.5, smooth=1e-6):
        """
        Weighted Dice Loss for binary segmentation.
        :param alpha: Weight for the foreground class (class 1).
                      Higher value gives more importance to class 1.
        :param smooth: Smoothing factor to avoid division by zero.
        """
        super(BinaryWeightedDiceLoss, self).__init__()
        self.alpha = alpha
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        :param logits: Raw model outputs (before sigmoid), shape (batch, 1, H, W)
        :param targets: Ground truth binary masks, shape (batch, 1, H, W)
        :return: Weighted Dice loss value
        """
        probs = torch.sigmoid(logits)  # Convert logits to probabilities
        targets = targets.float()  # Ensure targets are float

        # Compute intersection and denominator
        intersection = (probs * targets).sum(dim=(2, 3))
        denominator = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

        # Weighted Dice score
        dice_score = (2. * intersection + self.smooth) / (denominator + self.smooth)

        # Apply weight: Background (1-alpha), Foreground (alpha)
        weights = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # Weighted Dice loss
        dice_loss = (1 - dice_score) * weights.mean()

        return dice_loss.mean()
    

class LogCoshDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(LogCoshDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1) if num_classes > 1 else torch.sigmoid(logits)
        targets = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float() if num_classes > 1 else targets.float()

        intersection = (probs * targets).sum(dim=(2, 3))
        denominator = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

        dice_loss = 1 - (2. * intersection + self.smooth) / (denominator + self.smooth)
        return torch.log(torch.cosh(dice_loss)).mean()
    

class DiceFocalLoss(nn.Module):
    def __init__(self, dice_alpha=0.9, focal_alpha=0.15, gamma=2.0, 
                 lambda_dice=0.5, lambda_focal=0.5):
        super(DiceFocalLoss, self).__init__()
        self.dice_loss = BinaryWeightedDiceLoss(alpha=dice_alpha)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=gamma)
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal

    def forward(self, logits, targets):
        dice_loss = self.dice_loss(logits, targets)
        focal_loss = self.focal_loss(logits, targets)
        return self.lambda_dice * dice_loss + self.lambda_focal * focal_loss


class FocalLoss(nn.Module):
    """
    Implementation of Focal Loss for handling class imbalance.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        :param alpha: Balancing factor for positive class (default: 0.25 for class imbalance handling)
        :param gamma: Focusing parameter to down-weight easy examples (default: 2.0)
        :param reduction: 'mean' (default), 'sum', or 'none' for loss reduction
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Compute Focal Loss
        :param logits: Raw logits from the model (batch_size, num_classes, ...)
        :param targets: Ground truth labels (batch_size, ...), class indices (not one-hot)
        :return: Focal loss value
        """
        num_classes = logits.shape[1]
        
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1) if num_classes > 1 else torch.sigmoid(logits)
        
        # Convert targets to one-hot if multi-class
        if num_classes > 1:
            targets = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        else:
            targets = targets.float()
        
        ce_loss = F.cross_entropy(logits, targets.argmax(dim=1), reduction='none') if num_classes > 1 \
                  else F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        p_t = (targets * probs) + ((1 - targets) * (1 - probs))
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * ce_loss
        
        # Apply alpha weighting for class imbalance (only for binary/multi-class cases)
        if num_classes == 1:
            alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            loss *= alpha_weight
        
        # Reduce loss based on the reduction parameter
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# Usage:
# criterion = DiceFocalLoss(lambda_dice=0.7, lambda_focal=0.3)
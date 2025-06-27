import torch.nn as nn
from monai import losses
from monai.losses import SSIMLoss


class SSIM_L1_Loss(nn.Module):
    def __init__(self, alpha=0.84):
        """
        alpha: 控制 L1 和 SSIM 的比例，默认 0.84 * L1 + 0.16 * SSIM
        """
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss(reduction="mean")
        self.ssim = SSIMLoss(spatial_dims=2, data_range=1.0)

    def forward(self, predictions, targets):
        l1_loss = self.l1(predictions, targets)
        ssim_loss = self.ssim(predictions, targets)
        return self.alpha * l1_loss + (1 - self.alpha) * ssim_loss

class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = nn.L1Loss(reduction="mean")

    def forward(self, predictions, targets):
        return self._loss(predictions, targets)

class L2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = nn.MSELoss(reduction="mean")

    def forward(self, predictions, targets):
        return self._loss(predictions, targets)


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, predictions, targets):
        loss = self._loss(predictions, targets)
        return loss


class BinaryCrossEntropyWithLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, predictions, tragets):
        loss = self._loss(predictions, tragets)
        return loss


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = losses.DiceLoss(to_onehot_y=False, sigmoid=True)

    def forward(self, predicted, target):
        loss = self._loss(predicted, target)
        return loss


class DiceCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = losses.DiceCELoss(to_onehot_y=False, sigmoid=True)

    def forward(self, predicted, target):
        loss = self._loss(predicted, target)
        return loss


class DiceFocalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = losses.DiceFocalLoss(to_onehot_y=False, sigmoid=True)

    def forward(self, predicted, target):
        loss = self._loss(predicted, target)
        return loss


def build_loss(config):
    if config.loss.name == "CE":
        return CrossEntropyLoss()
    elif config.loss.name == "BCE":
        return BinaryCrossEntropyWithLogits()
    elif config.loss.name == "Dice":
        return DiceLoss()
    elif config.loss.name == "Dice_CE":
        return DiceCELoss()
    elif config.loss.name == "Dice_Focal":
        return DiceFocalLoss()
    elif config.loss.name == "MAE":
        return L1Loss()
    elif config.loss.name == "MSE":
        return L2Loss()
    elif config.loss.name == "SSIM_L1":
        return SSIM_L1_Loss()
    else:
        raise ValueError("must be cross entropy or soft dice loss for now!")

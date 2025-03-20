import torch
import torch.nn as nn
import torch.nn.functional as F

# SSIM 计算
def ssim_loss(pred, target):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(pred, kernel_size=3, stride=1, padding=1)
    mu_y = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)

    sigma_x = F.avg_pool2d(pred ** 2, kernel_size=3, stride=1, padding=1) - mu_x ** 2
    sigma_y = F.avg_pool2d(target ** 2, kernel_size=3, stride=1, padding=1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(pred * target, kernel_size=3, stride=1, padding=1) - mu_x * mu_y

    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    ssim_map = ssim_n / ssim_d
    return torch.clamp((1 - ssim_map) / 2, 0, 1).mean()

# 自定义组合损失
class CustomLoss(nn.Module):
    def __init__(self, alpha=0.85, beta=0.15):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        ssim = ssim_loss(pred, target)
        return self.alpha * ssim + self.beta * l1


if __name__ == '__main__':
    criterion = CustomLoss(alpha=0.85, beta=0.15)

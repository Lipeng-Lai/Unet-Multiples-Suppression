import torch
import torch.nn.functional as F
import kornia

def compute_mse(pred, target):
    """向量化计算MSE"""
    return F.mse_loss(pred, target, reduction='mean')

def compute_psnr(pred, target, max_val=1.0, eps=1e-10):
    """向量化计算PSNR"""
    mse = torch.mean((pred - target)**2, dim=tuple(range(1, pred.dim())))
    psnr = 10 * torch.log10(max_val**2 / (mse + eps))
    psnr = torch.nan_to_num(psnr, posinf=0.0)  # 处理无穷大
    return psnr.mean()

def compute_snr(pred, target, eps=1e-10):
    """向量化计算SNR"""
    noise = pred - target
    signal_power = torch.mean(target**2, dim=tuple(range(1, target.dim())))
    noise_power = torch.mean(noise**2, dim=tuple(range(1, noise.dim())))
    snr = 10 * torch.log10((signal_power + eps) / (noise_power + eps))
    return snr.mean()

def compute_ssim(pred, target, max_val=1.0):
    """使用kornia优化SSIM计算（适配不同版本）"""
    # 自动检测kornia版本并选择对应参数
    try:
        # 适用于kornia 0.6.0及以上版本
        return kornia.metrics.ssim(
            pred, target, 
            window_size=11, 
            max_val=max_val
        ).mean()
    except TypeError:
        # 回退到旧版本参数
        return kornia.metrics.ssim(
            pred, target, 
            window_size=11, 
            max_val=max_val,
            reduction='mean'
        )

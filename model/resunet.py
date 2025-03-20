import torch
import torch.nn as nn
import torch.nn.functional as F

class ResConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果输入通道数和输出通道数不同，则使用1x1卷积调整
        self.residual_conv = None
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)  # 统一通道数
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return residual + x  # 现在 residual 和 x 的通道数一致


class Down2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ResConv2D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ResConv2D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 计算空间维度差异
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class ResUNet2D(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = ResConv2D(n_channels, 16)
        self.down1 = Down2D(16, 32)
        self.down2 = Down2D(32, 64)
        self.down3 = Down2D(64, 128)

        self.up2 = Up2D(192, 64)  # 128+64
        self.up3 = Up2D(96, 32)   # 64+32
        self.up4 = Up2D(48, 16)   # 32+16
        self.out = OutConv2D(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out(x)
        return logits

if __name__ == "__main__":
    x = torch.randn(1, 1, 256, 256)
    model = ResUNet2D(n_channels=1, n_classes=1)
    output = model(x)
    print("输入形状:", x.shape)
    print("输出形状:", output.shape)
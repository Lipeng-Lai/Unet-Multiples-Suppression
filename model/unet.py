import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """双层卷积模块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """下采样模块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.mpconv(x)

class Up(nn.Module):
    """上采样模块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels*2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 拼接特征图
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet2D(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        # 编码器
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        # 解码器
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        
        # 输出层
        self.outc = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # 编码路径
        x1 = self.inc(x)        # [1,64,256,256]
        x2 = self.down1(x1)     # [1,128,128,128]
        x3 = self.down2(x2)     # [1,256,64,64]
        x4 = self.down3(x3)     # [1,512,32,32]
        x5 = self.down4(x4)     # [1,1024,16,16]
        
        # 解码路径
        d = self.up1(x5, x4)    # [1,512,32,32]
        d = self.up2(d, x3)     # [1,256,64,64]
        d = self.up3(d, x2)     # [1,128,128,128]
        d = self.up4(d, x1)     # [1,64,256,256]
        
        # 输出
        return self.outc(d)     # [1,1,256,256]

if __name__ == "__main__":
    x = torch.randn(1, 1, 256, 256)
    model = UNet2D(in_channels=1)
    output = model(x)
    
    print("输入形状:", x.shape)
    print("输出形状:", output.shape)

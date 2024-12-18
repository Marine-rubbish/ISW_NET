import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """
    DoubleConv执行两个连续的卷积层，每个卷积层后可选地进行批量归一化和ReLU激活。

    参数
    ---
    in_channels (int): 输入通道数。
    out_channels (int): 输出通道数。
    
    功能
    ---
    将双重卷积序列应用于输入张量。

    示例
    ---
    ```python
    double_conv = DoubleConv(in_channels=3, out_channels=64)    # 创建DoubleConv实例
    output = double_conv(input_tensor)                          # 执行forward()方法
    ```
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),  # 可选
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),  # 可选
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class SEBlock(nn.Module):
    """
    SEBlock模块实现了Squeeze-and-Excitation (SE) 机制。

    参数
    ---
    channels (int): 输入特征图的通道数。
    reduction (int, 可选): 通道缩减比例，默认为16。

    功能
    ---
    通过全局平均池化获取通道的全局信息，经过两个全连接层和激活函数，生成每个通道的权重系数。
    使用Sigmoid函数将权重归一化至0到1之间，对输入特征图的每个通道进行加权，增强有用特征，抑制无用特征。

    示例
    ---
    ```python
    se_block = SEBlock(channels=64)     # 创建SEBlock实例
    output = se_block(input_tensor)     # 执行forward()方法
    ```
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        # print(y.size())
        y = self.fc(y).view(b, c, 1, 1)
        # print(y.size(), x.size())
        return x * y.expand_as(x)
    
class ISW_Net(nn.Module):
    """
    ISW_Net类实现了 ISW_Net 网络结构。

    参数
    ---
    in_channels (int, 可选): 输入通道数，默认为3。
    out_channels (int, 可选): 输出通道数，默认为1。

    功能
    ---
    实现 ISW_Net 网络结构，包括下采样路径（包括SE块）、中间层和上采样路径。

    示例
    ---
    ```python
    isw_net = ISW_Net(in_channels=3, out_channels=1)    # 创建 ISW_Net 实例
    output = isw_net(input_tensor)                       # 执行 forward() 方法
    ```
    """
    def __init__(self, in_channels=3, out_channels=1):
        super(ISW_Net, self).__init__()
        self.in_conv = DoubleConv(in_channels, 64)
        
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128),
            SEBlock(128)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256),
            SEBlock(256)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(256, 512),
            SEBlock(512)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(512, 1024),
            SEBlock(1024)
        )

        self.horizon = nn.Sequential(
            DoubleConv(1024, 1024),
        )

        # 上采样层（仅包含ConvTranspose2d）
        self.up1_conv = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.up2_conv = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up3_conv = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up4_conv = nn.ConvTranspose2d(128, 64, 2, stride=2)

        # DoubleConv应用于拼接后的特征图
        self.up1_conv2 = DoubleConv(1024, 512)
        self.up2_conv2 = DoubleConv(512, 256)
        self.up3_conv2 = DoubleConv(256, 128)
        self.up4_conv2 = DoubleConv(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # 下采样路径
        c1 = self.in_conv(x)
        c2 = self.down1(c1)
        c3 = self.down2(c2)
        c4 = self.down3(c3)
        c5 = self.down4(c4)

        # 中间层
        h = self.horizon(c5)

        # 上采样路径
        u1 = self.up1_conv(h)
        u1 = torch.cat([u1, c4], dim=1)
        u1 = self.up1_conv2(u1)

        u2 = self.up2_conv(u1)
        u2 = torch.cat([u2, c3], dim=1)
        u2 = self.up2_conv2(u2)

        u3 = self.up3_conv(u2)
        u3 = torch.cat([u3, c2], dim=1)
        u3 = self.up3_conv2(u3)

        u4 = self.up4_conv(u3)
        u4 = torch.cat([u4, c1], dim=1)
        u4 = self.up4_conv2(u4)

        output = self.out_conv(u4)
        return output
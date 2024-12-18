import os
import random
import time 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import geopandas as gpd
from shapely.geometry import mapping
import rasterio
from rasterio import features
import glob
import matplotlib.pyplot as plt

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置随机种子
set_seed(42)

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

class UNet(nn.Module):
    """
    UNet类实现了 UNet 网络结构。

    参数
    ---
    in_channels (int, 可选): 输入通道数，默认为3。
    out_channels (int, 可选): 输出通道数，默认为1。

    功能
    ---
    实现 UNet 网络结构，包括下采样路径（包括SE块）、中间层和上采样路径。

    示例
    ---
    ```python
    unet = UNet(in_channels=3, out_channels=1)    # 创建 UNet 实例
    output = unet(input_tensor)                   # 执行 forward() 方法
    ```
    """
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
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
    
def load_image(image_path):
    """
    使用 PIL.Image 加载 PNG 图像。
    """
    image = Image.open(image_path).convert('RGB')  # 确保图像为 RGB 格式
    return image

class MODISDataset(Dataset):
    """
    MODISDataset类继承自 torch.utils.data.Dataset 类，用于加载 MODIS 图像和标签。

    参数
    ---
    image_paths (list): 包含图像文件路径的列表。
    label_paths (list): 包含标签文件路径的列表。
    transform (callable, 可选): 应用于图像和标签的转换函数，默认为 None。

    示例
    ---
    ```python
    image_folder = 'path/to/image/folder'
    label_folder = 'path/to/label/folder'
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')]
    label_paths = [os.path.join(label_folder, f) for f in os.listdir(label_folder) if f.endswith('.png')]
    dataset = MODISDataset(image_paths, label_paths, transform=transform)
    ```
    """
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图像
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        # 加载标签
        label_path = self.label_paths[idx]
        label = Image.open(label_path).convert('L')  # 标签为灰度图

        # 应用转换
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label
    
class MCCLoss(nn.Module):
    """
    MCCLoss类实现了基于 Matthews 相关系数的损失函数。

    参数
    ---
    无
    
    功能
    ---
    计算 Matthews 相关系数的损失。

    示例
    ---
    ```python
    criterion = MCCLoss()    # 创建 MCCLoss 实例
    loss = criterion(logits, labels)    # 计算损失
    ```
    """
    def __init__(self):
        super(MCCLoss, self).__init__()

    def forward(self, logits, labels):
        logits = torch.sigmoid(logits)
        labels = labels.float()
        
        tp = (logits * labels).sum(dim=(1, 2, 3))
        tn = ((1 - logits) * (1 - labels)).sum(dim=(1, 2, 3))
        fp = (logits * (1 - labels)).sum(dim=(1, 2, 3))
        fn = ((1 - logits) * labels).sum(dim=(1, 2, 3))
        
        numerator = (tp * tn - fp * fn)
        denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = numerator / (denominator + 1e-5)
        
        return 1 - mcc.mean()
    
def train_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, criterion: torch.nn.Module, 
                optimizer: torch.optim.Optimizer, num_epochs: int, latest_epoch: int):
    """
    训练模型。

    参数
    ---
    model torch.nn.Module: 
        要训练的模型。
    dataloader torch.utils.data.DataLoader: 
        用于训练的数据加载器。
    criterion torch.nn.Module: 
        损失函数。
    optimizer torch.optim.Optimizer: 
        优化器。
    num_epochs int: 
        训练的轮数。
    latest_epochd int: 
        最新模型已经训练过的次数.

    示例
    ---
    ```python
    train_model(model, dataloader, criterion, optimizer, num_epochs)
    ```
    """
    model.train()
    loss_list = np.zeros(num_epochs)
    lr_list = np.zeros(num_epochs)
    for epoch in range(latest_epoch, num_epochs):
        epoch_start = time.time()
        print(f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        loss_list[epoch] = avg_loss
        lr_list[epoch] = optimizer.param_groups[0]['lr']
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), f'./model/UNet_epoch_{epoch+1}.pth')
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, lr: {optimizer.param_groups[0]['lr']}")
        print(f"Epoch time: {time.time() - epoch_start:.0f}s")
    np.save('loss_list.npy', loss_list)
    np.save('lr_list.npy', lr_list)
    return loss_list, lr_list
    
# 可视化损失函数变化
def plot_loss_curve(loss_list):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_list, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
    
# 定义数据转换
transform = transforms.Compose([
    # transforms.Resize((256, 256)),
    transforms.ToTensor()
])

if __name__ == '__main__':
    # 图像和标签路径列表
    image_folder = r'E:\内波数据集-大数据中心上传\IW_images_seg'
    label_folder = r'E:\内波数据集-大数据中心上传\IW_images_label'
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')]
    label_paths = [os.path.join(label_folder, f) for f in os.listdir(label_folder) if f.endswith('.png')]

    # 创建数据集
    dataset = MODISDataset(image_paths, label_paths, transform=transform)

    # 创建数据加载器
    batch_size = 16
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    model = UNet(in_channels=3, out_channels=1).to(device)  # 根据实际图像通道数调整 in_channels
    model_dir = '../model'
    model_files = glob.glob(os.path.join(model_dir, 'UNet_epoch_*.pth'))

    if model_files:
        latest_model = max(model_files, key=lambda x: int(x.split('_')[-1].split('.pth')[0]))
        latest_epoch = int(latest_model.split('_')[-1].split('.pth')[0])
        model.load_state_dict(torch.load(latest_model, weights_only=True))
        print(f"Loaded model from {latest_model}")
    else:
        latest_epoch = 0
        print("No existing model found. Creating a new model.")

    criterion = MCCLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 开始训练
    num_epochs = 1000
    start_time = time.time()
    loss_list, lr_list = train_model(model, dataloader, criterion, optimizer, num_epochs, latest_epoch)
    print(f"Training time: {time.time() - start_time:.0f}s")

    # 调用可视化函数
    plot_loss_curve(loss_list)
# 创建数据集
# dataset = CustomDataset(image_paths, label_paths, transform=transform)



# def test_model(model, image_path, shapefile_path, transform):
#     model.eval()
#     with torch.no_grad():
#         # 加载图像
#         image, profile = load_modis_image(image_path)
#         image = np.transpose(image, (1, 2, 0))
#         image_pil = Image.fromarray(image.astype(np.uint8))
#         input_image = transform(image_pil).unsqueeze(0).to(device)

#         # 预测
#         output = model(input_image)
#         output = torch.sigmoid(output)
#         output_np = output.cpu().squeeze().numpy()
#         pred_mask = (output_np > 0.5).astype(np.uint8)

#         # 加载真实掩码
#         true_mask = shapefile_to_mask(shapefile_path, profile)
        
#         # 可视化结果
#         plt.figure(figsize=(15,5))
#         plt.subplot(1,3,1)
#         plt.imshow(image_pil)
#         plt.title('Original Image')
#         plt.axis('off')

#         plt.subplot(1,3,2)
#         plt.imshow(pred_mask, cmap='gray')
#         plt.title('Predicted Mask')
#         plt.axis('off')

#         plt.subplot(1,3,3)
#         plt.imshow(true_mask, cmap='gray')
#         plt.title('Ground Truth Mask')
#         plt.axis('off')

#         plt.show()

# # 测试
# test_image_path = 'path/to/test_image.tif'
# test_shapefile_path = 'path/to/test_shapefile.shp'
# test_model(model, test_image_path, test_shapefile_path, transform)
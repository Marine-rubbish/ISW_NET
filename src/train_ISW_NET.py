import os
import random
import time 
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt

from utils.MODISDataset import *
from utils.ISW_NET import ISW_Net
from utils.MCCLoss import MCCLoss

def set_seed(seed):
    """
    设置随机种子。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def load_image(image_path):
    """
    使用 PIL.Image 加载 PNG 图像。
    """
    image = Image.open(image_path).convert('RGB')  # 确保图像为 RGB 格式
    return image

def load_checkpoint(model, optimizer, checkpoint_path):
    """
    加载模型检查点。
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss_list = checkpoint['loss_list']
    lr_list = checkpoint['lr_list']
    return model, optimizer, epoch, loss_list, lr_list
    
def train_model(model: torch.nn.Module, images_path: str, labels_path: str,
                criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                checkpoint_path: str, num_epochs: int, batch_size: int = 16, best_loss: float = float('inf')):
    """
    训练模型。

    参数
    ---
    model torch.nn.Module: 
        要训练的模型。
    images_path str:
        图像文件路径。
    labels_path str:
        标签文件路径。
    augmented_dataloader torch.utils.data.DataLoader:
        用于训练的增强数据加载器。
    criterion torch.nn.Module: 
        损失函数。
    optimizer torch.optim.Optimizer: 
        优化器。
    checkpoint_path str:
        保存模型检查点的路径。
    num_epochs int: 
        训练的轮数。
    batch_size int:
        批处理大小，默认为 16。
    best_loss float:
        最佳损失，默认为正无穷。

    示例
    ---
    ```python
    train_model(model, image_paths, label_paths, augmented_dataloader, criterion, optimizer, checkpoint_path, num_epochs)
    ```
    """
    # 开始训练模式
    model.train()

    # 初始化变量
    loss_list = np.zeros(num_epochs)
    lr_list = np.zeros(num_epochs)
    model_files = glob.glob(os.path.join(checkpoint_path, 'checkpoint_epoch_*.pth'))

    # 加载最新的检查点
    if model_files:
        model, optimizer, latest_epoch, loss_list, lr_list = load_checkpoint(model, optimizer, model_files[-1])
        print(f"Loaded checkpoint from {model_files}, starting at epoch {latest_epoch}")
    else:
        latest_epoch = 0
        print("No checkpoint found, starting training from epoch 0")

    # 创建增强数据集
    aug_dataset = get_augmentation_dataset(images_path, labels_path)

    # 创建数据加载器
    dataloader = DataLoader(aug_dataset, batch_size=batch_size, shuffle=True)
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
        avg_loss = epoch_loss / (len(dataloader))
        loss_list[epoch] = avg_loss
        lr_list[epoch] = optimizer.param_groups[0]['lr']
        # 保存Checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_list': loss_list,
            'lr_list': lr_list
        }
        torch.save(checkpoint, os.path.join(checkpoint_path, f'checkpoint_epoch_{epoch+1}.pth'))

        # 保存最佳模型（基于平均损失）
        if avg_loss < best_loss:
            best_loss = avg_loss
            if not os.path.exists(checkpoint_path):
                 os.makedirs(checkpoint_path)
            torch.save(model.state_dict(), os.path.join(checkpoint_path, 'best_model.pth'))
            print(f"Checkpoint saved at epoch {epoch+1}")
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), f'./model/UNet_epoch_{epoch+1}.pth')

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, lr: {optimizer.param_groups[0]['lr']}")
        print(f"Epoch time: {time.time() - epoch_start:.0f}s")
        np.save('loss_list.npy', loss_list)
        # np.save('lr_list.npy', lr_list)
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

if __name__ == '__main__':
    # 设置随机种子
    set_seed(42)

    # 图像和标签路径列表
    image_folder = r'E:\内波数据集-大数据中心上传\IW_images_seg'
    label_folder = r'E:\内波数据集-大数据中心上传\IW_images_label'
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')]
    label_paths = [os.path.join(label_folder, f) for f in os.listdir(label_folder) if f.endswith('.png')]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # 创建数据加载器
    batch_size = 16

    # 创建模型、损失函数和优化器
    model = ISW_Net(in_channels=3, out_channels=1).to(device)  # 根据实际图像通道数调整 in_channels
    model_dir = './model'   # 相对工作路径而言的路径
    model_files = glob.glob(os.path.join(model_dir, 'UNet_epoch_*.pth'))
    checkpoint_path = './model/checkpoints'

    criterion = MCCLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 开始训练
    num_epochs = 1000
    start_time = time.time()
    loss_list, lr_list = train_model(model, image_paths, label_paths, criterion, 
                                     optimizer, checkpoint_path, num_epochs)
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
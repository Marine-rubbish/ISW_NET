import os
import random
import time 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt

from utils.MODISDataset import MODISDataset
from utils.ISW_NET import ISW_Net
from utils.MCCLoss import MCCLoss

def set_seed(seed):
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
    # 设置随机种子
    set_seed(42)

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
    model = ISW_Net(in_channels=3, out_channels=1).to(device)  # 根据实际图像通道数调整 in_channels
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
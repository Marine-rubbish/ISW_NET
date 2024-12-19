import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

# def get_training_augmentation():
#     return A.Compose([
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.Rotate(limit=360, p=0.5),
#         A.ShiftScaleRotate(scale_limit=0.2, p=0.5),
#         A.RandomCrop(width=256, height=256),
#         A.RandomBrightnessContrast(p=0.2),
#         A.Resize(width=256, height=256),
#         ToTensorV2()
#     ])

def get_training_augmentation():
    """
    定义训练阶段的图像增强方法。

    返回
    ---
    albumentations.Compose
        包含多种图像增强操作的组合，包括水平翻转、垂直翻转、仿射变换、亮度对比度调整、乘性噪声、调整大小以及将图像转换为张量。

    示例
    ---
    ```python
    transform = get_training_augmentation()
    ```
    """

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(scale=(0.8, 1.2), p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.5),
        A.Resize(width=256, height=256),
        ToTensorV2()
    ])
    

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

        image = np.array(image)
        label = np.array(label)

        # 应用转换
        if self.transform:
            if isinstance(self.transform, A.Compose):
                augmented = self.transform(image=image, mask=label)
                image = augmented['image']
                label = augmented['mask']
                image = image.float()
                label = label.float()
            else:
                # TODO: 处理异常情况
                image = self.transform(image)
                label = self.transform(label)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            label = torch.from_numpy(label).unsqueeze(0).float()

        return image, label
    

# 仅作为测试代码    
if __name__ == '__main__':
    image_folder = r'E:\内波数据集-大数据中心上传\IW_images_seg'
    label_folder = r'E:\内波数据集-大数据中心上传\IW_images_label'
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')]
    label_paths = [os.path.join(label_folder, f) for f in os.listdir(label_folder) if f.endswith('.png')]
    transform = get_training_augmentation()
    dataset = MODISDataset(image_paths, label_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
    for images, masks in dataloader:
        print(images.size(), masks.size())
        break
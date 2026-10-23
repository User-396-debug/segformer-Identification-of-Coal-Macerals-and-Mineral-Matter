import os
import torch
import numpy as np
from PIL import Image
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2


class CoalMicroscopyDataset(Dataset):
    """煤矿显微组分数据集"""

    def __init__(self, images_path, masks_path, transform=None, is_train=True):
        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.is_train = is_train

        # 获取所有图像文件
        all_image_files = sorted([f for f in os.listdir(images_path)
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))])

        # 只保留有对应标签文件的图像
        self.image_files = []
        for img_file in all_image_files:
            name_without_ext = os.path.splitext(img_file)[0]
            mask_file = name_without_ext + ".png"
            mask_path = os.path.join(masks_path, mask_file)

            if os.path.exists(mask_path):
                self.image_files.append(img_file)

        print(f"找到 {len(self.image_files)} 个有标签的图像文件")
    #def __init__(self, images_path, masks_path, transform=None, is_train=True):
        #self.images_path = images_path
       # self.masks_path = masks_path
        #self.transform = transform
        #self.is_train = is_train

        # 获取所有图像文件
        #self.image_files = sorted([f for f in os.listdir(images_path)
                                # if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))])

        #print(f"找到 {len(self.image_files)} 个图像文件")
    
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 读取图像
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_path, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 读取标签
        mask_name = self._get_mask_name(img_name)
        mask_path = os.path.join(self.masks_path, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # 颜色到类别ID映射
        color2id = {
            (0, 0, 0): 0,
            (255, 0, 0): 1,
            (0, 255, 0): 2,
            (0, 0, 255): 3,
            (255, 255, 0): 4
        }
        mask_id = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for rgb, idx in color2id.items():
            matches = np.all(mask == rgb, axis=-1)
            mask_id[matches] = idx

        # 数据增强
        if self.transform:
            augmented = self.transform(image=image, mask=mask_id)
            image = augmented['image']
            mask_id = augmented['mask']

        # 转为 torch 张量
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        mask = torch.from_numpy(mask_id).long()  # 这里mask是[H, W]，不是[H, W, 3]

        return image, mask


    # def __getitem__(self, idx):
    #     # 加载图像
    #     img_name = self.image_files[idx]
    #     img_path = os.path.join(self.images_path, img_name)
    #
    #     # 获取对应的标签文件名
    #     mask_name = self._get_mask_name(img_name)
    #     mask_path = os.path.join(self.masks_path, mask_name)
    #
    #     # 读取图像和标签
    #     image = cv2.imread(img_path)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #
    #     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    #     mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    #
    #     # 定义颜色和类别ID的映射
    #     color2id = {
    #         (0, 0, 0): 0,
    #         (255, 0, 0): 1,
    #         (0, 255, 0): 2,
    #         (0, 0, 255): 3,
    #         (255, 255, 0): 4
    #     }
    #
    #     # 创建类别ID掩码
    #     mask_id = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    #     for rgb, idx in color2id.items():
    #         matches = np.all(mask == rgb, axis=-1)
    #         mask_id[matches] = idx
    #
    #     #mask = torch.from_numpy(mask_id).long()
    #
    #     if image is None:
    #         raise ValueError(f"无法读取图像: {img_path}")
    #     if mask is None:
    #         raise ValueError(f"无法读取标签: {mask_path}")
    #
    #     # 确保图像和标签尺寸一致
    #     if image.shape[:2] != mask.shape[:2]:
    #         # 调整标签尺寸以匹配图像
    #         mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    #
    #     # 应用数据增强
    #     if self.transform:
    #         augmented = self.transform(image=image, mask=mask)
    #         image = augmented['image']
    #         mask = augmented['mask']
    #
    #     # 转换为张量
    #     image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
    #     mask = torch.from_numpy(mask).long()
    #
    #     return image, mask
    # def __getitem__(self, idx):
    #     # 加载图像
    #     img_name = self.image_files[idx]
    #     img_path = os.path.join(self.images_path, img_name)
    #
    #     # 获取对应的标签文件名（假设标签文件名与图像文件名相同或相似）
    #     mask_name = self._get_mask_name(img_name)
    #     mask_path = os.path.join(self.masks_path, mask_name)
    #
    #     # 读取图像和标签
    #     image = cv2.imread(img_path)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #
    #     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    #
    #     if image is None:
    #         raise ValueError(f"无法读取图像: {img_path}")
    #     if mask is None:
    #         raise ValueError(f"无法读取标签: {mask_path}")
    #
    #     # 应用数据增强
    #     if self.transform:
    #         augmented = self.transform(image=image, mask=mask)
    #         image = augmented['image']
    #         mask = augmented['mask']
    #
    #     # 转换为张量
    #     image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
    #     mask = torch.from_numpy(mask).long()
    #
    #     return image, mask
    
    def _get_mask_name(self, img_name):
        """根据图像文件名获取对应的标签文件名"""
        # 这里假设标签文件名与图像文件名相同，但扩展名可能不同
        # 您可以根据实际情况调整这个逻辑
        name_without_ext = os.path.splitext(img_name)[0]
        
        # 尝试不同的扩展名
        for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            mask_name = name_without_ext + ext
            if os.path.exists(os.path.join(self.masks_path, mask_name)):
                return mask_name
        
        # 如果找不到，返回原文件名
        return img_name


def get_transforms(config, is_train=True):
    """获取数据增强变换"""
    if is_train:
        transform = A.Compose([
            A.Resize(config['image']['height'], config['image']['width']),
            A.HorizontalFlip(p=config['augmentation']['train']['horizontal_flip_prob']),
            A.VerticalFlip(p=config['augmentation']['train']['vertical_flip_prob']),
            A.Rotate(limit=config['augmentation']['train']['rotate_limit'], p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=config['augmentation']['train']['brightness_limit'],
                    contrast_limit=config['augmentation']['train']['contrast_limit']
                ),
                A.RandomGamma(),
            ], p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=config['augmentation']['train']['blur_limit']),
                A.MotionBlur(blur_limit=3),
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(var_limit=tuple(config['augmentation']['train']['noise_limit'])),
                A.ISONoise(),
            ], p=0.3),
            A.Normalize(
                mean=config['image']['mean'],
                std=config['image']['std'],
                max_pixel_value=255.0
            ),
        ], additional_targets={'mask': 'mask'})  # 确保标签也被处理
    else:
        transform = A.Compose([
            A.Resize(config['image']['height'], config['image']['width']),
            A.Normalize(
                mean=config['image']['mean'],
                std=config['image']['std'],
                max_pixel_value=255.0
            ),
        ], additional_targets={'mask': 'mask'})

    return transform
# def get_transforms(config, is_train=True):
#     """获取数据增强变换"""
#     if is_train:
#         transform = A.Compose([
#             A.Resize(config['image']['height'], config['image']['width']),
#             A.HorizontalFlip(p=config['augmentation']['train']['horizontal_flip_prob']),
#             A.VerticalFlip(p=config['augmentation']['train']['vertical_flip_prob']),
#             A.Rotate(limit=config['augmentation']['train']['rotate_limit'], p=0.5),
#             A.OneOf([
#                 A.RandomBrightnessContrast(
#                     brightness_limit=config['augmentation']['train']['brightness_limit'],
#                     contrast_limit=config['augmentation']['train']['contrast_limit']
#                 ),
#                 A.RandomGamma(),
#             ], p=0.5),
#             A.OneOf([
#                 A.GaussianBlur(blur_limit=config['augmentation']['train']['blur_limit']),
#                 A.MotionBlur(blur_limit=3),
#             ], p=0.3),
#             A.OneOf([
#                 A.GaussNoise(var=config['augmentation']['train']['noise_limit']),
#                 A.ISONoise(),
#             ], p=0.3),
#             A.Normalize(
#                 mean=config['image']['mean'],
#                 std=config['image']['std'],
#                 max_pixel_value=255.0
#             ),
#         ])
#     else:
#         transform = A.Compose([
#             A.Resize(config['image']['height'], config['image']['width']),
#             A.Normalize(
#                 mean=config['image']['mean'],
#                 std=config['image']['std'],
#                 max_pixel_value=255.0
#             ),
#         ])
#
#     return transform


def create_data_loaders(config):
    """创建训练和验证数据加载器"""
    
    # 获取数据增强
    train_transform = get_transforms(config, is_train=True)
    val_transform = get_transforms(config, is_train=False)
    
    # 创建数据集
    train_dataset = CoalMicroscopyDataset(
        images_path=config['data']['train_images_path'],
        masks_path=config['data']['train_masks_path'],
        transform=train_transform,
        is_train=True
    )
    
    val_dataset = CoalMicroscopyDataset(
        images_path=config['data']['val_images_path'],
        masks_path=config['data']['val_masks_path'],
        transform=val_transform,
        is_train=False
    )

    # 创建数据加载器 - 内存优化版本
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,  # 减少worker数量以节省内存
        pin_memory=False,  # 关闭pin_memory以节省内存
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,  # 减少worker数量以节省内存
        pin_memory=False,  # 关闭pin_memory以节省内存
        drop_last=False
    )
    # # 创建数据加载器
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=config['training']['batch_size'],
    #     shuffle=True,
    #     num_workers=4,
    #     pin_memory=True,
    #     drop_last=True
    # )
    #
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=config['training']['batch_size'],
    #     shuffle=False,
    #     num_workers=4,
    #     pin_memory=True,
    #     drop_last=False
    # )
    
    return train_loader, val_loader


def create_test_loader(config):
    """创建测试数据加载器"""
    test_transform = get_transforms(config, is_train=False)
    
    test_dataset = CoalMicroscopyDataset(
        images_path=config['data']['test_images_path'],
        masks_path=config['data']['test_masks_path'],
        transform=test_transform,
        is_train=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # 测试时使用batch_size=1
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return test_loader


if __name__ == "__main__":
    # 测试数据加载器
    import yaml
    
    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    train_loader, val_loader = create_data_loaders(config)
    
    print(f"训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(val_loader)}")
    
    # 测试一个批次
    for batch_idx, (images, masks) in enumerate(train_loader):
        print(f"图像形状: {images.shape}")
        print(f"标签形状: {masks.shape}")
        print(f"标签值范围: {masks.min().item()} - {masks.max().item()}")
        break 
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import os
import random
from PIL import Image
import yaml

class ImprovedCoalMicroscopyDataset(Dataset):
    """改进的煤矿显微组分数据集"""
    
    def __init__(
        self,
        images_dir,
        masks_dir,
        transform=None,
        is_training=True,
        focus_green_patch=False,
        green_patch_prob=0.5,
        green_patch_min_size=256,
        green_patch_scale=0.6
    ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.is_training = is_training
        self.focus_green_patch = bool(focus_green_patch) and is_training
        self.green_patch_prob = float(green_patch_prob)
        self.green_patch_min_size = int(green_patch_min_size)
        self.green_patch_scale = float(green_patch_scale)
        
        # 颜色到ID的映射
        self.color_to_id = {
            (0, 0, 0): 0,      # 背景 - 黑色
            (255, 0, 0): 1,    # 类别1 - 红色
            (0, 255, 0): 2,    # 类别2 - 绿色
            (0, 0, 255): 3,    # 类别3 - 蓝色
            (255, 255, 0): 4   # 类别4 - 黄色
        }
        
        # 获取图像文件列表及类别存在信息
        self.image_files = []
        self.contains_red = []
        self.contains_green = []
        self.red_indices = []
        self.green_indices = []
        
        files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        for filename in files:
            mask_filename = filename.replace('.jpg', '.png').replace('.jpeg', '.png')
            mask_path = os.path.join(masks_dir, mask_filename)
            if not os.path.exists(mask_path):
                continue
            
            has_red, has_green = self._analyze_mask_classes(mask_path)
            self.image_files.append(filename)
            self.contains_red.append(has_red)
            self.contains_green.append(has_green)
            
            current_idx = len(self.image_files) - 1
            if has_red:
                self.red_indices.append(current_idx)
            if has_green:
                self.green_indices.append(current_idx)
        
        print(f"找到 {len(self.image_files)} 个有效图像文件")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 加载图像
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_filename)
        
        # 处理不同的图像格式
        if img_filename.lower().endswith('.jpg'):
            mask_filename = img_filename.replace('.jpg', '.png')
        else:
            mask_filename = img_filename
        
        mask_path = os.path.join(self.masks_dir, mask_filename)
        
        # 读取图像
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"无法读取图像: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"读取图像出错 {img_path}: {e}")
            # 返回一个默认图像
            image = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # 读取掩码
        try:
            mask = cv2.imread(mask_path)
            if mask is None:
                raise ValueError(f"无法读取掩码: {mask_path}")
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"读取掩码出错 {mask_path}: {e}")
            # 返回一个默认掩码
            mask = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # 绿色ROI聚焦裁剪
        if self.focus_green_patch:
            image, mask = self._apply_green_focus_patch(image, mask)
        
        # 应用变换
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # 将RGB掩码转换为ID掩码
        mask_id = self.rgb_to_id(mask)
        
        # 确保mask_id是numpy数组
        if isinstance(mask_id, np.ndarray):
            mask_id = torch.from_numpy(mask_id).long()
        
        return image, mask_id
    
    def rgb_to_id(self, mask):
        """将RGB掩码转换为ID掩码"""
        # 确保mask是numpy数组
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        
        # 检查mask的维度
        if mask.ndim == 0:
            # 如果mask是标量，创建一个全零掩码
            mask_id = np.zeros((512, 512), dtype=np.uint8)
            return mask_id
        
        if mask.ndim == 2:
            # 如果mask已经是2D，直接返回
            return mask.astype(np.uint8)
        
        if mask.ndim != 3:
            # 如果维度不对，创建默认掩码
            mask_id = np.zeros((512, 512), dtype=np.uint8)
            return mask_id
        
        mask_id = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        
        for color, id_val in self.color_to_id.items():
            # 创建颜色掩码
            color_mask = np.all(mask == color, axis=2)
            mask_id[color_mask] = id_val
        
        return mask_id
    
    def _apply_green_focus_patch(self, image, mask):
        """对包含绿色的图像执行ROI裁剪并放大"""
        if random.random() > self.green_patch_prob:
            return image, mask
        
        mask_id = self.rgb_to_id(mask)
        green_coords = np.argwhere(mask_id == 2)
        if green_coords.size == 0:
            return image, mask
        
        h, w = mask_id.shape
        min_side = min(h, w)
        patch_size = int(min_side * self.green_patch_scale)
        patch_size = max(self.green_patch_min_size, patch_size)
        patch_size = min(patch_size, min_side)
        if patch_size <= 0:
            return image, mask
        
        center_y, center_x = green_coords[random.randrange(len(green_coords))]
        half = patch_size // 2
        y0 = max(0, center_y - half)
        x0 = max(0, center_x - half)
        y1 = min(h, y0 + patch_size)
        x1 = min(w, x0 + patch_size)
        y0 = max(0, y1 - patch_size)
        x0 = max(0, x1 - patch_size)
        
        cropped_image = image[y0:y1, x0:x1]
        cropped_mask = mask[y0:y1, x0:x1]
        if cropped_image.size == 0 or cropped_mask.size == 0:
            return image, mask
        
        target_size = (image.shape[1], image.shape[0])
        resized_image = cv2.resize(cropped_image, target_size, interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(cropped_mask, target_size, interpolation=cv2.INTER_NEAREST)
        return resized_image, resized_mask
    
    def _analyze_mask_classes(self, mask_path):
        """分析掩码中是否包含红色或绿色类别"""
        try:
            mask = cv2.imread(mask_path)
            if mask is None:
                return False, False
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        except Exception:
            return False, False
        
        mask_id = self.rgb_to_id(mask)
        has_red = np.any(mask_id == 1)
        has_green = np.any(mask_id == 2)
        return bool(has_red), bool(has_green)

def get_improved_transforms(config, is_training=True):
    """获取改进的数据变换"""
    
    if is_training:
        # 训练时的增强变换
        transform = A.Compose([
            A.Resize(config['image']['height'], config['image']['width']),
            A.HorizontalFlip(p=config['augmentation']['train']['horizontal_flip_prob']),
            A.VerticalFlip(p=config['augmentation']['train']['vertical_flip_prob']),
            A.RandomRotate90(p=0.5),
            A.Rotate(
                limit=config['augmentation']['train']['rotate_limit'],
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT
            ),
            A.OneOf([
                A.MotionBlur(blur_limit=config['augmentation']['train']['blur_limit']),
                A.MedianBlur(blur_limit=config['augmentation']['train']['blur_limit']),
                A.Blur(blur_limit=config['augmentation']['train']['blur_limit']),
            ], p=0.3),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.PiecewiseAffine(p=0.3),
            ], p=0.2),
            A.OneOf([
                A.GaussNoise(var_limit=tuple(config['augmentation']['train']['noise_limit'])),
                A.ISONoise(),
                A.MultiplicativeNoise(),
            ], p=0.2),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=config['augmentation']['train']['brightness_limit'],
                    contrast_limit=config['augmentation']['train']['contrast_limit']
                ),
                A.HueSaturationValue(),
                A.RGBShift(),
            ], p=0.3),
            A.ElasticTransform(
                alpha=config['augmentation']['train']['elastic_transform_alpha'],
                sigma=config['augmentation']['train']['elastic_transform_sigma'],
                alpha_affine=config['augmentation']['train']['elastic_transform_alpha_affine'],
                p=0.2
            ),
            A.Normalize(
                mean=config['image']['mean'],
                std=config['image']['std']
            ),
            ToTensorV2(),
        ])
    else:
        # 验证/测试时的变换
        transform = A.Compose([
            A.Resize(config['image']['height'], config['image']['width']),
            A.Normalize(
                mean=config['image']['mean'],
                std=config['image']['std']
            ),
            ToTensorV2(),
        ])
    
    return transform

def get_simple_transforms(config, is_training=True):
    """获取简化的数据变换，减少GPU利用率问题"""
    
    if is_training:
        # 简化的训练增强变换
        transform = A.Compose([
            A.Resize(config['image']['height'], config['image']['width']),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=30, p=0.3, border_mode=cv2.BORDER_CONSTANT),
            A.OneOf([
                A.MotionBlur(blur_limit=3),
                A.Blur(blur_limit=3),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            ], p=0.3),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
            A.Normalize(
                mean=config['image']['mean'],
                std=config['image']['std']
            ),
            ToTensorV2(),
        ])
    else:
        # 验证/测试时的变换
        transform = A.Compose([
            A.Resize(config['image']['height'], config['image']['width']),
            A.Normalize(
                mean=config['image']['mean'],
                std=config['image']['std']
            ),
            ToTensorV2(),
        ])
    
    return transform

def create_improved_data_loaders(config):
    """创建改进的数据加载器"""
    
    # 获取变换
    train_transform = get_improved_transforms(config, is_training=True)
    val_transform = get_improved_transforms(config, is_training=False)
    test_transform = get_improved_transforms(config, is_training=False)
    
    # 创建数据集
    train_dataset = ImprovedCoalMicroscopyDataset(
        images_dir=config['data']['train_images_path'],
        masks_dir=config['data']['train_masks_path'],
        transform=train_transform,
        is_training=True
    )
    
    val_dataset = ImprovedCoalMicroscopyDataset(
        images_dir=config['data']['val_images_path'],
        masks_dir=config['data']['val_masks_path'],
        transform=val_transform,
        is_training=False
    )
    
    test_dataset = ImprovedCoalMicroscopyDataset(
        images_dir=config['data']['test_images_path'],
        masks_dir=config['data']['test_masks_path'],
        transform=test_transform,
        is_training=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,  # 减少内存使用
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # 测试数据加载器
    with open('config/config_improved.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    train_loader, val_loader, test_loader = create_improved_data_loaders(config)
    
    # 测试一个批次
    for images, masks in train_loader:
        print(f"图像形状: {images.shape}")
        print(f"掩码形状: {masks.shape}")
        print(f"掩码唯一值: {torch.unique(masks)}")
        break 
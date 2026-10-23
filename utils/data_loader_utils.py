import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import random


def custom_collate_fn(batch):
    """自定义collate函数，强制统一图像尺寸"""
    images, masks, class_labels, img_names = zip(*batch)

    # 强制调整所有图像到512x512
    processed_images = []
    processed_masks = []

    for img, mask in zip(images, masks):
        # 确保数据类型正确
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()

        # 强制调整尺寸
        if img.shape[1:] != (512, 512):
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0),
                size=(512, 512),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        if mask.shape != (512, 512):
            # 确保掩码是float类型进行插值
            mask_float = mask.float()
            mask_resized = torch.nn.functional.interpolate(
                mask_float.unsqueeze(0).unsqueeze(0),
                size=(512, 512),
                mode='nearest'
            ).squeeze(0).squeeze(0)
            # 转换回long类型
            mask = mask_resized.long()

        # 最终确保数据类型和设备正确
        if mask.dtype != torch.long:
            mask = mask.long()

        processed_images.append(img)
        processed_masks.append(mask)

    # 堆叠张量
    images = torch.stack(processed_images, 0)
    masks = torch.stack(processed_masks, 0)
    class_labels = torch.stack(class_labels, 0)

    return images, masks, class_labels, img_names


def rgb_to_class_id(rgb_image):
    """将RGB掩码转换为类别ID"""
    # 定义RGB颜色到类别的映射
    color_to_class = {
        (0, 0, 0): 0,  # 黑色 - 背景
        (255, 0, 0): 1,  # 红色 - 类别1
        (0, 255, 0): 2,  # 绿色 - 类别2
        (0, 0, 255): 3,  # 蓝色 - 类别3
        (255, 255, 0): 4,  # 黄色 - 类别4
    }

    # 创建类别掩码
    class_mask = np.zeros(rgb_image.shape[:2], dtype=np.uint8)

    # 遍历每个颜色映射
    for color, class_id in color_to_class.items():
        # 找到匹配的像素（允许一定的容差）
        mask = np.all(np.abs(rgb_image - np.array(color)) < 10, axis=2)
        class_mask[mask] = class_id

    return class_mask


class CoalMicroscopyDataset(Dataset):
    """煤炭显微镜图像数据集"""

    def __init__(self, image_dir, mask_dir, transform=None, class2_aware=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.class2_aware = class2_aware

        # 获取所有图像文件
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_files.sort()

        print(f"找到 {len(self.image_files)} 个图像文件")

        # 计算类别2比例用于加权采样
        if class2_aware:
            self.class2_ratios = self._compute_class2_ratios()

    def _compute_class2_ratios(self):
        """计算每个图像中类别2的像素比例"""
        ratios = []
        print("计算类别2像素比例...")

        for img_file in self.image_files:
            mask_path = os.path.join(self.mask_dir, img_file)
            if os.path.exists(mask_path):
                # 读取RGB掩码而不是灰度
                mask_rgb = cv2.imread(mask_path, cv2.IMREAD_COLOR)
                if mask_rgb is not None:
                    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
                    # 转换为类别掩码
                    mask = rgb_to_class_id(mask_rgb)
                else:
                    mask = None

                if mask is not None:
                    try:
                        # 计算类别2(绿色)的像素比例
                        class2_pixels = np.sum(mask == 2)
                        total_pixels = mask.size
                        ratio = class2_pixels / total_pixels if total_pixels > 0 else 0
                        ratios.append(ratio)
                    except Exception as e:
                        print(f"计算类别2比例时出错 {img_file}: {e}")
                        ratios.append(0)
                else:
                    ratios.append(0)
            else:
                ratios.append(0)

        ratios_array = np.array(ratios)
        print(f"类别2比例统计: 最小={ratios_array.min():.4f}, 最大={ratios_array.max():.4f}, 平均={ratios_array.mean():.4f}")
        return ratios_array

    def _get_class2_label(self, mask):
        """根据掩码计算类别2标签"""
        # 检查mask是否为None或空
        if mask is None:
            return 0

        # 确保mask是numpy数组
        if not isinstance(mask, np.ndarray):
            return 0

        # 检查mask是否为空
        if mask.size == 0:
            return 0

        try:
            class2_pixels = np.sum(mask == 2)
            total_pixels = mask.size
            ratio = class2_pixels / total_pixels if total_pixels > 0 else 0

            # 如果类别2像素比例超过0.1%，标记为包含类别2（降低阈值）
            return 1 if ratio > 0.001 else 0
        except Exception as e:
            print(f"计算类别2标签时出错: {e}")
            return 0

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 类别2感知采样
        if self.class2_aware and hasattr(self, 'class2_ratios'):
            # 根据类别2比例进行加权采样
            weights = self.class2_ratios + 0.1  # 添加基础权重避免为0
            weights = weights / weights.sum()
            idx = np.random.choice(len(self.image_files), p=weights)

        img_file = self.image_files[idx]

        # 加载图像
        img_path = os.path.join(self.image_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"警告: 无法加载图像 {img_path}")
            # 创建默认图像
            image = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 加载掩码 - 尝试多种扩展名
        mask_paths = [
            os.path.join(self.mask_dir, img_file),
            os.path.join(self.mask_dir, img_file.replace('.jpg', '.png')),
            os.path.join(self.mask_dir, img_file.replace('.jpg', '.tiff')),
            os.path.join(self.mask_dir, img_file.replace('.jpg', '.tif'))
        ]

        mask = None
        for mask_path in mask_paths:
            if os.path.exists(mask_path):
                # 读取RGB掩码
                mask_rgb = cv2.imread(mask_path, cv2.IMREAD_COLOR)
                if mask_rgb is not None:
                    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
                    # 转换为类别掩码
                    mask = rgb_to_class_id(mask_rgb)
                else:
                    mask = None
                if mask is not None:
                    break

        if mask is None:
            print(f"警告: 无法加载掩码 {img_file}")
            # 创建默认掩码（全为背景类别0）
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # 应用变换
        if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']

        # 计算类别2标签
        class2_label = self._get_class2_label(mask)

        # 转换为张量
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
        else:
            # 确保已经是张量的掩码也是long类型
            if mask.dtype != torch.long:
                    mask = mask.long()

        if not isinstance(class2_label, torch.Tensor):
            class2_label = torch.tensor(class2_label, dtype=torch.long)

        return image, mask, class2_label, img_file


def create_advanced_transforms(config):
    """创建高级数据增强变换"""

    # 训练集变换 - 强增强
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=45, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.RandomScale(scale_limit=0.2, p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        A.OpticalDistortion(distort_limit=0.3, shift_limit=0.05, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Blur(blur_limit=3, p=0.3),
        A.MotionBlur(blur_limit=3, p=0.3),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
        A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.3),
        A.PiecewiseAffine(scale=(0.01, 0.05), nb_rows=4, nb_cols=4, p=0.3),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
        A.Resize(512, 512, interpolation=cv2.INTER_NEAREST),  # 移到最后，确保尺寸统一
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # 验证集变换 - 轻增强
    val_transform = A.Compose([
        A.Resize(512, 512),  # 确保所有图像都是512x512
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    return train_transform, val_transform


def create_class2_aware_data_loaders(config, train_transform=None, val_transform=None):
    """创建类别2感知的数据加载器"""

    # 数据路径
    data_config = config['data']
    train_image_dir = data_config.get('train_images_path', data_config.get('train_images', 'data/train/images'))
    train_mask_dir = data_config.get('train_masks_path', data_config.get('train_masks', 'data/train/masks'))
    val_image_dir = data_config.get('val_images_path', data_config.get('val_images', 'data/val/images'))
    val_mask_dir = data_config.get('val_masks_path', data_config.get('val_masks', 'data/val/masks'))
    test_image_dir = data_config.get('test_images_path', data_config.get('test_images', 'data/test/images'))
    test_mask_dir = data_config.get('test_masks_path', data_config.get('test_masks', 'data/test/masks'))

    # 创建变换
    if train_transform is None or val_transform is None:
        train_transform, val_transform = create_advanced_transforms(config)

    # 创建数据集
    train_dataset = CoalMicroscopyDataset(
        train_image_dir, train_mask_dir,
        transform=train_transform, class2_aware=True
    )

    val_dataset = CoalMicroscopyDataset(
        val_image_dir, val_mask_dir,
        transform=val_transform, class2_aware=False
    )

    test_dataset = CoalMicroscopyDataset(
        test_image_dir, test_mask_dir,
        transform=val_transform, class2_aware=False
    )

    # 训练参数
    training_config = config['training']
    batch_size = training_config.get('batch_size', 4)
    num_workers = config.get('num_workers', 2)

    # 创建数据加载器 - 使用自定义collate函数
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn  # 使用自定义collate函数
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn  # 使用自定义collate函数
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn  # 使用自定义collate函数
    )

    print(f"数据加载器创建完成:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")

    return train_loader, val_loader, test_loader


def debug_dataset_samples(dataset, num_samples=5):
    """调试数据集样本"""
    print(f"\n🔍 调试数据集样本 (前{num_samples}个):")

    for i in range(min(num_samples, len(dataset))):
        try:
            image, mask, class2_label, img_name = dataset[i]

            print(f"\n样本 {i}: {img_name}")
            print(f"  图像形状: {image.shape}, 数据类型: {image.dtype}")
            print(f"  掩码形状: {mask.shape}, 数据类型: {mask.dtype}")
            print(f"  类别2标签: {class2_label}, 类型: {type(class2_label)}")
            print(f"  图像值范围: [{image.min():.3f}, {image.max():.3f}]")
            print(f"  掩码唯一值: {torch.unique(mask)}")

            # 检查掩码值范围
            mask_min, mask_max = mask.min().item(), mask.max().item()
            if mask_min < 0 or mask_max > 4:
                print(f"  ⚠️ 警告: 掩码值超出预期范围[0,4]")

        except Exception as e:
            print(f"❌ 样本 {i} 调试失败: {e}")


def create_simple_transforms():
    """创建简单的数据变换（用于测试）"""
    train_transform = A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    return train_transform, val_transform


class GreenClassAugmentation:
    """针对绿色类别的专项数据增强"""
    
    def __init__(self, base_transform):
        self.base_transform = base_transform
        # 绿色RGB值：(0, 255, 0)
        self.green_color = np.array([0, 255, 0])
        # 更温和的绿色专项增强，避免破坏绿色像素
        self.green_enhance = A.Compose([
            A.HueSaturationValue(
                hue_shift_limit=6,
                sat_shift_limit=20,
                val_shift_limit=6,
                p=0.6
            ),
            A.RGBShift(
                r_shift_limit=4,
                g_shift_limit=8,
                b_shift_limit=4,
                p=0.4
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.4
            ),
        ], additional_targets={'mask': 'mask'})
    
    def __call__(self, image, mask):
        """
        对包含绿色类别的图像应用专项增强
        Args:
            image: numpy array, shape (H, W, 3), RGB格式
            mask: numpy array, shape (H, W, 3), RGB格式（绿色类别RGB=(0,255,0)）
        """
        # 检查mask是否为RGB格式（3通道）
        if mask.ndim == 3 and mask.shape[2] == 3:
            # 检查是否包含绿色类别（RGB值接近(0, 255, 0)）
            # 允许一定的容差（±10）
            green_mask = np.all(np.abs(mask - self.green_color) < 10, axis=2)
            has_green = np.any(green_mask)
        else:
            # 如果是ID格式（2D），检查类别2
            has_green = np.any(mask == 2)
        
        if has_green:
            # 应用温和的绿色专项增强（仅颜色空间调整）
            augmented = self.green_enhance(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # 应用基础增强（几何变换等）
        result = self.base_transform(image=image, mask=mask)
        return result


def create_balanced_transforms(enable_green_augmentation=True):
    """
    创建平衡的数据增强配置，减少过拟合
    Args:
        enable_green_augmentation: 是否启用绿色类别专项增强
    """
    # 根据是否启用绿色专项增强，动态调整几何增强强度
    horiz_flip_prob = 0.5
    vert_flip_prob = 0.5
    rotate90_prob = 0.5
    rotate_limit = 15
    rotate_prob = 0.3
    blur_prob = 0.2
    hue_shift_limit = 10
    sat_shift_limit = 15
    val_shift_limit = 10
    
    if enable_green_augmentation:
        horiz_flip_prob = 0.4
        vert_flip_prob = 0.25
        rotate90_prob = 0.3
        rotate_limit = 8
        rotate_prob = 0.2
        blur_prob = 0.1
        hue_shift_limit = 8
        sat_shift_limit = 12
        val_shift_limit = 8
    
    # 基础增强（几何变换等）
    base_train_transform = A.Compose([
        A.HorizontalFlip(p=horiz_flip_prob),
        A.VerticalFlip(p=vert_flip_prob),
        A.RandomRotate90(p=rotate90_prob),
        A.Rotate(limit=rotate_limit, p=rotate_prob, border_mode=cv2.BORDER_CONSTANT),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.HueSaturationValue(hue_shift_limit=hue_shift_limit,
                             sat_shift_limit=sat_shift_limit,
                             val_shift_limit=val_shift_limit,
                             p=0.3),
        A.RandomGamma(gamma_limit=(95, 105), p=0.2),
        A.Blur(blur_limit=3, p=blur_prob),
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})
    
    # 如果启用绿色类别专项增强，则包装基础增强
    if enable_green_augmentation:
        train_transform = GreenClassAugmentation(base_train_transform)
    else:
        train_transform = base_train_transform

    val_transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})
    
    return train_transform, val_transform


def validate_data_paths(config):
    """验证数据路径是否存在"""
    data_config = config['data']

    # 修复：使用正确的键名
    paths_to_check = [
        ('训练图像', data_config.get('train_images_path', data_config.get('train_images', ''))),
        ('训练掩码', data_config.get('train_masks_path', data_config.get('train_masks', ''))),
        ('验证图像', data_config.get('val_images_path', data_config.get('val_images', ''))),
        ('验证掩码', data_config.get('val_masks_path', data_config.get('val_masks', ''))),
        ('测试图像', data_config.get('test_images_path', data_config.get('test_images', ''))),
        ('测试掩码', data_config.get('test_masks_path', data_config.get('test_masks', '')))
    ]

    print("🔍 验证数据路径:")
    all_valid = True

    for name, path in paths_to_check:
        if path and os.path.exists(path):
            file_count = len([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"  ✅ {name}: {path} ({file_count} 文件)")
        else:
            print(f"  ❌ {name}: {path} (不存在)")
            all_valid = False

    return all_valid


if __name__ == "__main__":
    # 测试数据加载器
    print("测试数据加载器...")

    # 创建测试配置
    test_config = {
        'data': {
            'train_images': 'data/train/images',
            'train_masks': 'data/train/masks',
            'val_images': 'data/val/images',
            'val_masks': 'data/val/masks',
            'test_images': 'data/test/images',
            'test_masks': 'data/test/masks'
        },
        'training': {
            'batch_size': 2,
            'num_workers': 2
        }
    }

    # 验证路径
    if validate_data_paths(test_config):
        # 创建数据加载器
        train_loader, val_loader, test_loader = create_class2_aware_data_loaders(test_config)

        # 测试数据加载
        print("\n测试数据加载:")
        for batch_idx, (images, masks, class2_labels, img_names) in enumerate(train_loader):
            print(f"批次 {batch_idx}:")
            print(f"  图像形状: {images.shape}")
            print(f"  掩码形状: {masks.shape}")
            print(f"  类别2标签形状: {class2_labels.shape}")
            print(f"  图像名称: {img_names[:2]}")
            break
    else:
        print("数据路径验证失败，请检查配置文件")
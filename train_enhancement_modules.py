"""
训练增强模块脚本
冻结SegFormer主干网络，只训练边缘增强和颜色注意力模块
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np

# 设置Windows控制台编码为UTF-8（解决emoji显示问题）
if sys.platform == 'win32':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.segformer_model import create_segformer
from models.segformer_enhancements import create_enhanced_segformer
from utils.improved_data_loader import ImprovedCoalMicroscopyDataset
from utils.data_loader_utils import create_balanced_transforms
from utils.class_balanced_sampler import ClassBalancedBatchSampler


def freeze_backbone(model, verbose=False):
    """
    冻结主干网络（SegFormer的encoder和decoder）
    
    Args:
        model: EnhancedSegFormer模型
        verbose: 是否打印详细信息
    """
    frozen_params = 0
    trainable_params = 0
    
    # 冻结原始SegFormer模型的所有参数
    for name, param in model.original_model.named_parameters():
        param.requires_grad = False
        frozen_params += param.numel()
    
    # 保持增强模块可训练
    if hasattr(model, 'edge_enhance'):
        for name, param in model.edge_enhance.named_parameters():
            param.requires_grad = True
            trainable_params += param.numel()
            if verbose:
                print(f"✅ 训练: edge_enhance.{name} ({param.numel():,} 参数)")
    
    if hasattr(model, 'color_attention'):
        for name, param in model.color_attention.named_parameters():
            param.requires_grad = True
            trainable_params += param.numel()
            if verbose:
                print(f"✅ 训练: color_attention.{name} ({param.numel():,} 参数)")
    
    print(f"📊 参数统计: 冻结={frozen_params:,}, 可训练={trainable_params:,} ({trainable_params / (frozen_params + trainable_params) * 100:.2f}%)")
    
    return frozen_params, trainable_params


def get_trainable_parameters(model, freeze_backbone=True):
    """
    获取所有可训练的参数
    
    Args:
        model: EnhancedSegFormer模型
        freeze_backbone: 是否冻结主干网络
    Returns:
        trainable_params: 可训练参数列表
    """
    trainable_params = []
    
    # 检查是否有增强模块
    has_edge = hasattr(model, 'edge_enhance') and model.edge_enhance is not None
    has_color = hasattr(model, 'color_attention') and model.color_attention is not None
    
    # 如果冻结主干网络，只训练增强模块
    if freeze_backbone:
        if has_edge:
            trainable_params.extend(model.edge_enhance.parameters())
        if has_color:
            trainable_params.extend(model.color_attention.parameters())
        # 如果有绿色辅助头，也加入
        if hasattr(model, 'green_aux_head') and model.green_aux_head is not None:
            trainable_params.extend(model.green_aux_head.parameters())
    else:
        # 全量训练：训练所有参数
        trainable_params = list(model.parameters())
    
    # 如果没有任何可训练参数，训练整个模型（包括主干网络）
    if len(trainable_params) == 0:
        trainable_params = list(model.parameters())
    
    return trainable_params


def verify_frozen_parameters(model):
    """验证参数是否被正确冻结"""
    print("\n" + "=" * 60)
    print("🔍 验证参数冻结状态")
    print("=" * 60)
    
    frozen_count = 0
    trainable_count = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_count += 1
            if trainable_count <= 10:  # 只显示前10个
                print(f"✅ 可训练: {name}")
        else:
            frozen_count += 1
    
    if trainable_count > 10:
        print(f"... 还有 {trainable_count - 10} 个可训练参数")
    
    print(f"\n📊 统计:")
    print(f"  冻结参数组: {frozen_count}")
    print(f"  可训练参数组: {trainable_count}")
    
    # 验证原始模型是否被冻结
    original_trainable = sum(1 for p in model.original_model.parameters() if p.requires_grad)
    if original_trainable == 0:
        print("✅ 主干网络已正确冻结")
    else:
        print(f"⚠️  警告: 主干网络仍有 {original_trainable} 个可训练参数")


class EnhancementTrainer:
    """增强模块训练器"""
    
    def __init__(self, config_path, pretrained_checkpoint):
        self.config = self.load_config(config_path)
        self.device = torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.pretrained_checkpoint = pretrained_checkpoint
        self.loss_config = self.config.get('loss', {})
        self.use_green_aux_loss = self.loss_config.get('use_green_aux_loss', False)
        self.green_aux_loss_weight = self.loss_config.get(
            'green_aux_loss_weight',
            self.loss_config.get('green_aux_weight', 0.0)
        )
        self.green_aux_pos_weight = self.loss_config.get('green_aux_pos_weight', 50.0)
        self.green_aux_bce = None
        self._green_aux_fallback_warned = False
        if self.use_green_aux_loss:
            pos_weight_tensor = torch.tensor(
                [self.green_aux_pos_weight],
                dtype=torch.float32,
                device=self.device
            )
            self.green_aux_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        
        print(f"🚀 初始化训练器 | 设备: {self.device} | 检查点: {pretrained_checkpoint}")
        
        # 创建模型
        self.model = self.create_model()
        self.model.to(self.device)
        
        # 验证模型输出形状
        test_input = torch.randn(1, 3, 512, 512).to(self.device)
        with torch.no_grad():
            raw_output = self.model(test_input)
        if isinstance(raw_output, (tuple, list)):
            test_output = raw_output[0]
        elif isinstance(raw_output, dict):
            test_output = raw_output.get('final') or raw_output.get('logits') or next(iter(raw_output.values()))
        else:
            test_output = raw_output
        if test_output.shape[1] != 5:
            print(f"❌ 模型输出通道数错误: {test_output.shape[1]} (应为5)")
        
        # 加载预训练权重
        self.load_pretrained_weights()
        
        # 根据配置决定是否冻结主干网络
        self.freeze_backbone = self.config.get('optimization', {}).get('freeze_backbone', True)
        has_edge = hasattr(self.model, 'edge_enhance') and self.model.edge_enhance is not None
        has_color = hasattr(self.model, 'color_attention') and self.model.color_attention is not None
        
        if self.freeze_backbone and (has_edge or has_color):
            freeze_backbone(self.model, verbose=False)
        else:
            for param in self.model.parameters():
                param.requires_grad = True
            if has_edge or has_color:
                print("✅ 全量训练模式（主干网络+增强模块）")
            else:
                print("✅ 全量训练模式（无增强模块）")
        
        # 创建数据加载器
        self.train_loader, self.val_loader = self.create_data_loaders()
        
        # 创建损失函数和优化器（损失函数会在训练循环中根据epoch更新）
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler()
        
        # 损失函数将在训练循环中根据epoch动态创建
        self.criterion = None  # 将在train方法中初始化
        
        # 训练状态
        self.best_val_iou = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_ious = []
        
    def load_config(self, config_path):
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def create_model(self):
        """创建增强模型"""
        # 1. 创建原始SegFormer模型
        model_config = self.config['model']
        original_model = create_segformer(
            model_size=model_config.get('model_size', 'b1'),
            num_classes=self.config['model'].get('num_classes', 5),
            img_size=self.config['data'].get('image_size', 512),
            pretrained=False,
            drop_rate=model_config.get('dropout_rate', 0.1),
            drop_path_rate=model_config.get('drop_path_rate', 0.1)
        )
        
        # 2. 创建增强模型（启用所有增强模块）
        # 从配置文件读取是否启用增强模块
        use_edge = self.config.get('model', {}).get('use_edge_enhancement', True)
        use_color = self.config.get('model', {}).get('use_color_attention', True)
        use_green_aux_head = self.config.get('model', {}).get('use_green_aux_head', False)
        
        enhanced_model = create_enhanced_segformer(
            original_model=original_model,
            use_edge_enhancement=use_edge,  # 启用边缘增强
            use_color_attention=use_color,  # 启用颜色注意力
            use_green_aux_head=use_green_aux_head
        )
        
        freeze_backbone = self.config.get('optimization', {}).get('freeze_backbone', True)
        
        if use_edge and use_color:
            strategy = "冻结主干，只训练增强模块" if freeze_backbone else "全量训练（主干+增强模块）"
            print(f"✅ 已启用增强模块（边缘增强 + 颜色注意力）| 策略: {strategy}")
        elif use_edge:
            strategy = "冻结主干，只训练边缘增强" if freeze_backbone else "全量训练"
            print(f"✅ 已启用边缘增强模块 | 策略: {strategy}")
        elif use_color:
            strategy = "冻结主干，只训练颜色注意力" if freeze_backbone else "全量训练"
            print(f"✅ 已启用颜色注意力模块 | 策略: {strategy}")
        else:
            print("✅ 全量训练模式（无增强模块）")
        
        return enhanced_model
    
    def load_pretrained_weights(self):
        """加载预训练权重（只加载主干网络部分）"""
        if not os.path.exists(self.pretrained_checkpoint):
            raise FileNotFoundError(f"预训练检查点不存在: {self.pretrained_checkpoint}")
        
        print(f"\n📥 加载预训练权重: {self.pretrained_checkpoint}")
        checkpoint = torch.load(self.pretrained_checkpoint, map_location=self.device, weights_only=False)
        
        # 只加载原始模型的权重
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 过滤出原始模型的权重
        original_state_dict = {}
        for key, value in state_dict.items():
            # 只加载原始SegFormer的权重（排除增强模块）
            if key.startswith('original_model.'):
                # 移除original_model前缀，直接加载到original_model
                new_key = key[len('original_model.'):]
                original_state_dict[new_key] = value
            elif not key.startswith('edge_enhance') and not key.startswith('color_attention'):
                # 如果键名是encoder或decoder，直接使用（不需要添加前缀）
                # 这些权重会直接加载到original_model
                original_state_dict[key] = value
        
        # 直接加载到original_model（如果使用EnhancedSegFormer）
        if hasattr(self.model, 'original_model'):
            missing_keys, unexpected_keys = self.model.original_model.load_state_dict(original_state_dict, strict=False)
            print(f"✅ 原始模型权重加载完成")
        else:
            # 如果没有original_model，直接加载到整个模型
            missing_keys, unexpected_keys = self.model.load_state_dict(original_state_dict, strict=False)
        
        print(f"✅ 权重加载完成")
        if missing_keys:
            print(f"⚠️  缺失的键（增强模块，正常）: {len(missing_keys)} 个")
            if len(missing_keys) <= 10:
                for key in missing_keys:
                    print(f"     - {key}")
            else:
                print(f"     ... 还有 {len(missing_keys) - 10} 个")
        if unexpected_keys:
            print(f"⚠️  意外的键: {len(unexpected_keys)} 个")
            if len(unexpected_keys) <= 10:
                for key in unexpected_keys:
                    print(f"     - {key}")
        
        # 显示检查点信息
        if 'val_iou' in checkpoint:
            print(f"📊 预训练模型验证IoU: {checkpoint['val_iou']:.4f}")
        if 'epoch' in checkpoint:
            print(f"📅 预训练模型Epoch: {checkpoint['epoch'] + 1}")
    
    def create_data_loaders(self):
        """创建数据加载器"""
        data_config = self.config['data']
        
        # 创建数据变换（启用绿色类别专项增强）
        enable_green_aug = self.config.get('data', {}).get('enable_green_augmentation', True)
        train_transform, val_transform = create_balanced_transforms(enable_green_augmentation=enable_green_aug)
        
        if enable_green_aug:
            print("✅ 已启用绿色类别专项数据增强")
        
        # 创建数据集
        train_dataset = ImprovedCoalMicroscopyDataset(
            images_dir=data_config['train_images'],
            masks_dir=data_config['train_masks'],
            transform=train_transform,
            is_training=True,
            focus_green_patch=data_config.get('green_focus_patch', False),
            green_patch_prob=data_config.get('green_patch_prob', 0.6),
            green_patch_min_size=data_config.get('green_patch_min_size', 256),
            green_patch_scale=data_config.get('green_patch_scale', 0.6)
        )
        
        val_dataset = ImprovedCoalMicroscopyDataset(
            images_dir=data_config['val_images'],
            masks_dir=data_config['val_masks'],
            transform=val_transform,
            is_training=False
        )
        
        # 创建数据加载器（训练集使用类别平衡采样）
        batch_size = self.config['training']['batch_size']
        sampler_min_red = data_config.get('sampler_min_red', 1)
        sampler_min_green = data_config.get('sampler_min_green', 1)
        sampler_min_red = data_config.get('sampler_min_red', 1)
        sampler_min_green = data_config.get('sampler_min_green', 1)
        use_balanced_sampler = (
            hasattr(train_dataset, 'red_indices') and len(train_dataset.red_indices) > 0 and
            hasattr(train_dataset, 'green_indices') and len(train_dataset.green_indices) > 0
        )

        if use_balanced_sampler:
            sampler = ClassBalancedBatchSampler(
                dataset=train_dataset,
                batch_size=batch_size,
                min_red=sampler_min_red,
                min_green=sampler_min_green,
                shuffle=True
            )
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=sampler,
                num_workers=self.config.get('num_workers', 4),
                pin_memory=True
            )
            print("✅ 启用了类别平衡采样（每个batch包含红色和绿色样本）")
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.config.get('num_workers', 4),
                pin_memory=True
            )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        print(f"\n📚 数据加载器创建完成")
        print(f"  训练集: {len(train_dataset)} 张图像")
        print(f"  验证集: {len(val_dataset)} 张图像")
        print(f"  批次大小: {self.config['training']['batch_size']}")
        
        return train_loader, val_loader

    def _split_model_outputs(self, model_output):
        """兼容主分支和辅助分支的模型输出"""
        green_aux_logits = None

        if isinstance(model_output, tuple):
            main_logits = model_output[0]
            if len(model_output) > 1:
                green_aux_logits = model_output[1]
        elif isinstance(model_output, list):
            main_logits = model_output[0]
            if len(model_output) > 1:
                green_aux_logits = model_output[1]
        elif isinstance(model_output, dict):
            main_logits = (
                model_output.get('final') or
                model_output.get('logits') or
                model_output.get('output') or
                next(iter(model_output.values()))
            )
            green_aux_logits = model_output.get('green_aux')
        else:
            main_logits = model_output

        return main_logits, green_aux_logits
    
    def _compute_loss(self, logits, masks, green_aux_logits):
        """兼容是否支持绿色辅助分支的损失计算"""
        if self.criterion is None:
            raise RuntimeError("损失函数尚未初始化")
        
        # 如果启用绿色辅助损失且模型提供了辅助输出，优先尝试交给损失函数处理
        if self.use_green_aux_loss and green_aux_logits is not None:
            try:
                return self.criterion(logits, masks, green_aux_logits)
            except TypeError as exc:
                # 如果损失函数不支持第三个参数，则回退到手动计算辅助损失
                if "takes 3 positional arguments" not in str(exc):
                    raise
                if not self._green_aux_fallback_warned:
                    print("⚠️  当前损失函数不接受绿色辅助输出，改用手动辅助损失回退方案。")
                    self._green_aux_fallback_warned = True
                base_loss = self.criterion(logits, masks)
                if self.green_aux_bce is not None:
                    green_target = (masks == 2).float().unsqueeze(1)
                    aux_loss = self.green_aux_bce(green_aux_logits, green_target)
                    return base_loss + self.green_aux_loss_weight * aux_loss
                return base_loss
        
        # 默认路径：不使用辅助损失或没有辅助输出
        return self.criterion(logits, masks)
    
    def create_criterion(self, epoch=0):
        """创建损失函数（使用像素级加权Focal Loss + Dice Loss组合）"""
        from utils.advanced_loss_functions import CombinedFocalDiceLoss
        
        loss_config = self.config.get('loss', {})
        
        # 检查是否使用渐进式权重
        use_progressive = loss_config.get('use_progressive_weights', False)
        progressive_start = loss_config.get('progressive_start_epoch', 36)
        
        if use_progressive and epoch >= progressive_start:
            # 使用渐进式权重
            class_weights = loss_config.get('progressive_weights', [1.0, 3.0, 1.2, 1.1, 1.1])
            print(f"🎯 Epoch {epoch}: 启用渐进式权重调整")
        else:
            # 使用标准权重
            class_weights = loss_config.get('class_weights', [1.0, 3.0, 1.2, 1.1, 1.1])
        
        class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)
        focal_gamma = loss_config.get('focal_gamma', 1.5)
        
        # 像素级加权配置：为稀疏类别（绿色、红色）分配更高权重
        # 绿色类别（class_idx=2）像素权重×20，红色类别（class_idx=1）像素权重×10
        sparse_class_weights = {
            2: loss_config.get('green_pixel_weight', 20.0),  # 绿色类别像素权重×20
            1: loss_config.get('red_pixel_weight', 10.0),    # 红色类别像素权重×10
        }
        
        # 组合损失权重配置
        focal_weight = loss_config.get('focal_weight', 0.4)  # Focal Loss权重（降低以给IoU Loss空间）
        dice_weight = loss_config.get('dice_weight', 0.3)    # Dice Loss权重
        iou_weight = loss_config.get('iou_weight', 0.3)      # IoU Loss权重（新增，直接优化IoU指标）
        use_green_aux_loss = loss_config.get('use_green_aux_loss', False)
        green_aux_weight = loss_config.get('green_aux_loss_weight', loss_config.get('green_aux_weight', 0.0))
        green_aux_pos_weight = loss_config.get('green_aux_pos_weight', 50.0)
        
        # IoU Loss的类别权重（可选，用于强调绿色类别）
        iou_class_weights = loss_config.get('iou_class_weights', None)
        if iou_class_weights is None:
            # 默认使用与class_weights相同的权重
            iou_class_weights = class_weights
        
        # 创建组合损失函数
        criterion = CombinedFocalDiceLoss(
            alpha=class_weights_tensor,
            gamma=focal_gamma,
            sparse_class_weights=sparse_class_weights,
            focal_weight=focal_weight,
            dice_weight=dice_weight,
            iou_weight=iou_weight,
            dice_smooth=1e-5,
            use_green_aux_loss=use_green_aux_loss,
            green_aux_weight=green_aux_weight,
            green_aux_pos_weight=green_aux_pos_weight,
            iou_class_weights=iou_class_weights
        )
        
        # 仅在epoch 0打印简要信息
        if epoch == 0:
            print(f"📉 损失函数: Focal({focal_weight})+Dice({dice_weight})+IoU({iou_weight}) | "
                  f"类别权重=[{class_weights[0]:.1f},{class_weights[1]:.1f},{class_weights[2]:.1f},{class_weights[3]:.1f},{class_weights[4]:.1f}] | "
                  f"像素权重: 红×{sparse_class_weights[1]:.1f}, 绿×{sparse_class_weights[2]:.1f}")
        
        return criterion
    
    def create_optimizer(self):
        """创建优化器（根据freeze_backbone配置决定训练范围）"""
        freeze_backbone_flag = getattr(self, 'freeze_backbone', self.config.get('optimization', {}).get('freeze_backbone', True))
        trainable_params = get_trainable_parameters(self.model, freeze_backbone=freeze_backbone_flag)
        
        if len(trainable_params) == 0:
            raise ValueError("❌ 错误: 没有可训练的参数！请至少启用一个增强模块，或者检查模型结构。")
        
        training_config = self.config['training']
        base_lr = training_config.get('learning_rate', 0.0001)
        enhancement_lr = training_config.get('enhancement_learning_rate', base_lr)
        full_model_lr = training_config.get('full_model_learning_rate', base_lr)
        lr = enhancement_lr if freeze_backbone_flag else full_model_lr
        
        optimizer = optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=training_config.get('weight_decay', 0.0001)
        )
        
        total_trainable = sum(p.numel() for p in trainable_params)
        mode_msg = "只训练增强模块（主干网络已冻结）" if freeze_backbone_flag else "全量训练（主干网络+增强模块）"
        
        print(f"\n⚙️  优化器: AdamW")
        print(f"  学习率: {lr}")
        print(f"  权重衰减: {training_config.get('weight_decay', 0.0001)}")
        print(f"  优化参数数量: {total_trainable:,}")
        print(f"  模式: {mode_msg}")
        
        return optimizer
    
    def create_scheduler(self):
        """创建学习率调度器"""
        training_config = self.config['training']
        scheduler_config = training_config.get('scheduler', {})
        
        if scheduler_config.get('type') == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 3),
                min_lr=scheduler_config.get('min_lr', 1e-6),
                verbose=True
            )
            print(f"\n📈 学习率调度器: ReduceLROnPlateau")
            print(f"  模式: max (监控验证IoU)")
            print(f"  因子: {scheduler_config.get('factor', 0.5)}")
            print(f"  耐心: {scheduler_config.get('patience', 3)}")
        else:
            scheduler = None
            print(f"\n📈 学习率调度器: 无")
        
        return scheduler
    
    def train_epoch(self, epoch):
        """训练一个epoch（添加诊断信息）"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        # 第一个batch的诊断信息
        first_batch_diagnosed = False
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['training']['num_epochs']}")
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device).long()
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            logits, green_aux_logits = self._split_model_outputs(outputs)
            
            # 计算损失
            loss = self._compute_loss(logits, masks, green_aux_logits)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if self.config['training'].get('gradient_clip_norm', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    get_trainable_parameters(
                        self.model,
                        freeze_backbone=getattr(self, 'freeze_backbone', True)
                    ),
                    self.config['training']['gradient_clip_norm']
                )
            
            # 更新参数
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, epoch):
        """验证（修复版本，改进IoU计算）"""
        self.model.eval()
        total_loss = 0.0
        total_iou = 0.0
        class_ious = torch.zeros(5).to(self.device)
        class_counts = torch.zeros(5).to(self.device)
        batch_count = 0
        
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc="验证中"):
                images = images.to(self.device)
                masks = masks.to(self.device).long()
                
                # 前向传播
                outputs = self.model(images)
                logits, green_aux_logits = self._split_model_outputs(outputs)
                
                # 计算损失
                loss = self._compute_loss(logits, masks, green_aux_logits)
                total_loss += loss.item()
                
                # 计算IoU
                predictions = torch.argmax(logits, dim=1)
                
                
                iou, class_iou = self.compute_iou(predictions, masks, num_classes=5)
                
                # 只累加有效的IoU值（非NaN）
                if not (isinstance(iou, float) and (iou != iou)):  # 检查NaN
                    total_iou += iou
                    batch_count += 1
                
                # 累加各类别IoU（处理NaN）
                for cls in range(5):
                    cls_iou = class_iou[cls].item()
                    if not (isinstance(cls_iou, float) and (cls_iou != cls_iou)):  # 非NaN
                        class_ious[cls] += class_iou[cls]
                        class_counts[cls] += 1
        
        avg_loss = total_loss / len(self.val_loader)
        avg_iou = total_iou / batch_count if batch_count > 0 else 0.0
        
        # 计算各类别平均IoU（避免除零）
        avg_class_ious = []
        for cls in range(5):
            if class_counts[cls] > 0:
                avg_class_ious.append((class_ious[cls] / class_counts[cls]).item())
            else:
                avg_class_ious.append(0.0)
        avg_class_ious = np.array(avg_class_ious)
        
        self.val_losses.append(avg_loss)
        self.val_ious.append(avg_iou)
        
        return avg_loss, avg_iou, avg_class_ious
    
    def compute_iou(self, pred, target, num_classes=5):
        """计算IoU（修复版本，正确处理空类别和边界情况）"""
        ious = []
        
        for cls in range(num_classes):
            pred_cls = (pred == cls)
            target_cls = (target == cls)
            
            intersection = (pred_cls & target_cls).sum().float()
            union = (pred_cls | target_cls).sum().float()
            
            if union == 0:
                # 如果union为0，有两种情况：
                # 1. 该类别在预测和真实标签中都不存在 -> IoU未定义，设为0（不参与平均）
                # 2. 该类别在真实标签中存在但预测中没有 -> IoU = 0
                # 实际上，如果union=0且target中有该类别，那么pred中肯定没有，所以intersection=0
                # 如果union=0且target中没有该类别，那么pred中也没有，intersection=0
                # 所以union=0时，intersection一定为0
                if target_cls.sum() > 0:
                    # 真实标签中有，但预测中没有，IoU = 0
                    iou = 0.0
                else:
                    # 真实标签中也没有，IoU未定义，设为0（但会在平均时考虑）
                    iou = 0.0
            else:
                iou = (intersection / union).item()
            
            ious.append(iou)
        
        # 计算平均IoU（所有类别都参与，包括0值）
        mean_iou = sum(ious) / len(ious)
        
        # 转换为tensor
        ious_tensor = torch.tensor(ious, dtype=torch.float32).to(pred.device)
        
        return mean_iou, ious_tensor
    
    def save_checkpoint(self, epoch, val_iou, is_best=False):
        """保存检查点"""
        checkpoint_dir = Path('checkpoints/segformer_enhanced')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_iou': val_iou,
            'best_val_iou': self.best_val_iou,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_ious': self.val_ious
        }
        
        # 保存最新检查点
        checkpoint_path = checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = checkpoint_dir / 'best_enhanced_model.pth'
            torch.save(checkpoint, best_path)
            print(f"✅ 保存最佳模型: {best_path}")
    
    def train(self):
        """训练主循环"""
        num_epochs = self.config['training']['num_epochs']
        
        print("\n" + "=" * 60)
        print("🎯 开始训练增强模块")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            # 根据epoch更新损失函数（支持渐进式权重）
            self.criterion = self.create_criterion(epoch)
            
            # 重置验证诊断标志（仅在epoch 0开始时初始化）
            if epoch == 0 and not hasattr(self, '_val_first_batch_printed'):
                self._val_first_batch_printed = False
            
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_iou, class_ious = self.validate(epoch)
            
            # 渐进式解冻策略检查（提前解冻decoder - 方案1，仅冻结模式启用）
            if self.freeze_backbone:
                optimization_config = self.config.get('optimization', {})
                unfreeze_epoch = optimization_config.get('unfreeze_decoder_epoch', 12)
                unfreeze_condition = optimization_config.get('unfreeze_decoder_condition', 'always')
                unfreeze_lr_ratio = optimization_config.get('unfreeze_decoder_lr_ratio', 0.4)
                
                should_unfreeze = False
                if unfreeze_condition == 'always':
                    should_unfreeze = (epoch == unfreeze_epoch)
                elif unfreeze_condition == 'if_green_low':
                    should_unfreeze = (epoch == unfreeze_epoch and class_ious[2] < 0.20)
                
                if should_unfreeze and not hasattr(self, '_decoder_unfrozen'):
                    # 解冻decoder层（使用更小的学习率）
                    print("\n" + "=" * 60)
                    print(f"🔓 Epoch {unfreeze_epoch}: 提前解冻decoder层（方案1）")
                    print("=" * 60)
                    
                    # 检查是否有original_model
                    if hasattr(self.model, 'original_model'):
                        # 解冻decoder
                        decoder_unfrozen = 0
                        for name, param in self.model.original_model.decoder.named_parameters():
                            param.requires_grad = True
                            decoder_unfrozen += param.numel()
                        
                        print(f"✅ 解冻decoder参数: {decoder_unfrozen:,}")
                        
                        # 获取当前学习率
                        current_lr = self.optimizer.param_groups[0]['lr']
                        # 使用更小的学习率（decoder使用当前学习率的指定比例）
                        decoder_lr = max(current_lr * unfreeze_lr_ratio, 0.0000005)  # 至少0.0000005
                        print(f"   增强模块学习率: {current_lr:.8f}")
                        print(f"   Decoder学习率: {decoder_lr:.8f} (当前学习率的{unfreeze_lr_ratio*100:.0f}%)")
                        
                        # 分离decoder和其他参数
                        decoder_params = []
                        other_params = []
                        for name, param in self.model.named_parameters():
                            if param.requires_grad:
                                if 'decoder' in name:
                                    decoder_params.append(param)
                                else:
                                    other_params.append(param)
                        
                        # 使用不同的学习率
                        if len(decoder_params) > 0 and len(other_params) > 0:
                            param_groups = [
                                {'params': other_params, 'lr': current_lr},
                                {'params': decoder_params, 'lr': decoder_lr}
                            ]
                            self.optimizer = optim.AdamW(
                                param_groups,
                                weight_decay=self.config['training'].get('weight_decay', 0.0001)
                            )
                            print(f"📊 优化器更新:")
                            print(f"  增强模块学习率: {current_lr:.8f}")
                            print(f"  Decoder学习率: {decoder_lr:.8f}")
                            print(f"✅ 已重新创建优化器，使用分组学习率")
                            self._decoder_unfrozen = True
                        else:
                            # 如果分离失败，使用统一学习率
                            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                            self.optimizer = optim.AdamW(
                                trainable_params,
                                lr=decoder_lr,
                                weight_decay=self.config['training'].get('weight_decay', 0.0001)
                            )
                            print(f"📊 优化器更新: 统一学习率 {decoder_lr:.6f}")
                        
                        # 重新创建scheduler（使用新的优化器）
                        self.scheduler = self.create_scheduler()
            
            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step(val_iou)
            
            # 获取当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 打印结果
            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  验证损失: {val_loss:.4f}")
            print(f"  验证IoU: {val_iou:.4f}")
            print(f"  学习率: {current_lr:.6f}")
            print(f"  类别IoU: 背景={class_ious[0]:.4f}, 红={class_ious[1]:.4f}, "
                  f"绿={class_ious[2]:.4f}, 蓝={class_ious[3]:.4f}, 黄={class_ious[4]:.4f}")
            
            # 每个epoch打印logits值（关键诊断信息）
            with torch.no_grad():
                self.model.eval()
                sample_images, sample_masks = next(iter(self.val_loader))
                sample_images = sample_images[:1].to(self.device)
                sample_outputs = self.model(sample_images)
                sample_logits, _ = self._split_model_outputs(sample_outputs)
                logits_avg = [sample_logits[:, cls, :, :].mean().item() for cls in range(5)]
                print(f"  📈 验证集logits均值 [背景,红,绿,蓝,黄] = "
                      f"[{logits_avg[0]:.3f}, {logits_avg[1]:.3f}, {logits_avg[2]:.3f}, "
                      f"{logits_avg[3]:.3f}, {logits_avg[4]:.3f}]")
                
                # 计算绿色类别的预测概率
                probs = torch.softmax(sample_logits, dim=1)
                green_prob = probs[:, 2, :, :].mean().item() * 100
                print(f"  🎯 绿色类别平均预测概率: {green_prob:.2f}%")
                
                # 如果绿色logits仍然很低，给出警告
                if logits_avg[2] < -2.0:
                    print(f"  ⚠️  警告: 绿色logits值仍然很低({logits_avg[2]:.3f})，像素级加权可能未生效！")
                elif logits_avg[2] < -1.0:
                    print(f"  📊 绿色logits值: {logits_avg[2]:.3f} (正在改善中...)")
                else:
                    print(f"  ✅ 绿色logits值: {logits_avg[2]:.3f} (已进入正常范围)")
                
                self.model.train()
            
            # 红色类别专项监控
            if class_ious[1] > 0.35:
                print(f"  ✅ 红色类别IoU突破0.35: {class_ious[1]:.4f}")
            elif class_ious[1] > 0.32:
                print(f"  📈 红色类别IoU改善: {class_ious[1]:.4f}")
            else:
                print(f"  ⚠️  红色类别IoU仍偏低: {class_ious[1]:.4f}，继续优化中...")
            
            # 绿色类别专项监控（关键瓶颈）
            if class_ious[2] > 0.25:
                print(f"  ✅ 绿色类别IoU突破0.25: {class_ious[2]:.4f}")
            elif class_ious[2] > 0.20:
                print(f"  📈 绿色类别IoU改善: {class_ious[2]:.4f}")
            elif class_ious[2] > 0.18:
                print(f"  ⚠️  绿色类别IoU仍偏低: {class_ious[2]:.4f}，继续优化中...")
            else:
                print(f"  🚨 绿色类别IoU严重偏低: {class_ious[2]:.4f}，需要重点优化！")
            
            # 保存检查点
            is_best = val_iou > self.best_val_iou
            if is_best:
                self.best_val_iou = val_iou
                print(f"  🎉 新的最佳IoU: {val_iou:.4f}")
            
            self.save_checkpoint(epoch, val_iou, is_best)
            
            print("-" * 60)
        
        print(f"\n✅ 训练完成！最佳验证IoU: {self.best_val_iou:.4f}")
        print(f"最佳模型保存位置: checkpoints/segformer_enhanced/best_enhanced_model.pth")


def main():
    parser = argparse.ArgumentParser(
        description='训练增强模块',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置训练（自动查找检查点）
  python train_enhancement_modules.py
  
  # 使用增强模块训练配置
  python train_enhancement_modules.py \\
      --config config/config_enhancement_training.yaml
  
  # 手动指定检查点路径
  python train_enhancement_modules.py \\
      --checkpoint checkpoints/segformer1/best_model.pth
        """
    )
    parser.add_argument('--config', type=str, default='config/config_enhancement_training.yaml',
                       help='配置文件路径 (default: config/config_enhancement_training.yaml)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='预训练SegFormer模型检查点路径（可选，默认自动查找）')
    
    args = parser.parse_args()
    
    # 自动查找检查点文件（如果未指定）
    if args.checkpoint is None:
        # 按优先级尝试多个常见路径
        checkpoint_paths = [
            'checkpoints/segformer1/best_model.pth',
            'checkpoints/segformer/best_model.pth',
            'checkpoints/best_model.pth',
            'checkpoints/segformer1/checkpoint_epoch_50.pth',
            'checkpoints/segformer1/checkpoint_epoch_40.pth',
            'checkpoints/segformer1/checkpoint_epoch_30.pth',
        ]
        
        checkpoint_found = None
        for path in checkpoint_paths:
            if os.path.exists(path):
                checkpoint_found = path
                print(f"✅ 自动找到检查点文件: {checkpoint_found}")
                break
        
        if checkpoint_found is None:
            print("❌ 未找到检查点文件！已尝试查找以下路径：")
            for path in checkpoint_paths:
                print(f"   - {path}")
            raise FileNotFoundError(
                f"请手动指定 --checkpoint 参数，或确保检查点文件存在于上述路径之一。"
            )
        
        args.checkpoint = checkpoint_found
    
    # 创建训练器
    trainer = EnhancementTrainer(args.config, args.checkpoint)
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()


"""
SegFormer增强模块
- 边缘增强模块(EdgeEnhancementModule)
- 颜色注意力模块(ColorAttentionModule)
专门针对绿色和黄色混淆问题优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EdgeEnhancementModule(nn.Module):
    """
    边缘增强模块
    使用多尺度边缘检测和特征融合来增强边缘细节
    """
    
    def __init__(self, num_classes=5, channels=256, dilations=(1, 2, 4)):
        super().__init__()
        self.dilations = tuple(int(dilation) for dilation in dilations)
        if len(self.dilations) != 3 or any(dilation < 1 for dilation in self.dilations):
            raise ValueError("EEM dilations must contain exactly three positive integers")
        self.num_classes = num_classes
        
        # 多尺度边缘检测卷积核
        # Sobel算子用于边缘检测
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        # 注册为buffer，不参与训练
        self.register_buffer('sobel_x', sobel_x.repeat(num_classes, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.repeat(num_classes, 1, 1, 1))
        
        # 边缘特征提取
        self.edge_conv = nn.Sequential(
            nn.Conv2d(num_classes, channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels // 2, channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # 边缘特征融合
        self.edge_fusion = nn.Sequential(
            nn.Conv2d(num_classes + channels // 4, num_classes, kernel_size=1),
            nn.BatchNorm2d(num_classes),
            nn.Sigmoid()
        )
        
        # 多尺度边缘增强
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv2d(
                num_classes, num_classes, kernel_size=3,
                padding=dilation, dilation=dilation,
            )
            for dilation in self.dilations
        ])
        
        self.scale_fusion = nn.Conv2d(num_classes * 3, num_classes, kernel_size=1)
    
    def forward(self, logits):
        """
        Args:
            logits: [B, num_classes, H, W] 模型输出的logits
        Returns:
            enhanced_logits: [B, num_classes, H, W] 边缘增强后的logits
        """
        B, C, H, W = logits.shape
        
        # 1. 计算边缘图
        # 对每个类别的概率图进行边缘检测
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]
        
        # 使用Sobel算子计算边缘
        edge_maps = []
        for i in range(C):
            prob_map = probs[:, i:i+1, :, :]  # [B, 1, H, W]
            edge_x = F.conv2d(prob_map, self.sobel_x[i:i+1], padding=1)
            edge_y = F.conv2d(prob_map, self.sobel_y[i:i+1], padding=1)
            edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
            edge_maps.append(edge_magnitude)
        
        edge_map = torch.cat(edge_maps, dim=1)  # [B, C, H, W]
        
        # 2. 提取边缘特征
        edge_features = self.edge_conv(edge_map)  # [B, channels//4, H, W]
        
        # 3. 融合原始logits和边缘特征
        fused = torch.cat([logits, edge_features], dim=1)  # [B, C + channels//4, H, W]
        edge_attention = self.edge_fusion(fused)  # [B, C, H, W]
        
        # 4. 应用边缘注意力
        edge_enhanced = logits * (1 + edge_attention)  # 增强边缘区域的响应
        
        # 5. 多尺度边缘增强
        multi_scale_features = []
        for conv in self.multi_scale_conv:
            multi_scale_features.append(conv(edge_enhanced))
        
        multi_scale_fused = torch.cat(multi_scale_features, dim=1)  # [B, C*3, H, W]
        final_enhanced = edge_enhanced + self.scale_fusion(multi_scale_fused)  # 残差连接
        
        return final_enhanced


class ColorAttentionModule(nn.Module):
    """
    颜色注意力模块
    专门用于区分绿色(类别2)和黄色(类别4)，以及红色(类别1)和蓝色(类别3)
    """
    
    def __init__(self, num_classes=5, embedding_dim=64, strength=1.0):
        super().__init__()
        self.strength = float(strength)
        if not 0.0 < self.strength <= 2.0:
            raise ValueError("CAM strength must be in (0, 2]")
        self.num_classes = num_classes
        
        # 颜色特征提取(RGB空间)
        self.color_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, embedding_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True)
        )
        
        # 类别特定的颜色注意力(针对容易混淆的类别对)
        # 绿色 vs 黄色注意力
        self.green_yellow_attention = nn.Sequential(
            nn.Conv2d(embedding_dim + num_classes, 64, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 2, kernel_size=1),  # 2个输出：绿色权重和黄色权重
            nn.Softmax(dim=1)
        )
        
        # 红色 vs 蓝色注意力
        self.red_blue_attention = nn.Sequential(
            nn.Conv2d(embedding_dim + num_classes, 64, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 2, kernel_size=1),  # 2个输出：红色权重和蓝色权重
            nn.Softmax(dim=1)
        )
        
        # 最终融合
        self.final_fusion = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=False),
            nn.Conv2d(num_classes, num_classes, kernel_size=1)
        )
    
    def forward(self, logits, rgb_image):
        """
        Args:
            logits: [B, num_classes, H, W] 模型输出的logits
            rgb_image: [B, 3, H, W] RGB图像(归一化后)
        Returns:
            refined_logits: [B, num_classes, H, W] 颜色注意力细化后的logits
        """
        B, C, H, W = logits.shape
        
        # 1. 提取颜色特征
        # 将归一化图像转换为[0, 1]范围用于颜色特征提取
        # 假设使用ImageNet归一化：mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        # 反归一化：(x * std) + mean
        mean = torch.tensor([0.485, 0.456, 0.406], device=rgb_image.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=rgb_image.device).view(1, 3, 1, 1)
        rgb_normalized = (rgb_image * std + mean).clamp(0, 1)  # [B, 3, H, W], [0, 1]范围
        
        color_features = self.color_encoder(rgb_normalized)  # [B, embedding_dim, H, W]
        
        # 2. 获取当前预测的概率分布
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]
        
        # 3. 融合颜色特征和预测概率
        color_prob_fused = torch.cat([color_features, probs], dim=1)  # [B, embedding_dim + C, H, W]
        
        # 4. 绿色 vs 黄色注意力
        green_yellow_weights = self.green_yellow_attention(color_prob_fused)  # [B, 2, H, W]
        green_weight = green_yellow_weights[:, 0:1, :, :]  # [B, 1, H, W]
        yellow_weight = green_yellow_weights[:, 1:2, :, :]  # [B, 1, H, W]
        
        # 5. 红色 vs 蓝色注意力
        red_blue_weights = self.red_blue_attention(color_prob_fused)  # [B, 2, H, W]
        red_weight = red_blue_weights[:, 0:1, :, :]  # [B, 1, H, W]
        blue_weight = red_blue_weights[:, 1:2, :, :]  # [B, 1, H, W]
        
        # 6. 应用颜色注意力到对应的类别logits(使用非原地操作避免梯度问题)
        # 分离各个类别
        logits_bg = logits[:, 0:1, :, :]  # 背景
        logits_red = logits[:, 1:2, :, :]  # 红色
        logits_green = logits[:, 2:3, :, :]  # 绿色
        logits_blue = logits[:, 3:4, :, :]  # 蓝色
        logits_yellow = logits[:, 4:5, :, :]  # 黄色
        
        # 应用颜色注意力（非原地操作，添加保护机制防止过度抑制蓝色类别）
        # 使用更温和的抑制系数，并确保不会让负数logits变得更负
        
        # 绿色类别：增强绿色，轻度抑制黄色
        logits_green_enhanced = logits_green * (1 + green_weight * 0.3 * self.strength) * (1 - yellow_weight * 0.1 * self.strength)
        
        # 黄色类别：增强黄色，轻度抑制绿色
        logits_yellow_enhanced = logits_yellow * (1 - green_weight * 0.1 * self.strength) * (1 + yellow_weight * 0.3 * self.strength)
        
        # 红色类别：增强红色，轻度抑制蓝色（但不会过度抑制）
        blue_suppression = blue_weight * 0.05 * self.strength  # 大幅降低抑制强度（从0.2降到0.05）
        logits_red_enhanced = logits_red * (1 + red_weight * 0.3 * self.strength) * (1 - blue_suppression)
        
        # 蓝色类别：增强蓝色，轻度抑制红色（关键修复）
        # 使用更安全的增强方式，避免负数logits变得更负
        red_suppression = red_weight * 0.05 * self.strength  # 大幅降低抑制强度（从0.2降到0.05）
        blue_enhancement = blue_weight * 0.2 * self.strength  # 使用较小的增强系数
        
        # 对于蓝色类别，使用更安全的增强方式
        # 如果蓝色logits已经是负数，使用加法增强；否则使用乘法
        blue_positive_mask = (logits_blue > 0).float()
        blue_negative_mask = 1.0 - blue_positive_mask
        
        # 正数部分：使用乘法增强
        blue_positive = logits_blue * blue_positive_mask
        blue_positive_enhanced = blue_positive * (1 - red_suppression) * (1 + blue_enhancement)
        
        # 负数部分：使用加法增强（避免变得更负）
        blue_negative = logits_blue * blue_negative_mask
        blue_negative_enhanced = blue_negative + blue_enhancement * 0.5 * blue_negative_mask
        
        # 合并正负部分
        logits_blue_enhanced = blue_positive_enhanced + blue_negative_enhanced
        
        # 重新组合（使用torch.cat，非原地操作）
        refined_logits = torch.cat([
            logits_bg,  # 背景保持不变
            logits_red_enhanced,
            logits_green_enhanced,
            logits_blue_enhanced,
            logits_yellow_enhanced
        ], dim=1)
        
        # 7. 最终融合（使用残差连接，防止过度修改）
        fusion_output = self.final_fusion(refined_logits)
        # 使用残差连接：保留80%的原始logits，只使用20%的融合输出（更保守）
        fusion_alpha = 0.2  # 从0.3降低到0.2，更保守
        final_refined = refined_logits * (1 - fusion_alpha) + fusion_output * fusion_alpha
        
        return final_refined


class EnhancedSegFormer(nn.Module):
    """
    增强的SegFormer模型
    在原始SegFormer基础上添加边缘增强和颜色注意力模块
    """
    
    def __init__(self, original_model, use_edge_enhancement=True, use_color_attention=True,
                 use_green_aux_head=False):
        super().__init__()
        self.original_model = original_model
        self.use_edge_enhancement = use_edge_enhancement
        self.use_color_attention = use_color_attention
        self.use_green_aux_head = use_green_aux_head
        
        num_classes = original_model.decoder.num_classes
        
        # 边缘增强模块
        if use_edge_enhancement:
            self.edge_enhance = EdgeEnhancementModule(num_classes=num_classes)
        
        # 颜色注意力模块
        if use_color_attention:
            self.color_attention = ColorAttentionModule(num_classes=num_classes)
        
        if use_green_aux_head:
            hidden_c = max(4, num_classes // 2)
            self.green_aux_head = nn.Sequential(
                nn.Conv2d(num_classes, hidden_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_c, 1, kernel_size=1)
            )
        else:
            self.green_aux_head = None
    
    def forward(self, x, return_intermediates=False):
        """
        Args:
            x: [B, 3, H, W] 输入图像
            return_intermediates: 是否返回中间结果(用于调试)
        Returns:
            output: [B, num_classes, H, W] 增强后的logits
        """
        # 1. 原始SegFormer预测
        original_out = self.original_model(x)  # [B, num_classes, H, W]
        
        # 2. 边缘增强
        if self.use_edge_enhancement:
            edge_enhanced = self.edge_enhance(original_out)
        else:
            edge_enhanced = original_out
        
        # 3. 颜色注意力细化（需要原始图像）
        if self.use_color_attention:
            # 确保图像在[0, 1]范围内(如果已经归一化，需要反归一化)
            # 这里假设输入已经归一化，我们需要使用原始图像
            # 在实际使用时，可能需要传入原始图像
            color_refined = self.color_attention(edge_enhanced, x)
        else:
            color_refined = edge_enhanced
        
        green_aux_logits = None
        if self.use_green_aux_head and self.green_aux_head is not None:
            green_aux_logits = self.green_aux_head(color_refined)
        
        if return_intermediates:
            return {
                'original': original_out,
                'edge_enhanced': edge_enhanced,
                'final': color_refined,
                'green_aux': green_aux_logits
            }
        
        if self.use_green_aux_head:
            return color_refined, green_aux_logits
        return color_refined


def create_enhanced_segformer(original_model, use_edge_enhancement=True,
                              use_color_attention=True, use_green_aux_head=False):
    """
    创建增强的SegFormer模型
    
    Args:
        original_model: 原始SegFormer模型
        use_edge_enhancement: 是否使用边缘增强
        use_color_attention: 是否使用颜色注意力
    Returns:
        EnhancedSegFormer模型
    """
    return EnhancedSegFormer(
        original_model=original_model,
        use_edge_enhancement=use_edge_enhancement,
        use_color_attention=use_color_attention,
        use_green_aux_head=use_green_aux_head
    )


class EdgeEnhancementModule(nn.Module):
    """
    边缘增强模块
    使用多尺度边缘检测和特征融合来增强边缘细节
    """
    
    def __init__(self, num_classes=5, channels=256, dilations=(1, 2, 4)):
        super().__init__()
        self.dilations = tuple(int(dilation) for dilation in dilations)
        if len(self.dilations) != 3 or any(dilation < 1 for dilation in self.dilations):
            raise ValueError("EEM dilations must contain exactly three positive integers")
        self.num_classes = num_classes
        
        # 多尺度边缘检测卷积核
        # Sobel算子用于边缘检测
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        # 注册为buffer，不参与训练
        self.register_buffer('sobel_x', sobel_x.repeat(num_classes, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.repeat(num_classes, 1, 1, 1))
        
        # 边缘特征提取
        self.edge_conv = nn.Sequential(
            nn.Conv2d(num_classes, channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels // 2, channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # 边缘特征融合
        self.edge_fusion = nn.Sequential(
            nn.Conv2d(num_classes + channels // 4, num_classes, kernel_size=1),
            nn.BatchNorm2d(num_classes),
            nn.Sigmoid()
        )
        
        # 多尺度边缘增强
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv2d(
                num_classes, num_classes, kernel_size=3,
                padding=dilation, dilation=dilation,
            )
            for dilation in self.dilations
        ])
        
        self.scale_fusion = nn.Conv2d(num_classes * 3, num_classes, kernel_size=1)
    
    def forward(self, logits):
        """
        Args:
            logits: [B, num_classes, H, W] 模型输出的logits
        Returns:
            enhanced_logits: [B, num_classes, H, W] 边缘增强后的logits
        """
        B, C, H, W = logits.shape
        
        # 1. 计算边缘图
        # 对每个类别的概率图进行边缘检测
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]
        
        # 使用Sobel算子计算边缘
        edge_maps = []
        for i in range(C):
            prob_map = probs[:, i:i+1, :, :]  # [B, 1, H, W]
            edge_x = F.conv2d(prob_map, self.sobel_x[i:i+1], padding=1)
            edge_y = F.conv2d(prob_map, self.sobel_y[i:i+1], padding=1)
            edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
            edge_maps.append(edge_magnitude)
        
        edge_map = torch.cat(edge_maps, dim=1)  # [B, C, H, W]
        
        # 2. 提取边缘特征
        edge_features = self.edge_conv(edge_map)  # [B, channels//4, H, W]
        
        # 3. 融合原始logits和边缘特征
        fused = torch.cat([logits, edge_features], dim=1)  # [B, C + channels//4, H, W]
        edge_attention = self.edge_fusion(fused)  # [B, C, H, W]
        
        # 4. 应用边缘注意力
        edge_enhanced = logits * (1 + edge_attention)  # 增强边缘区域的响应
        
        # 5. 多尺度边缘增强
        multi_scale_features = []
        for conv in self.multi_scale_conv:
            multi_scale_features.append(conv(edge_enhanced))
        
        multi_scale_fused = torch.cat(multi_scale_features, dim=1)  # [B, C*3, H, W]
        final_enhanced = edge_enhanced + self.scale_fusion(multi_scale_fused)  # 残差连接
        
        return final_enhanced


class ColorAttentionModule(nn.Module):
    """
    颜色注意力模块
    专门用于区分绿色(类别2)和黄色(类别4)，以及红色(类别1)和蓝色(类别3)
    """
    
    def __init__(self, num_classes=5, embedding_dim=64, strength=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.strength = float(strength)
        if not 0.0 < self.strength <= 2.0:
            raise ValueError("CAM strength must be in (0, 2]")
        
        # 颜色特征提取(RGB空间)
        self.color_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, embedding_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True)
        )
        
        # 类别特定的颜色注意力(针对容易混淆的类别对)
        # 绿色 vs 黄色注意力
        self.green_yellow_attention = nn.Sequential(
            nn.Conv2d(embedding_dim + num_classes, 64, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 2, kernel_size=1),  # 2个输出：绿色权重和黄色权重
            nn.Softmax(dim=1)
        )
        
        # 红色 vs 蓝色注意力
        self.red_blue_attention = nn.Sequential(
            nn.Conv2d(embedding_dim + num_classes, 64, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 2, kernel_size=1),  # 2个输出：红色权重和蓝色权重
            nn.Softmax(dim=1)
        )
        
        # 最终融合
        self.final_fusion = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=False),
            nn.Conv2d(num_classes, num_classes, kernel_size=1)
        )
    
    def forward(self, logits, rgb_image):
        """
        Args:
            logits: [B, num_classes, H, W] 模型输出的logits
            rgb_image: [B, 3, H, W] RGB图像(归一化后)
        Returns:
            refined_logits: [B, num_classes, H, W] 颜色注意力细化后的logits
        """
        B, C, H, W = logits.shape
        
        # 1. 提取颜色特征
        # 将归一化图像转换为[0, 1]范围用于颜色特征提取
        # 假设使用ImageNet归一化：mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        # 反归一化：(x * std) + mean
        mean = torch.tensor([0.485, 0.456, 0.406], device=rgb_image.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=rgb_image.device).view(1, 3, 1, 1)
        rgb_normalized = (rgb_image * std + mean).clamp(0, 1)  # [B, 3, H, W], [0, 1]范围
        
        color_features = self.color_encoder(rgb_normalized)  # [B, embedding_dim, H, W]
        
        # 2. 获取当前预测的概率分布
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]
        
        # 3. 融合颜色特征和预测概率
        color_prob_fused = torch.cat([color_features, probs], dim=1)  # [B, embedding_dim + C, H, W]
        
        # 4. 绿色 vs 黄色注意力
        green_yellow_weights = self.green_yellow_attention(color_prob_fused)  # [B, 2, H, W]
        green_weight = green_yellow_weights[:, 0:1, :, :]  # [B, 1, H, W]
        yellow_weight = green_yellow_weights[:, 1:2, :, :]  # [B, 1, H, W]
        
        # 5. 红色 vs 蓝色注意力
        red_blue_weights = self.red_blue_attention(color_prob_fused)  # [B, 2, H, W]
        red_weight = red_blue_weights[:, 0:1, :, :]  # [B, 1, H, W]
        blue_weight = red_blue_weights[:, 1:2, :, :]  # [B, 1, H, W]
        
        # 6. 应用颜色注意力到对应的类别logits(使用非原地操作避免梯度问题)
        # 分离各个类别
        logits_bg = logits[:, 0:1, :, :]  # 背景
        logits_red = logits[:, 1:2, :, :]  # 红色
        logits_green = logits[:, 2:3, :, :]  # 绿色
        logits_blue = logits[:, 3:4, :, :]  # 蓝色
        logits_yellow = logits[:, 4:5, :, :]  # 黄色
        
        # 应用颜色注意力（非原地操作，添加保护机制防止过度抑制蓝色类别）
        # 使用更温和的抑制系数，并确保不会让负数logits变得更负
        
        # 绿色类别：增强绿色，轻度抑制黄色
        logits_green_enhanced = logits_green * (1 + green_weight * 0.3 * self.strength) * (1 - yellow_weight * 0.1 * self.strength)
        
        # 黄色类别：增强黄色，轻度抑制绿色
        logits_yellow_enhanced = logits_yellow * (1 - green_weight * 0.1 * self.strength) * (1 + yellow_weight * 0.3 * self.strength)
        
        # 红色类别：增强红色，轻度抑制蓝色（但不会过度抑制）
        blue_suppression = blue_weight * 0.05 * self.strength  # 大幅降低抑制强度（从0.2降到0.05）
        logits_red_enhanced = logits_red * (1 + red_weight * 0.3 * self.strength) * (1 - blue_suppression)
        
        # 蓝色类别：增强蓝色，轻度抑制红色（关键修复）
        # 使用更安全的增强方式，避免负数logits变得更负
        red_suppression = red_weight * 0.05 * self.strength  # 大幅降低抑制强度（从0.2降到0.05）
        blue_enhancement = blue_weight * 0.2 * self.strength  # 使用较小的增强系数
        
        # 对于蓝色类别，使用更安全的增强方式
        # 如果蓝色logits已经是负数，使用加法增强；否则使用乘法
        blue_positive_mask = (logits_blue > 0).float()
        blue_negative_mask = 1.0 - blue_positive_mask
        
        # 正数部分：使用乘法增强
        blue_positive = logits_blue * blue_positive_mask
        blue_positive_enhanced = blue_positive * (1 - red_suppression) * (1 + blue_enhancement)
        
        # 负数部分：使用加法增强（避免变得更负）
        blue_negative = logits_blue * blue_negative_mask
        blue_negative_enhanced = blue_negative + blue_enhancement * 0.5 * blue_negative_mask
        
        # 合并正负部分
        logits_blue_enhanced = blue_positive_enhanced + blue_negative_enhanced
        
        # 重新组合（使用torch.cat，非原地操作）
        refined_logits = torch.cat([
            logits_bg,  # 背景保持不变
            logits_red_enhanced,
            logits_green_enhanced,
            logits_blue_enhanced,
            logits_yellow_enhanced
        ], dim=1)
        
        # 7. 最终融合（使用残差连接，防止过度修改）
        fusion_output = self.final_fusion(refined_logits)
        # 使用残差连接：保留80%的原始logits，只使用20%的融合输出（更保守）
        fusion_alpha = 0.2  # 从0.3降低到0.2，更保守
        final_refined = refined_logits * (1 - fusion_alpha) + fusion_output * fusion_alpha
        
        return final_refined


class EnhancedSegFormer(nn.Module):
    """
    增强的SegFormer模型
    在原始SegFormer基础上添加边缘增强和颜色注意力模块
    """
    
    def __init__(self, original_model, use_edge_enhancement=True,
                 use_color_attention=True, use_green_aux_head=False,
                 edge_enhancement_version="v1", edge_dilations=(1, 2, 4),
                 cam_strength=1.0):
        super().__init__()
        self.original_model = original_model
        self.use_edge_enhancement = use_edge_enhancement
        self.use_color_attention = use_color_attention
        self.use_green_aux_head = use_green_aux_head
        self.edge_enhancement_version = edge_enhancement_version
        
        num_classes = original_model.decoder.num_classes
        
        # 边缘增强模块
        if use_edge_enhancement:
            if edge_enhancement_version == "v1":
                self.edge_enhance = EdgeEnhancementModule(
                    num_classes=num_classes, dilations=edge_dilations
                )
            elif edge_enhancement_version == "v2":
                self.edge_enhance = EfficientEdgeEnhancementModule(
                    num_classes=num_classes
                )
            elif edge_enhancement_version == "v2.1":
                self.edge_enhance = DirectSupervisedEdgeEnhancementModule(
                    num_classes=num_classes
                )
            else:
                raise ValueError(
                    "Unsupported edge_enhancement_version: "
                    f"{edge_enhancement_version}"
                )
        
        # 颜色注意力模块
        if use_color_attention:
            self.color_attention = ColorAttentionModule(
                num_classes=num_classes, strength=cam_strength
            )
        
        if use_green_aux_head:
            hidden_c = max(4, num_classes // 2)
            self.green_aux_head = nn.Sequential(
                nn.Conv2d(num_classes, hidden_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_c, 1, kernel_size=1)
            )
        else:
            self.green_aux_head = None
    
    def forward(self, x, return_intermediates=False):
        """
        Args:
            x: [B, 3, H, W] 输入图像
            return_intermediates: 是否返回中间结果(用于调试)
        Returns:
            output: [B, num_classes, H, W] 增强后的logits
        """
        # 1. 原始SegFormer预测
        original_out = self.original_model(x)  # [B, num_classes, H, W]
        
        # 2. 边缘增强
        edge_aux_logits = None
        if self.use_edge_enhancement:
            if self.edge_enhancement_version in {"v2", "v2.1"}:
                edge_enhanced, edge_aux_logits = self.edge_enhance(
                    original_out, return_boundary=True
                )
            else:
                edge_enhanced = self.edge_enhance(original_out)
        else:
            edge_enhanced = original_out
        
        # 3. 颜色注意力细化（需要原始图像）
        if self.use_color_attention:
            # 确保图像在[0, 1]范围内(如果已经归一化，需要反归一化)
            # 这里假设输入已经归一化，我们需要使用原始图像
            # 在实际使用时，可能需要传入原始图像
            color_refined = self.color_attention(edge_enhanced, x)
        else:
            color_refined = edge_enhanced
        
        green_aux_logits = None
        if self.use_green_aux_head and self.green_aux_head is not None:
            green_aux_logits = self.green_aux_head(color_refined)
        
        if return_intermediates:
            return {
                'original': original_out,
                'edge_enhanced': edge_enhanced,
                'final': color_refined,
                'green_aux': green_aux_logits,
                'edge_aux': edge_aux_logits
            }
        
        if self.use_green_aux_head:
            if edge_aux_logits is not None:
                return color_refined, green_aux_logits, edge_aux_logits
            return color_refined, green_aux_logits
        return color_refined


def create_enhanced_segformer(original_model, use_edge_enhancement=True,
                              use_color_attention=True, use_green_aux_head=False,
                              edge_enhancement_version="v1", edge_dilations=(1, 2, 4),
                              cam_strength=1.0):
    """
    创建增强的SegFormer模型
    
    Args:
        original_model: 原始SegFormer模型
        use_edge_enhancement: 是否使用边缘增强
        use_color_attention: 是否使用颜色注意力
    Returns:
        EnhancedSegFormer模型
    """
    return EnhancedSegFormer(
        original_model=original_model,
        use_edge_enhancement=use_edge_enhancement,
        use_color_attention=use_color_attention,
        use_green_aux_head=use_green_aux_head,
        edge_enhancement_version=edge_enhancement_version,
        edge_dilations=edge_dilations,
        cam_strength=cam_strength,
    )


class EfficientEdgeEnhancementModule(nn.Module):
    """Lightweight EEM with explicit, boundary-localized refinement.

    This preserves the paper's Sobel and multi-scale design while replacing
    global multiplicative logit changes with a bounded additive residual that
    is gated by a supervised boundary prediction.
    """

    def __init__(
        self,
        num_classes=5,
        hidden_channels=16,
        max_residual_scale=0.5,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.max_residual_scale = float(max_residual_scale)

        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer(
            "sobel_x", sobel_x.repeat(self.num_classes, 1, 1, 1)
        )
        self.register_buffer(
            "sobel_y", sobel_y.repeat(self.num_classes, 1, 1, 1)
        )

        boundary_inputs = self.num_classes + 1
        norm_groups = 4 if hidden_channels % 4 == 0 else 1
        self.boundary_encoder = nn.Sequential(
            nn.Conv2d(boundary_inputs, hidden_channels, kernel_size=1),
            nn.GroupNorm(norm_groups, hidden_channels),
            nn.SiLU(inplace=False),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
            ),
            nn.GroupNorm(norm_groups, hidden_channels),
            nn.SiLU(inplace=False),
        )
        self.boundary_head = nn.Conv2d(
            hidden_channels, 1, kernel_size=1
        )
        nn.init.zeros_(self.boundary_head.weight)
        nn.init.constant_(self.boundary_head.bias, -1.5)

        self.context_branches = nn.ModuleList(
            [
                nn.Conv2d(
                    self.num_classes,
                    self.num_classes,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    groups=self.num_classes,
                    bias=False,
                )
                for dilation in (1, 2, 4)
            ]
        )
        self.residual_fusion = nn.Conv2d(
            self.num_classes * 3, self.num_classes, kernel_size=1
        )
        nn.init.zeros_(self.residual_fusion.weight)
        nn.init.zeros_(self.residual_fusion.bias)

        # sigmoid(-1.386) ~= 0.2. With max_residual_scale=0.5, the
        # learnable upper scale starts at 0.1; the zero-initialized residual
        # still makes the complete module an exact identity initially.
        self.residual_scale_logit = nn.Parameter(torch.tensor(-1.3862944))

    def forward(self, logits, return_boundary=False):
        probs = F.softmax(logits, dim=1)
        edge_x = F.conv2d(
            probs,
            self.sobel_x,
            padding=1,
            groups=self.num_classes,
        )
        edge_y = F.conv2d(
            probs,
            self.sobel_y,
            padding=1,
            groups=self.num_classes,
        )
        edge_magnitude = torch.sqrt(
            edge_x.square() + edge_y.square() + 1e-6
        )
        edge_signal = torch.tanh(edge_magnitude)

        entropy = -(
            probs * torch.log(probs.clamp_min(1e-6))
        ).sum(dim=1, keepdim=True)
        entropy = entropy / np.log(float(self.num_classes))

        boundary_features = self.boundary_encoder(
            torch.cat([edge_signal, entropy], dim=1)
        )
        boundary_logits = self.boundary_head(boundary_features)
        boundary_gate = torch.sigmoid(boundary_logits)

        multi_scale = torch.cat(
            [branch(logits) for branch in self.context_branches], dim=1
        )
        bounded_residual = torch.tanh(self.residual_fusion(multi_scale))
        residual_scale = self.max_residual_scale * torch.sigmoid(
            self.residual_scale_logit
        )
        enhanced = logits + residual_scale * boundary_gate * bounded_residual

        if return_boundary:
            return enhanced, boundary_logits
        return enhanced


class DirectSupervisedEdgeEnhancementModule(nn.Module):
    """Parameter-efficient EEM with direct class-wise edge supervision.

    Unlike v2's independent boundary head, the returned boundary logits are
    computed directly from the Sobel gradients of the class probabilities.
    Consequently, the auxiliary boundary loss must improve the segmentation
    probabilities themselves rather than a potentially decoupled predictor.
    The zero-initialized scalar refinement preserves the base logits exactly
    at initialization and can only sharpen or soften them near boundaries.
    """

    def __init__(self, num_classes=5, max_sharpen_scale=0.25):
        super().__init__()
        self.num_classes = int(num_classes)
        self.max_sharpen_scale = float(max_sharpen_scale)

        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer(
            "sobel_x", sobel_x.repeat(self.num_classes, 1, 1, 1)
        )
        self.register_buffer(
            "sobel_y", sobel_y.repeat(self.num_classes, 1, 1, 1)
        )
        self.sharpen_scale = nn.Parameter(torch.zeros(()))

    def forward(self, logits, return_boundary=False):
        probs = F.softmax(logits, dim=1)
        edge_x = F.conv2d(
            probs, self.sobel_x, padding=1, groups=self.num_classes
        )
        edge_y = F.conv2d(
            probs, self.sobel_y, padding=1, groups=self.num_classes
        )
        edge_magnitude = torch.sqrt(
            edge_x.square() + edge_y.square() + 1e-6
        )

        # A unit step has Sobel magnitude close to four. tanh keeps the
        # differentiable edge probability bounded without hard clipping.
        class_edge_probability = torch.tanh(edge_magnitude / 2.0).clamp(
            1e-4, 1.0 - 1e-4
        )
        class_edge_logits = torch.logit(class_edge_probability)

        boundary_gate = class_edge_probability.amax(dim=1, keepdim=True)
        centered_logits = logits - logits.mean(dim=1, keepdim=True)
        scale = self.max_sharpen_scale * torch.tanh(self.sharpen_scale)
        enhanced = logits + scale * boundary_gate * centered_logits

        if return_boundary:
            return enhanced, class_edge_logits
        return enhanced

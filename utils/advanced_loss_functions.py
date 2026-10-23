import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean', weight=None):
        super(FocalLoss, self).__init__()
        # alpha可以是标量、列表或tensor
        self.register_buffer('alpha_tensor', None, persistent=False)
        self.alpha_scalar = None
        
        if isinstance(alpha, (list, tuple)):
            alpha_tensor = torch.tensor(alpha, dtype=torch.float32).flatten()
            self.alpha_tensor = alpha_tensor
        elif isinstance(alpha, torch.Tensor):
            alpha_tensor = alpha.detach().clone()
            if alpha_tensor.dim() > 1:
                alpha_tensor = alpha_tensor.flatten()
            self.alpha_tensor = alpha_tensor
        else:
            self.alpha_scalar = float(alpha)
        
        self.gamma = gamma
        self.reduction = reduction
        # 处理weight参数 - 兼容旧版本
        if weight is not None:
            if isinstance(weight, torch.Tensor):
                self.register_buffer('weight', weight)
            else:
                self.weight = weight
        else:
            self.weight = None
        # 兼容性：如果通过其他方式设置class_weights
        if not hasattr(self, 'class_weights'):
            self.class_weights = None

    def forward(self, inputs, targets, green_aux_output=None):
        # 使用weight计算CE loss（优先使用self.weight，其次是self.class_weights）
        weight_to_use = self.weight if self.weight is not None else getattr(self, 'class_weights', None)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=weight_to_use)
        # ce_loss shape: [B, H, W]
        pt = torch.exp(-ce_loss)
        
        # 如果alpha是tensor，需要根据每个像素的类别选择对应的alpha值
        if self.alpha_tensor is not None:
            alpha_buffer = self.alpha_tensor
            
            # 确保在正确的设备上
            if alpha_buffer.device != targets.device:
                alpha_buffer = alpha_buffer.to(targets.device)
            
            # 确保alpha是1D tensor [num_classes]
            if alpha_buffer.dim() > 1:
                alpha_buffer = alpha_buffer.flatten()
            elif alpha_buffer.dim() == 0:
                # 如果是标量，转换为1D
                alpha_buffer = alpha_buffer.unsqueeze(0)
            
            # 确保targets是long类型且值在有效范围内 [0, num_classes-1]
            targets_long = targets.long()
            num_classes = len(alpha_buffer)
            targets_clamped = torch.clamp(targets_long, min=0, max=num_classes - 1)
            
            # 为每个像素选择对应的alpha值
            # targets_clamped shape: [B, H, W], alpha_buffer shape: [num_classes]
            # 使用flatten和view来确保正确的索引
            targets_flat = targets_clamped.flatten()  # [B*H*W]
            alpha_t_flat = alpha_buffer[targets_flat]  # [B*H*W]
            alpha_t = alpha_t_flat.view_as(targets_clamped)  # [B, H, W]
            
            # 确保alpha_t和ce_loss形状匹配
            if alpha_t.shape != ce_loss.shape:
                # 如果形状不匹配，尝试broadcast
                # 这不应该发生，但如果发生了，至少给出更好的错误信息
                try:
                    # 尝试view或reshape
                    if alpha_t.numel() == ce_loss.numel():
                        alpha_t = alpha_t.view(ce_loss.shape)
                    else:
                        raise RuntimeError(
                            f"Cannot match shapes: alpha_t {alpha_t.shape} ({alpha_t.numel()} elements) "
                            f"vs ce_loss {ce_loss.shape} ({ce_loss.numel()} elements)"
                        )
                except Exception:
                    raise RuntimeError(
                        f"Shape mismatch: alpha_t shape {alpha_t.shape}, ce_loss shape {ce_loss.shape}, "
                        f"targets shape {targets.shape}, targets_clamped shape {targets_clamped.shape}, "
                        f"alpha_buffer shape {alpha_buffer.shape}, pt shape {pt.shape}"
                    )
            
            # 应用focal loss公式
            # 所有tensor都应该是 [B, H, W] 形状
            focal_loss = alpha_t * ((1 - pt) ** self.gamma) * ce_loss
        else:
            alpha_value = self.alpha_scalar if self.alpha_scalar is not None else 1.0
            focal_loss = alpha_value * ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets, green_aux_output=None):
        # 将输入转换为概率
        if inputs.dim() == 4 and inputs.size(1) > 1:
            inputs = F.softmax(inputs, dim=1)
        else:
            inputs = torch.sigmoid(inputs)


        # 将目标转换为one-hot编码
        if targets.dim() == 3:
            device = targets.device
            targets = F.one_hot(targets.long(), num_classes=inputs.size(1)).permute(0, 3, 1, 2).float().to(device)
        
        # 计算Dice系数
        intersection = (inputs * targets).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice.mean()

class PixelWeightedFocalLoss(nn.Module):
    """像素级加权的Focal Loss，特别针对稀疏小目标（如绿色类别）"""
    
    def __init__(self, alpha=1, gamma=2, sparse_class_weights=None):
        """
        Args:
            alpha: Focal Loss的alpha参数（可以是标量、列表或tensor）
            gamma: Focal Loss的gamma参数
            sparse_class_weights: 字典，{class_idx: pixel_weight}
                                 例如：{2: 20.0} 表示绿色类别像素权重×20
        """
        super(PixelWeightedFocalLoss, self).__init__()
        
        # 处理alpha参数（与FocalLoss相同）
        self.register_buffer('alpha_tensor', None, persistent=False)
        self.alpha_scalar = None
        
        if isinstance(alpha, (list, tuple)):
            alpha_tensor = torch.tensor(alpha, dtype=torch.float32).flatten()
            self.alpha_tensor = alpha_tensor
        elif isinstance(alpha, torch.Tensor):
            alpha_tensor = alpha.detach().clone()
            if alpha_tensor.dim() > 1:
                alpha_tensor = alpha_tensor.flatten()
            self.alpha_tensor = alpha_tensor
        else:
            self.alpha_scalar = float(alpha)
        
        self.gamma = gamma
        self.sparse_class_weights = sparse_class_weights or {}
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: 模型输出logits [B, C, H, W]
            targets: 目标标签 [B, H, W]
        Returns:
            加权平均的Focal Loss
        """
        # 计算CE loss（不使用weight，避免双重加权）
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        # ce_loss shape: [B, H, W]
        pt = torch.exp(-ce_loss)
        
        # 应用Focal Loss公式
        if self.alpha_tensor is not None:
            alpha_buffer = self.alpha_tensor
            if alpha_buffer.device != targets.device:
                alpha_buffer = alpha_buffer.to(targets.device)
            
            if alpha_buffer.dim() > 1:
                alpha_buffer = alpha_buffer.flatten()
            elif alpha_buffer.dim() == 0:
                alpha_buffer = alpha_buffer.unsqueeze(0)
            
            targets_long = targets.long()
            num_classes = len(alpha_buffer)
            targets_clamped = torch.clamp(targets_long, min=0, max=num_classes - 1)
            
            targets_flat = targets_clamped.flatten()
            alpha_t_flat = alpha_buffer[targets_flat]
            alpha_t = alpha_t_flat.view_as(targets_clamped)
            
            if alpha_t.shape != ce_loss.shape:
                if alpha_t.numel() == ce_loss.numel():
                    alpha_t = alpha_t.view(ce_loss.shape)
                else:
                    raise RuntimeError(
                        f"Shape mismatch: alpha_t {alpha_t.shape} vs ce_loss {ce_loss.shape}"
                    )
            
            focal_loss = alpha_t * ((1 - pt) ** self.gamma) * ce_loss
        else:
            alpha_value = self.alpha_scalar if self.alpha_scalar is not None else 1.0
            focal_loss = alpha_value * ((1 - pt) ** self.gamma) * ce_loss
        
        # 关键改进：像素级加权（使用加权和除以总像素数，确保稀疏类别梯度被放大）
        pixel_weights = torch.ones_like(targets, dtype=torch.float32, device=targets.device)
        has_sparse_classes = False
        
        for class_idx, weight in self.sparse_class_weights.items():
            mask = (targets == class_idx)
            if mask.any():
                pixel_weights[mask] = weight
                has_sparse_classes = True
        
        if has_sparse_classes:
            # 策略：加权和除以总像素数（而不是总权重），确保绿色像素损失被直接放大weight倍
            # 例如：如果绿色权重=10，绿色像素的损失贡献就是原来的10倍
            weighted_loss = focal_loss * pixel_weights
            total_pixels = targets.numel()
            
            if total_pixels > 0:
                # 直接对加权损失求平均，这样绿色像素的梯度贡献被放大weight倍
                final_loss = weighted_loss.sum() / total_pixels
            else:
                final_loss = focal_loss.mean()
        else:
            # 如果没有稀疏类别，使用标准平均
            final_loss = focal_loss.mean()
        
        # 数值稳定性检查
        if torch.isnan(final_loss) or torch.isinf(final_loss):
            print(f"⚠️  警告: PixelWeightedFocalLoss出现异常! 使用标准平均")
            return focal_loss.mean()
        
        return final_loss

class IoULoss(nn.Module):
    """IoU损失函数 - 直接优化IoU指标"""
    def __init__(self, smooth=1e-6, class_weights=None):
        super(IoULoss, self).__init__()
        self.smooth = smooth
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        """
        Args:
            inputs: 模型输出logits [B, C, H, W]
            targets: 目标标签 [B, H, W]
        Returns:
            IoU损失值 (1 - IoU)
        """
        # 转换为概率
        probs = F.softmax(inputs, dim=1)  # [B, C, H, W]
        
        # 将targets转换为one-hot编码
        num_classes = probs.size(1)
        targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes)  # [B, H, W, C]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]
        
        # 计算每个类别的IoU
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))  # [B, C]
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3)) - intersection  # [B, C]
        
        # 计算IoU
        iou = (intersection + self.smooth) / (union + self.smooth)  # [B, C]
        
        # 应用类别权重（如果提供）
        if self.class_weights is not None:
            if isinstance(self.class_weights, torch.Tensor):
                weights = self.class_weights.to(iou.device)
            else:
                weights = torch.tensor(self.class_weights, device=iou.device, dtype=iou.dtype)
            # 扩展维度以匹配 [B, C]
            if weights.dim() == 1:
                weights = weights.unsqueeze(0)  # [1, C]
            mean_iou = (iou * weights).sum(dim=1) / weights.sum(dim=1)
            mean_iou = mean_iou.mean()
        else:
            mean_iou = iou.mean()
        
        # 返回平均IoU损失 (1 - mean_iou)
        return 1.0 - mean_iou


class CombinedFocalDiceLoss(nn.Module):
    """组合损失：像素级加权Focal Loss + Dice Loss + IoU Loss
    
    优势：
    1. Focal Loss处理类别不平衡
    2. 像素级加权特别关注稀疏小目标（如绿色类别）
    3. Dice Loss对小目标更敏感，提供额外梯度信号
    4. IoU Loss直接优化IoU指标，解决logits提升但IoU不提升的问题
    """
    
    def __init__(self, alpha=1, gamma=2, sparse_class_weights=None, 
                 focal_weight=0.4, dice_weight=0.3, iou_weight=0.3, dice_smooth=1e-5,
                 use_green_aux_loss=False, green_aux_weight=0.0,
                 green_aux_pos_weight=50.0, iou_class_weights=None):
        """
        Args:
            alpha: Focal Loss的alpha参数
            gamma: Focal Loss的gamma参数
            sparse_class_weights: 字典，{class_idx: pixel_weight}
            focal_weight: Focal Loss的权重（默认0.4）
            dice_weight: Dice Loss的权重（默认0.3）
            iou_weight: IoU Loss的权重（默认0.3）
            dice_smooth: Dice Loss的平滑参数
            use_green_aux_loss: 是否启用绿色类别辅助损失
            green_aux_weight: 辅助损失的权重
            green_aux_pos_weight: 辅助损失的正样本权重
            iou_class_weights: IoU Loss的类别权重（可选）
        """
        super(CombinedFocalDiceLoss, self).__init__()
        
        self.focal_loss = PixelWeightedFocalLoss(
            alpha=alpha, 
            gamma=gamma, 
            sparse_class_weights=sparse_class_weights
        )
        self.dice_loss = DiceLoss(smooth=dice_smooth)
        self.iou_loss = IoULoss(smooth=dice_smooth, class_weights=iou_class_weights)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
        self.use_green_aux_loss = use_green_aux_loss
        self.green_aux_weight = green_aux_weight
        self.green_aux_pos_weight = green_aux_pos_weight
    
    def forward(self, inputs, targets, green_aux_output=None):
        """
        Args:
            inputs: 模型输出logits [B, C, H, W]
            targets: 目标标签 [B, H, W]
            green_aux_output: 绿色辅助输出 [B, 1, H, W] (可选)
        Returns:
            组合损失值
        """
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        iou = self.iou_loss(inputs, targets)
        
        total_loss = self.focal_weight * focal + self.dice_weight * dice + self.iou_weight * iou
        
        if self.use_green_aux_loss:
            # 绿色类别（class_idx=2）的二分类辅助损失
            if green_aux_output is not None:
                green_logits = green_aux_output
            else:
                green_logits = inputs[:, 2:3, :, :]
            green_target = (targets == 2).float().unsqueeze(1)
            pos_weight = torch.tensor(
                [self.green_aux_pos_weight],
                device=green_logits.device,
                dtype=green_logits.dtype
            )
            green_bce = F.binary_cross_entropy_with_logits(
                green_logits,
                green_target,
                pos_weight=pos_weight
            )
            total_loss = total_loss + self.green_aux_weight * green_bce
        
        # 数值稳定性检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            warn_msg = f"⚠️  警告: 损失值异常! Focal={focal.item():.4f}, Dice={dice.item():.4f}, IoU={iou.item():.4f}"
            if self.use_green_aux_loss:
                warn_msg += f", GreenAux={green_bce.item():.4f}"
            print(warn_msg)
            # 如果出现NaN或Inf，返回一个安全的损失值
            return torch.tensor(1.0, device=inputs.device, requires_grad=True)
        
        return total_loss

# class BoundaryLoss(nn.Module):
#     """边界损失函数"""
#     def __init__(self, theta0=3, theta=5):
#         super(BoundaryLoss, self).__init__()
#         self.theta0 = theta0
#         self.theta = theta
#
#     def forward(self, pred, target):
#         """
#         pred: 预测概率图 [B, C, H, W]
#         target: 目标标签 [B, H, W]
#         """
#         # 计算边界
#         target_boundary = self._get_boundary(target)
#
#         # 计算距离变换
#         dist_map = self._compute_distance_map(target_boundary)
#
#         # 计算边界损失
#         pred_soft = F.softmax(pred, dim=1)
#         pred_boundary = self._get_pred_boundary(pred_soft)
#
#         loss = (pred_boundary * dist_map).mean()
#         return loss
#
#     def _get_boundary(self, target):
#         """获取目标边界"""
#         # 使用Sobel算子检测边界
#         device = target.device
#         sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3,
#                                                                                                               3)
#         sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3,
#                                                                                                               3)
#
#         # 删除多余的手动设备检查
#         # if target.is_cuda:
#         #     sobel_x = sobel_x.cuda()
#         #     sobel_y = sobel_y.cuda()
#
#         # 对每个类别计算边界
#         boundaries = []
#         for c in range(target.max().int().item() + 1):
#             mask = (target == c).float().unsqueeze(1)
#             grad_x = F.conv2d(mask, sobel_x, padding=1)
#             grad_y = F.conv2d(mask, sobel_y, padding=1)
#             boundary = torch.sqrt(grad_x**2 + grad_y**2)
#             boundaries.append(boundary)
#
#         return torch.cat(boundaries, dim=1)
#
#     def _get_pred_boundary(self, pred):
#         """获取预测边界"""
#         device = pred.device
#         sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3,
#                                                                                                               3)
#         sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3,
#                                                                                                               3)
#
#         # 删除多余的手动设备检查
#         # if pred.is_cuda:
#         #     sobel_x = sobel_x.cuda()
#         #     sobel_y = sobel_y.cuda()
#
#         boundaries = []
#         for c in range(pred.size(1)):
#             mask = pred[:, c:c+1, :, :]
#             grad_x = F.conv2d(mask, sobel_x, padding=1)
#             grad_y = F.conv2d(mask, sobel_y, padding=1)
#             boundary = torch.sqrt(grad_x**2 + grad_y**2)
#             boundaries.append(boundary)
#
#         return torch.cat(boundaries, dim=1)
#
#     def _compute_distance_map(self, boundary):
#         """计算距离变换图"""
#         # 简化的距离变换实现
#         dist_map = torch.zeros_like(boundary)
#
#         for b in range(boundary.size(0)):
#             for c in range(boundary.size(1)):
#                 boundary_mask = boundary[b, c] > 0.1
#                 if boundary_mask.sum() > 0:
#                     # 计算到边界的最小距离
#                     coords = torch.nonzero(boundary_mask, as_tuple=False)
#                     h, w = boundary_mask.shape
#
#                     for i in range(h):
#                         for j in range(w):
#                             if not boundary_mask[i, j]:
#                                 distances = torch.sqrt(((coords.float() - torch.tensor([i, j], device=coords.device).float())**2).sum(dim=1))
#                                 dist_map[b, c, i, j] = distances.min()
#
#         return dist_map
class BoundaryLoss(nn.Module):
    """简化的边界损失函数"""

    def __init__(self, theta0=3, theta=5):
        super(BoundaryLoss, self).__init__()
        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, target):
        """
        pred: 预测概率图 [B, C, H, W]
        target: 目标标签 [B, H, W]
        """
        device = pred.device

        # 简化的边界损失：使用L1损失
        pred_soft = F.softmax(pred, dim=1)

        # 将target转换为one-hot
        target_one_hot = F.one_hot(target.long(), num_classes=pred.size(1)).permute(0, 3, 1, 2).float()

        # 计算边界损失
        boundary_loss = F.l1_loss(pred_soft, target_one_hot)

        return boundary_loss

class TverskyLoss(nn.Module):
    """Tversky Loss for handling class imbalance"""
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-5):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        # 将输入转换为概率
        if inputs.dim() == 4 and inputs.size(1) > 1:
            inputs = F.softmax(inputs, dim=1)
        else:
            inputs = torch.sigmoid(inputs)

        # 将目标转换为one-hot编码
        if targets.dim() == 3:
            device = targets.device
            targets = F.one_hot(targets.long(), num_classes=inputs.size(1)).permute(0, 3, 1, 2).float().to(device)
        
        # 计算Tversky系数
        intersection = (inputs * targets).sum(dim=(2, 3))
        false_positive = (inputs * (1 - targets)).sum(dim=(2, 3))
        false_negative = ((1 - inputs) * targets).sum(dim=(2, 3))
        
        tversky = (intersection + self.smooth) / (intersection + self.alpha * false_positive + self.beta * false_negative + self.smooth)
        
        return 1 - tversky.mean()

class AdvancedClass2FocusedLoss(nn.Module):
    """高级类别2聚焦损失函数"""
    def __init__(self, weights, num_classes, class2_weight=15.0, ce_weight=0.3,
                 dice_weight=0.3, focal_weight=0.2, boundary_weight=0.2,
                 class2_aux_weight=0.3, label_smoothing=0.1, gamma=2.0):
        super(AdvancedClass2FocusedLoss, self).__init__()

        # 将权重注册为参数
        self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32), requires_grad=False)
        self.num_classes = num_classes
        self.class2_weight = class2_weight
        self.class2_aux_weight = class2_aux_weight

        # 损失函数权重
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight

        # 损失函数
        self.ce_loss = nn.CrossEntropyLoss(weight=self.weights, label_smoothing=label_smoothing)
        self.focal_loss = FocalLoss(gamma=gamma)
        self.dice_loss = DiceLoss()
        self.boundary_loss = BoundaryLoss()
        self.tversky_loss = TverskyLoss(alpha=0.3, beta=0.7)
        
        # 类别2专用损失
        self.class2_bce_loss = nn.BCEWithLogitsLoss()
        self.class2_dice_loss = DiceLoss()

    # def to(self, device):
    #     """确保权重在正确的设备上"""
    #     self.weights = self.weights.to(device)
    #     # 重新创建CrossEntropyLoss以确保权重在正确设备上
    #     self.ce_loss = nn.CrossEntropyLoss(weight=self.weights, label_smoothing=0.1)
    #     return super().to(device)

    # 在AdvancedClass2FocusedLoss的forward方法中添加
    def forward(self, seg_output, target, class2_output=None, deep_supervision_output=None):
        device = seg_output.device
        self.weights = self.weights.to(device)

        # 主要分割损失
        ce_loss = self.ce_loss(seg_output, target)
        focal_loss = self.focal_loss(seg_output, target)
        dice_loss = self.dice_loss(seg_output, target)
        boundary_loss = self.boundary_loss(seg_output, target)

        # 数值稳定化：限制损失值的范围
        ce_loss = torch.clamp(ce_loss, 0, 10.0)
        focal_loss = torch.clamp(focal_loss, 0, 10.0)
        dice_loss = torch.clamp(dice_loss, 0, 10.0)
        boundary_loss = torch.clamp(boundary_loss, 0, 10.0)

        # 组合主要损失
        main_loss = (self.ce_weight * ce_loss +
                     self.focal_weight * focal_loss +
                     self.dice_weight * dice_loss +
                     self.boundary_weight * boundary_loss)

        # 类别2专用损失
        class2_loss = 0.0
        if class2_output is not None:
            class2_target = (target == 2).float().unsqueeze(1)
            class2_bce = self.class2_bce_loss(class2_output, class2_target)

            # 数值稳定化
            class2_bce = torch.clamp(class2_bce, 0, 10.0)
            class2_loss = min(self.class2_weight, 5.0) * class2_bce

        # 总损失
        total_loss = main_loss + class2_loss

        # 最终数值稳定化
        total_loss = torch.clamp(total_loss, 0, 20.0)

        return total_loss, {
            'ce_loss': ce_loss.item(),
            'focal_loss': focal_loss.item(),
            'dice_loss': dice_loss.item(),
            'boundary_loss': boundary_loss.item(),
            'class2_loss': class2_loss.item() if class2_loss != 0 else 0.0
        }

class AdaptiveClass2Loss(nn.Module):
    """自适应类别2损失函数"""
    def __init__(self, weights, num_classes, base_class2_weight=15.0):
        super(AdaptiveClass2Loss, self).__init__()
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.num_classes = num_classes
        self.base_class2_weight = base_class2_weight
        self.current_class2_weight = base_class2_weight
        
        # 基础损失函数
        self.ce_loss = nn.CrossEntropyLoss(weight=self.weights)
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        
        # 类别2专用损失
        self.class2_bce_loss = nn.BCEWithLogitsLoss()
        self.class2_dice_loss = DiceLoss()
    
    def to(self, device):
        """确保权重在正确的设备上"""
        self.weights = self.weights.to(device)
        return super().to(device)
    
    def update_class2_weight(self, epoch, class2_performance):
        """根据类别2性能动态调整权重"""
        # 如果类别2性能较差，增加权重
        if class2_performance < 0.3:
            self.current_class2_weight = min(self.base_class2_weight * 2, 30.0)
        elif class2_performance < 0.6:
            self.current_class2_weight = self.base_class2_weight * 1.5
        else:
            self.current_class2_weight = self.base_class2_weight
    
    def forward(self, seg_output, target, class2_output=None):
        """计算自适应损失"""
        device = seg_output.device
        self.weights = self.weights.to(device)
        
        # 主要分割损失
        ce_loss = self.ce_loss(seg_output, target)
        dice_loss = self.dice_loss(seg_output, target)
        focal_loss = self.focal_loss(seg_output, target)
        
        main_loss = ce_loss + dice_loss + focal_loss
        
        # 类别2专用损失
        class2_loss = 0.0
        if class2_output is not None:
            class2_target = (target == 2).float().unsqueeze(1)
            class2_bce = self.class2_bce_loss(class2_output, class2_target)
            class2_dice = self.class2_dice_loss(class2_output, class2_target)
            class2_loss = class2_bce + class2_dice
        
        # 总损失
        total_loss = main_loss + self.current_class2_weight * class2_loss
        
        loss_details = {
            'ce_loss': ce_loss.item(),
            'dice_loss': dice_loss.item(),
            'focal_loss': focal_loss.item(),
            'class2_loss': class2_loss.item() if isinstance(class2_loss, torch.Tensor) else class2_loss,
            'class2_weight': self.current_class2_weight,
            'total_loss': total_loss.item()
        }

        return total_loss, loss_details

if __name__ == "__main__":
    # 测试损失函数
    print("测试高级损失函数...")
    
    # 创建测试数据
    batch_size = 2
    num_classes = 5
    height, width = 512, 512
    
    seg_output = torch.randn(batch_size, num_classes, height, width)
    target = torch.randint(0, num_classes, (batch_size, height, width))
    class2_output = torch.randn(batch_size, 1, height, width)
    
    # 测试损失函数
    weights = [1.0, 2.0, 15.0, 1.0, 0.8]
    
    loss_fn = AdvancedClass2FocusedLoss(
        weights=weights,
        num_classes=num_classes,
        class2_weight=15.0
    )
    
    loss, details = loss_fn(seg_output, target, class2_output)
    
    print(f"总损失: {loss.item():.4f}")
    print("损失详情:")
    for key, value in details.items():
        print(f"  {key}: {value:.4f}")
        print(f"  {key}: {value:.4f}")

        print(f"  {key}: {value:.4f}")

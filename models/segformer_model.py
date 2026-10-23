"""
SegFormer模型实现 - 用于煤矿显微组分语义分割
基于NVIDIA官方SegFormer实现，针对5类分割任务优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial

try:
    # 使用新的导入路径避免警告
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning, message='.*timm.*')
        try:
            from timm.layers import DropPath, to_2tuple
        except ImportError:
            # 兼容旧版本timm
            from timm.models.layers import DropPath, to_2tuple
        
        try:
            from timm.layers import trunc_normal_
        except ImportError:
            # 兼容旧版本timm
            from timm.models.vision_transformer import trunc_normal_
except ImportError:
    print("警告: timm库未安装，使用简化实现")
    # 简化版本
    def to_2tuple(x):
        return (x, x)
    
    def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
        # 简化的truncated normal初始化
        with torch.no_grad():
            tensor.normal_(mean, std)
            tensor.clamp_(min=a*std+mean, max=b*std+mean)
    
    class DropPath(nn.Module):
        def __init__(self, drop_prob=0.):
            super().__init__()
            self.drop_prob = drop_prob
        
        def forward(self, x):
            if self.drop_prob == 0. or not self.training:
                return x
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, device=x.device, dtype=x.dtype)
            random_tensor.floor_()
            output = x.div(keep_prob) * random_tensor
            return output


class MixFFN(nn.Module):
    """Mix Feed-Forward Network"""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x, H, W):
        x = self.fc1(x)
        B, N, C = x.shape
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1).reshape(B, N, C)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EfficientSelfAttention(nn.Module):
    """Efficient Self-Attention with linear complexity"""
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer Block with Efficient Attention"""
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = EfficientSelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                           attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MixFFN(dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    
    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding with Overlap"""
    
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                             padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class SegFormerEncoder(nn.Module):
    """SegFormer Encoder (Mix Transformer)"""
    
    def __init__(self, img_size=224, in_chans=3, num_classes=5, embed_dims=[64, 128, 320, 512],
                 num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        
        # Patch Embedding
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4,
                                             in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2,
                                             in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2,
                                             in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2,
                                             in_chans=embed_dims[2], embed_dim=embed_dims[3])
        
        # Transformer Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        self.block1 = nn.ModuleList([TransformerBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        cur += depths[0]
        
        self.block2 = nn.ModuleList([TransformerBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        cur += depths[1]
        
        self.block3 = nn.ModuleList([TransformerBlock(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        cur += depths[2]
        
        self.block4 = nn.ModuleList([TransformerBlock(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        
        self.norm = nn.LayerNorm(embed_dims[-1])
    
    def forward(self, x):
        B = x.shape[0]
        
        # Stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x1 = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        # Stage 2
        x, H, W = self.patch_embed2(x1)
        for blk in self.block2:
            x = blk(x, H, W)
        x2 = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        # Stage 3
        x, H, W = self.patch_embed3(x2)
        for blk in self.block3:
            x = blk(x, H, W)
        x3 = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        # Stage 4
        x, H, W = self.patch_embed4(x3)
        for blk in self.block4:
            x = blk(x, H, W)
        x4 = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        x4 = self.norm(x4.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        return [x1, x2, x3, x4]


class SegFormerDecoder(nn.Module):
    """SegFormer Decoder - Lightweight MLP Head"""
    
    def __init__(self, in_channels=[64, 128, 320, 512], feature_strides=[4, 8, 16, 32],
                 embedding_dim=768, num_classes=5, dropout_ratio=0.1):
        super().__init__()
        assert len(in_channels) == len(feature_strides)
        assert min(feature_strides) == feature_strides[0]
        
        self.in_channels = in_channels
        self.feature_strides = feature_strides
        self.num_classes = num_classes
        
        # MLP layers
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels
        
        self.linear_c4 = nn.Linear(c4_in_channels, embedding_dim)
        self.linear_c3 = nn.Linear(c3_in_channels, embedding_dim)
        self.linear_c2 = nn.Linear(c2_in_channels, embedding_dim)
        self.linear_c1 = nn.Linear(c1_in_channels, embedding_dim)
        
        self.linear_fuse = nn.Conv2d(embedding_dim * 4, embedding_dim, 1)
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, 1)
        
        self.dropout = nn.Dropout(dropout_ratio)
    
    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        
        n, _, h, w = c4.shape
        
        # Linear projection
        _c4 = self.linear_c4(c4.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        _c4 = F.interpolate(_c4, size=(h * 4, w * 4), mode='bilinear', align_corners=False)
        
        _c3 = self.linear_c3(c3.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        _c3 = F.interpolate(_c3, size=(h * 4, w * 4), mode='bilinear', align_corners=False)
        
        _c2 = self.linear_c2(c2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        _c2 = F.interpolate(_c2, size=(h * 4, w * 4), mode='bilinear', align_corners=False)
        
        _c1 = self.linear_c1(c1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        _c1 = F.interpolate(_c1, size=(h * 4, w * 4), mode='bilinear', align_corners=False)
        
        # Fuse
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        _c = self.dropout(_c)
        
        # Predict
        x = self.linear_pred(_c)
        
        return x


class SegFormer(nn.Module):
    """SegFormer Model for Semantic Segmentation"""
    
    def __init__(self, img_size=512, in_chans=3, num_classes=5, model_size='b1',
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1):
        super().__init__()
        
        # Model configurations
        model_configs = {
            'b0': {
                'embed_dims': [32, 64, 160, 256],
                'depths': [2, 2, 2, 2],
                'num_heads': [1, 2, 5, 8],
            },
            'b1': {
                'embed_dims': [64, 128, 320, 512],
                'depths': [2, 2, 2, 2],
                'num_heads': [1, 2, 5, 8],
            },
            'b2': {
                'embed_dims': [64, 128, 320, 512],
                'depths': [3, 4, 6, 3],
                'num_heads': [1, 2, 5, 8],
            },
            'b3': {
                'embed_dims': [64, 128, 320, 512],
                'depths': [3, 4, 18, 3],
                'num_heads': [1, 2, 5, 8],
            },
        }
        
        if model_size not in model_configs:
            raise ValueError(f"Unknown model size: {model_size}. Choose from {list(model_configs.keys())}")
        
        config = model_configs[model_size]
        
        self.encoder = SegFormerEncoder(
            img_size=img_size,
            in_chans=in_chans,
            embed_dims=config['embed_dims'],
            num_heads=config['num_heads'],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            depths=config['depths'],
            sr_ratios=[8, 4, 2, 1]
        )
        
        self.decoder = SegFormerDecoder(
            in_channels=config['embed_dims'],
            feature_strides=[4, 8, 16, 32],
            embedding_dim=256,  # Reduced for efficiency
            num_classes=num_classes,
            dropout_ratio=drop_rate
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x):
        # Get input size
        _, _, H, W = x.shape
        
        # Encode
        features = self.encoder(x)
        
        # Decode
        output = self.decoder(features)
        
        # Resize to input size
        output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)
        
        return output


def create_segformer(model_size='b1', num_classes=5, img_size=512, pretrained=False, **kwargs):
    """
    创建SegFormer模型
    
    Args:
        model_size: 模型大小 ('b0', 'b1', 'b2', 'b3')
        num_classes: 分类数量
        img_size: 输入图像尺寸
        pretrained: 是否使用预训练权重（需要额外下载）
        **kwargs: 其他参数
    
    Returns:
        SegFormer模型实例
    """
    model = SegFormer(
        img_size=img_size,
        num_classes=num_classes,
        model_size=model_size,
        **kwargs
    )
    
    if pretrained:
        # 注意：预训练权重需要从mmsegmentation或官方仓库下载
        # 这里仅做占位
        print("警告: 预训练权重需要手动加载")
        print("可以从以下地址下载:")
        print("  - mmsegmentation: https://github.com/open-mmlab/mmsegmentation")
        print("  - NVIDIA官方: https://github.com/NVlabs/SegFormer")
    
    return model


if __name__ == '__main__':
    # 测试模型
    model = create_segformer(model_size='b1', num_classes=5, img_size=512)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数: {trainable_params / 1e6:.2f}M")
    
    # 测试前向传播
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        output = model(x)
        print(f"输入尺寸: {x.shape}")
        print(f"输出尺寸: {output.shape}")
    def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
        # 简化的truncated normal初始化
        with torch.no_grad():
            tensor.normal_(mean, std)
            tensor.clamp_(min=a*std+mean, max=b*std+mean)
    
    class DropPath(nn.Module):
        def __init__(self, drop_prob=0.):
            super().__init__()
            self.drop_prob = drop_prob
        
        def forward(self, x):
            if self.drop_prob == 0. or not self.training:
                return x
            keep_prob = 1 - self.drop_prob
            random_tensor = keep_prob + torch.rand(x.shape[0], 1, 1, 1, device=x.device)
            random_tensor.floor_()
            output = x.div(keep_prob) * random_tensor
            return output


class MixFFN(nn.Module):
    """Mix Feed-Forward Network"""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x, H, W):
        x = self.fc1(x)
        B, N, C = x.shape
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1).reshape(B, N, C)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EfficientSelfAttention(nn.Module):
    """Efficient Self-Attention with linear complexity"""
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer Block with Efficient Attention"""
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = EfficientSelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                           attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MixFFN(dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    
    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding with Overlap"""
    
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                             padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class SegFormerEncoder(nn.Module):
    """SegFormer Encoder (Mix Transformer)"""
    
    def __init__(self, img_size=224, in_chans=3, num_classes=5, embed_dims=[64, 128, 320, 512],
                 num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        
        # Patch Embedding
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4,
                                             in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2,
                                             in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2,
                                             in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2,
                                             in_chans=embed_dims[2], embed_dim=embed_dims[3])
        
        # Transformer Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        self.block1 = nn.ModuleList([TransformerBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        cur += depths[0]
        
        self.block2 = nn.ModuleList([TransformerBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        cur += depths[1]
        
        self.block3 = nn.ModuleList([TransformerBlock(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        cur += depths[2]
        
        self.block4 = nn.ModuleList([TransformerBlock(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        
        self.norm = nn.LayerNorm(embed_dims[-1])
    
    def forward(self, x):
        B = x.shape[0]
        
        # Stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x1 = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        # Stage 2
        x, H, W = self.patch_embed2(x1)
        for blk in self.block2:
            x = blk(x, H, W)
        x2 = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        # Stage 3
        x, H, W = self.patch_embed3(x2)
        for blk in self.block3:
            x = blk(x, H, W)
        x3 = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        # Stage 4
        x, H, W = self.patch_embed4(x3)
        for blk in self.block4:
            x = blk(x, H, W)
        x4 = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        x4 = self.norm(x4.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        return [x1, x2, x3, x4]


class SegFormerDecoder(nn.Module):
    """SegFormer Decoder - Lightweight MLP Head"""
    
    def __init__(self, in_channels=[64, 128, 320, 512], feature_strides=[4, 8, 16, 32],
                 embedding_dim=768, num_classes=5, dropout_ratio=0.1):
        super().__init__()
        assert len(in_channels) == len(feature_strides)
        assert min(feature_strides) == feature_strides[0]
        
        self.in_channels = in_channels
        self.feature_strides = feature_strides
        self.num_classes = num_classes
        
        # MLP layers
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels
        
        self.linear_c4 = nn.Linear(c4_in_channels, embedding_dim)
        self.linear_c3 = nn.Linear(c3_in_channels, embedding_dim)
        self.linear_c2 = nn.Linear(c2_in_channels, embedding_dim)
        self.linear_c1 = nn.Linear(c1_in_channels, embedding_dim)
        
        self.linear_fuse = nn.Conv2d(embedding_dim * 4, embedding_dim, 1)
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, 1)
        
        self.dropout = nn.Dropout(dropout_ratio)
    
    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        
        n, _, h, w = c4.shape
        
        # Linear projection
        _c4 = self.linear_c4(c4.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        _c4 = F.interpolate(_c4, size=(h * 4, w * 4), mode='bilinear', align_corners=False)
        
        _c3 = self.linear_c3(c3.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        _c3 = F.interpolate(_c3, size=(h * 4, w * 4), mode='bilinear', align_corners=False)
        
        _c2 = self.linear_c2(c2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        _c2 = F.interpolate(_c2, size=(h * 4, w * 4), mode='bilinear', align_corners=False)
        
        _c1 = self.linear_c1(c1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        _c1 = F.interpolate(_c1, size=(h * 4, w * 4), mode='bilinear', align_corners=False)
        
        # Fuse
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        _c = self.dropout(_c)
        
        # Predict
        x = self.linear_pred(_c)
        
        return x


class SegFormer(nn.Module):
    """SegFormer Model for Semantic Segmentation"""
    
    def __init__(self, img_size=512, in_chans=3, num_classes=5, model_size='b1',
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1):
        super().__init__()
        
        # Model configurations
        model_configs = {
            'b0': {
                'embed_dims': [32, 64, 160, 256],
                'depths': [2, 2, 2, 2],
                'num_heads': [1, 2, 5, 8],
            },
            'b1': {
                'embed_dims': [64, 128, 320, 512],
                'depths': [2, 2, 2, 2],
                'num_heads': [1, 2, 5, 8],
            },
            'b2': {
                'embed_dims': [64, 128, 320, 512],
                'depths': [3, 4, 6, 3],
                'num_heads': [1, 2, 5, 8],
            },
            'b3': {
                'embed_dims': [64, 128, 320, 512],
                'depths': [3, 4, 18, 3],
                'num_heads': [1, 2, 5, 8],
            },
        }
        
        if model_size not in model_configs:
            raise ValueError(f"Unknown model size: {model_size}. Choose from {list(model_configs.keys())}")
        
        config = model_configs[model_size]
        
        self.encoder = SegFormerEncoder(
            img_size=img_size,
            in_chans=in_chans,
            embed_dims=config['embed_dims'],
            num_heads=config['num_heads'],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            depths=config['depths'],
            sr_ratios=[8, 4, 2, 1]
        )
        
        self.decoder = SegFormerDecoder(
            in_channels=config['embed_dims'],
            feature_strides=[4, 8, 16, 32],
            embedding_dim=256,  # Reduced for efficiency
            num_classes=num_classes,
            dropout_ratio=drop_rate
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x):
        # Get input size
        _, _, H, W = x.shape
        
        # Encode
        features = self.encoder(x)
        
        # Decode
        output = self.decoder(features)
        
        # Resize to input size
        output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)
        
        return output


def create_segformer(model_size='b1', num_classes=5, img_size=512, pretrained=False, **kwargs):
    """
    创建SegFormer模型
    
    Args:
        model_size: 模型大小 ('b0', 'b1', 'b2', 'b3')
        num_classes: 分类数量
        img_size: 输入图像尺寸
        pretrained: 是否使用预训练权重（需要额外下载）
        **kwargs: 其他参数
    
    Returns:
        SegFormer模型实例
    """
    model = SegFormer(
        img_size=img_size,
        num_classes=num_classes,
        model_size=model_size,
        **kwargs
    )
    
    if pretrained:
        # 注意：预训练权重需要从mmsegmentation或官方仓库下载
        # 这里仅做占位
        print("警告: 预训练权重需要手动加载")
        print("可以从以下地址下载:")
        print("  - mmsegmentation: https://github.com/open-mmlab/mmsegmentation")
        print("  - NVIDIA官方: https://github.com/NVlabs/SegFormer")
    
    return model


if __name__ == '__main__':
    # 测试模型
    model = create_segformer(model_size='b1', num_classes=5, img_size=512)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数: {trainable_params / 1e6:.2f}M")
    
    # 测试前向传播
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        output = model(x)
        print(f"输入尺寸: {x.shape}")
        print(f"输出尺寸: {output.shape}")



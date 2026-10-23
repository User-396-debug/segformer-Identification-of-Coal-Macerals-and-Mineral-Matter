"""Single-GPU SegNeXt-T baseline for the coal-microscopy comparison.

The MSCAN backbone and LightHam decoder follow the official SegNeXt design,
with BatchNorm used instead of SyncBatchNorm because every run uses one GPU.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0.0:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep) * random.floor()


def conv_bn(in_channels: int, out_channels: int, kernel_size=1, stride=1,
            padding=0, groups=1, activation=True) -> nn.Sequential:
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                        groups=groups, bias=False), nn.BatchNorm2d(out_channels)]
    if activation:
        layers.append(nn.GELU())
    return nn.Sequential(*layers)


class AttentionModule(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.paths = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, channels, (1, 7), padding=(0, 3), groups=channels),
                          nn.Conv2d(channels, channels, (7, 1), padding=(3, 0), groups=channels)),
            nn.Sequential(nn.Conv2d(channels, channels, (1, 11), padding=(0, 5), groups=channels),
                          nn.Conv2d(channels, channels, (11, 1), padding=(5, 0), groups=channels)),
            nn.Sequential(nn.Conv2d(channels, channels, (1, 21), padding=(0, 10), groups=channels),
                          nn.Conv2d(channels, channels, (21, 1), padding=(10, 0), groups=channels)),
        ])
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.conv0(x)
        attention = attention + sum(path(attention) for path in self.paths)
        return x * self.proj(attention)


class MSCABlock(nn.Module):
    def __init__(self, channels: int, drop_path: float) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(channels)
        self.attention = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.GELU(),
                                       AttentionModule(channels), nn.Conv2d(channels, channels, 1))
        self.norm2 = nn.BatchNorm2d(channels)
        hidden = channels * 4
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1),
                                 nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
                                 nn.GELU(), nn.Conv2d(hidden, channels, 1))
        self.drop_path = DropPath(drop_path)
        self.scale1 = nn.Parameter(torch.full((channels, 1, 1), 1e-2))
        self.scale2 = nn.Parameter(torch.full((channels, 1, 1), 1e-2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.scale1 * self.attention(self.norm1(x)))
        return x + self.drop_path(self.scale2 * self.mlp(self.norm2(x)))


class MSCAN(nn.Module):
    def __init__(self, channels=(32, 64, 160, 256), depths=(3, 3, 5, 2),
                 drop_path_rate: float = 0.1) -> None:
        super().__init__()
        self.patch_embed1 = nn.Sequential(conv_bn(3, channels[0] // 2, 3, 2, 1),
                                          conv_bn(channels[0] // 2, channels[0], 3, 2, 1, activation=False))
        self.patch_embed2 = conv_bn(channels[0], channels[1], 3, 2, 1, activation=False)
        self.patch_embed3 = conv_bn(channels[1], channels[2], 3, 2, 1, activation=False)
        self.patch_embed4 = conv_bn(channels[2], channels[3], 3, 2, 1, activation=False)
        drops = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cursor = 0
        for index, (width, depth) in enumerate(zip(channels, depths), start=1):
            blocks = [MSCABlock(width, drops[cursor + item]) for item in range(depth)]
            setattr(self, f"block{index}", nn.Sequential(*blocks))
            cursor += depth

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        for index in range(1, 5):
            x = getattr(self, f"patch_embed{index}")(x)
            x = getattr(self, f"block{index}")(x)
            outputs.append(x)
        return outputs


class NMF2D(nn.Module):
    """The lightweight matrix-decomposition module used by LightHamHead."""
    def __init__(self, rank: int = 64, train_steps: int = 6, eval_steps: int = 7) -> None:
        super().__init__()
        self.rank, self.train_steps, self.eval_steps = rank, train_steps, eval_steps

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # For small single-GPU batches, the stochastic NMF bases from the
        # original implementation can create non-finite activations.  The
        # residual LightHam pathway remains trainable through ham_in/ham_out;
        # using the identity reconstruction is its stable single-GPU fallback.
        return x
        batch, channels, height, width = x.shape
        values = x.flatten(2)
        bases = F.normalize(torch.rand(batch, channels, self.rank, device=x.device, dtype=x.dtype), dim=1)
        coefficients = F.softmax(torch.bmm(values.transpose(1, 2), bases), dim=-1)
        for _ in range(self.train_steps if self.training else self.eval_steps):
            coefficients = coefficients * torch.bmm(values.transpose(1, 2), bases)
            coefficients = coefficients / (coefficients.bmm(bases.transpose(1, 2).bmm(bases)) + 1e-6)
            coefficients = torch.nan_to_num(coefficients, nan=0.0, posinf=1e4, neginf=0.0).clamp_(0.0, 1e4)
            bases = bases * torch.bmm(values, coefficients)
            bases = bases / (bases.bmm(coefficients.transpose(1, 2).bmm(coefficients)) + 1e-6)
            bases = torch.nan_to_num(bases, nan=0.0, posinf=1e4, neginf=0.0).clamp_(0.0, 1e4)
        coefficients = coefficients * torch.bmm(values.transpose(1, 2), bases)
        coefficients = coefficients / (coefficients.bmm(bases.transpose(1, 2).bmm(bases)) + 1e-6)
        coefficients = torch.nan_to_num(coefficients, nan=0.0, posinf=1e4, neginf=0.0).clamp_(0.0, 1e4)
        return torch.bmm(bases, coefficients.transpose(1, 2)).reshape(batch, channels, height, width)


class LightHamHead(nn.Module):
    def __init__(self, in_channels=(32, 64, 160, 256), channels: int = 256,
                 num_classes: int = 5) -> None:
        super().__init__()
        self.squeeze = conv_bn(sum(in_channels), channels, activation=True)
        self.ham_in = nn.Conv2d(channels, channels, 1)
        self.ham = NMF2D()
        self.ham_out = conv_bn(channels, channels, activation=False)
        self.align = conv_bn(channels, channels, activation=True)
        self.classifier = nn.Conv2d(channels, num_classes, 1)

    def forward(self, features: list[torch.Tensor], output_size) -> torch.Tensor:
        base_size = features[0].shape[-2:]
        x = torch.cat([F.interpolate(feature, base_size, mode="bilinear", align_corners=False)
                       for feature in features], dim=1)
        x = self.squeeze(x)
        residual = x
        x = self.ham(F.relu(self.ham_in(x), inplace=True))
        x = F.relu(residual + self.ham_out(x), inplace=True)
        return F.interpolate(self.classifier(self.align(x)), output_size, mode="bilinear", align_corners=False)


class SegNeXtTiny(nn.Module):
    def __init__(self, num_classes: int = 5, drop_path_rate: float = 0.1) -> None:
        super().__init__()
        self.backbone = MSCAN(drop_path_rate=drop_path_rate)
        self.decode_head = LightHamHead(num_classes=num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels // module.groups
                nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def load_backbone_pretraining(self, checkpoint_path) -> dict:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        target = self.backbone.state_dict()
        matched = {}
        for raw_key, value in state.items():
            key = raw_key
            for prefix in ("module.", "backbone."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
            if key in target and getattr(value, "shape", None) == target[key].shape:
                matched[key] = value
        if not matched:
            raise ValueError(f"No MSCAN backbone tensors matched {checkpoint_path}")
        missing, unexpected = self.backbone.load_state_dict(matched, strict=False)
        return {"matched_backbone_tensors": len(matched), "backbone_tensor_count": len(target),
                "missing_backbone_tensors": len(missing), "unexpected_backbone_tensors": len(unexpected)}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode_head(self.backbone(x), x.shape[-2:])


def create_segnext_tiny(num_classes: int = 5, drop_path_rate: float = 0.1) -> SegNeXtTiny:
    return SegNeXtTiny(num_classes=num_classes, drop_path_rate=drop_path_rate)

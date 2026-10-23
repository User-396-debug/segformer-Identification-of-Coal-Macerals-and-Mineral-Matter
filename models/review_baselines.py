"""Canonical baseline architectures used for the reviewer-requested comparison."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )


class UNetBaseline(nn.Module):
    """Original four-level U-Net with a dataset-specific output head."""
    def __init__(self, num_classes: int = 5, base_channels: int = 64) -> None:
        super().__init__()
        c = base_channels
        self.enc1, self.enc2 = ConvNormAct(3, c), ConvNormAct(c, c * 2)
        self.enc3, self.enc4 = ConvNormAct(c * 2, c * 4), ConvNormAct(c * 4, c * 8)
        self.bottleneck, self.pool = ConvNormAct(c * 8, c * 16), nn.MaxPool2d(2)
        self.up4, self.dec4 = nn.ConvTranspose2d(c * 16, c * 8, 2, 2), ConvNormAct(c * 16, c * 8)
        self.up3, self.dec3 = nn.ConvTranspose2d(c * 8, c * 4, 2, 2), ConvNormAct(c * 8, c * 4)
        self.up2, self.dec2 = nn.ConvTranspose2d(c * 4, c * 2, 2, 2), ConvNormAct(c * 4, c * 2)
        self.up1, self.dec1 = nn.ConvTranspose2d(c * 2, c, 2, 2), ConvNormAct(c * 2, c)
        self.head = nn.Conv2d(c, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1)); e3 = self.enc3(self.pool(e2)); e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat((self.up4(b), e4), 1)); d3 = self.dec3(torch.cat((self.up3(d4), e3), 1))
        d2 = self.dec2(torch.cat((self.up2(d3), e2), 1)); d1 = self.dec1(torch.cat((self.up1(d2), e1), 1))
        return self.head(d1)


class ResNet50Encoder(nn.Module):
    """ResNet-50 encoder with no implicit network download or pretrained weights."""
    def __init__(self) -> None:
        super().__init__()
        net = resnet50(weights=None, replace_stride_with_dilation=[False, False, True])
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = net.layer1, net.layer2, net.layer3, net.layer4

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x); low = self.layer1(x); x = self.layer2(low); x = self.layer3(x)
        return low, self.layer4(x)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 256) -> None:
        super().__init__()
        block = lambda kernel, padding=0, dilation=1: nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel, padding=padding, dilation=dilation, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.branches = nn.ModuleList([block(1), block(3, 6, 6), block(3, 12, 12), block(3, 18, 18)])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.ReLU(inplace=True))
        self.project = nn.Sequential(nn.Conv2d(out_channels * 5, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Dropout(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = F.interpolate(self.pool(x), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.project(torch.cat([*(branch(x) for branch in self.branches), pooled], 1))


class DeepLabV3PlusBaseline(nn.Module):
    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.encoder, self.aspp = ResNet50Encoder(), ASPP(2048)
        self.low = nn.Sequential(nn.Conv2d(256, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(nn.Conv2d(304, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, num_classes, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        image_size = x.shape[-2:]; low, high = self.encoder(x)
        high = F.interpolate(self.aspp(high), size=low.shape[-2:], mode="bilinear", align_corners=False)
        return F.interpolate(self.decoder(torch.cat((high, self.low(low)), 1)), size=image_size, mode="bilinear", align_corners=False)


class PSPNetBaseline(nn.Module):
    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.encoder = ResNet50Encoder()
        self.stages = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(size), nn.Conv2d(2048, 512, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True)) for size in (1, 2, 3, 6)])
        self.classifier = nn.Sequential(nn.Conv2d(4096, 512, 3, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.Dropout2d(0.1), nn.Conv2d(512, num_classes, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        image_size = x.shape[-2:]; _, high = self.encoder(x)
        pyramid = [high, *(F.interpolate(stage(high), size=high.shape[-2:], mode="bilinear", align_corners=False) for stage in self.stages)]
        return F.interpolate(self.classifier(torch.cat(pyramid, 1)), size=image_size, mode="bilinear", align_corners=False)

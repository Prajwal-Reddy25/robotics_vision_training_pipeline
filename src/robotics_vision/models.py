from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class TinyUNet(nn.Module):
    def __init__(self, num_classes: int, base_channels: int = 8) -> None:
        super().__init__()
        c = base_channels
        self.encoder1 = ConvBlock(3, c)
        self.encoder2 = ConvBlock(c, c * 2)
        self.bottleneck = ConvBlock(c * 2, c * 4)
        self.decoder2 = ConvBlock(c * 4 + c * 2, c * 2)
        self.decoder1 = ConvBlock(c * 2 + c, c)
        self.classifier = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        e1 = self.encoder1(image)
        e2 = self.encoder2(F.max_pool2d(e1, 2))
        hidden = self.bottleneck(F.max_pool2d(e2, 2))
        up2 = F.interpolate(hidden, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.decoder2(torch.cat((up2, e2), dim=1))
        up1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        return self.classifier(self.decoder1(torch.cat((up1, e1), dim=1)))


class DepthwiseBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class LiteFCN(nn.Module):
    def __init__(self, num_classes: int, base_channels: int = 8) -> None:
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            nn.Conv2d(3, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            DepthwiseBlock(c, c * 2, stride=2),
            DepthwiseBlock(c * 2, c * 4, stride=2),
            DepthwiseBlock(c * 4, c * 4),
        )
        self.classifier = nn.Conv2d(c * 4, num_classes, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(image))
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)


def build_model(model_config: dict[str, Any]) -> nn.Module:
    architecture = model_config["architecture"]
    kwargs = {
        "num_classes": int(model_config["num_classes"]),
        "base_channels": int(model_config["base_channels"]),
    }
    if architecture == "tiny_unet":
        return TinyUNet(**kwargs)
    if architecture == "lite_fcn":
        return LiteFCN(**kwargs)
    raise ValueError(f"Unknown architecture: {architecture}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

# -*- coding: utf-8 -*-
"""
Compact 4-stage multi-scale detection backbone.

IMPORTANT (fidelity note): the paper uses DANet [Fang et al., ACM MM 2023] as
the baseline detector, whose full architecture is NOT described in the UniCD
paper. This module is a lightweight stand-in that follows the properties of
DANet that ARE stated in the paper:

  * multi-scale features from 4 stages (used by the TEBS loss, Eq. 7, and the
    bias-robust loss, Eq. 10, which sum over i = 1..4);
  * dynamic (attention-based) feature perception, a core idea of DANet;
  * a small parameter count so the whole model stays near 0.4 M params.

To reproduce the exact DANet numbers you must plug in the official DANet
code (https://github.com/IVPLaboratory/UniCD does not release it).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel attention (Squeeze-and-Excitation style) for dynamic perception."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channels, max(8, channels // reduction), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, channels // reduction), channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.sigmoid(self.fc(x.mean(dim=(2, 3), keepdim=True)))
        return x * w


class ConvBlock(nn.Module):
    """conv 3x3 + BN + ReLU, optional stride."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class MultiScaleBackbone(nn.Module):
    """
    4-stage backbone producing features F1..F4 at strides 4/8/16/32.

    channels: [32, 64, 128, 256] by default (kept small for real-time use).
    """

    def __init__(self, channels=(16, 32, 64, 128), blocks_per_stage: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBlock(3, channels[0] // 2, stride=2),  # /2
            ConvBlock(channels[0] // 2, channels[0], stride=2),  # /4
        )
        self.stages = nn.ModuleList()
        in_ch = channels[0]
        for i, c in enumerate(channels):
            # first block downsamples by 2 (except stage 0 which already is at /4)
            modules = [ConvBlock(in_ch, c, stride=1 if i == 0 else 2)]
            for _ in range(blocks_per_stage - 1):
                modules.append(ConvBlock(c, c))
            modules.append(ChannelAttention(c))
            stage = nn.Sequential(*modules)
            self.stages.append(stage)
            in_ch = c
        self.out_channels = list(channels)

    def forward(self, x: torch.Tensor) -> list:
        """Returns [F1, F2, F3, F4] at strides 4, 8, 16, 32."""
        f = self.stem(x)
        feats = []
        for stage in self.stages:
            f = stage(f)
            feats.append(f)
        return feats

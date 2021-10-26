import logging
from typing import Optional, Tuple

import torch
from torch import nn, Tensor
from torch.functional import F

logger = logging.getLogger(__name__)


class FeatureExtractor(nn.Module):
    conv_layers: nn.ModuleList

    def __init__(
            self,
            conv_layers: nn.ModuleList,
    ):
        super(FeatureExtractor, self).__init__()
        self.conv_layers = conv_layers

    def forward(
            self,
            x: Tensor,
            length: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if x.ndim != 2:
            raise ValueError(
                f'''Expected the input Tensor to be 2D (batch, time),
                but received {list(x.shape)}'''
            )

        x = x.unsqueeze(1)

        for layer in self.conv_layers:
            x, length = layer(x, length)

        x = x.transpose(1, 2)
        return x, length


class ConvLayerBlock(nn.Module):
    layer_norm: nn.Module
    conv: nn.Module
    kernel_size: int
    stride: int

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            bias: bool,
            layer_norm: Optional[nn.Module]
    ):
        super(ConvLayerBlock, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.layer_norm = layer_norm
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias
        )

    def forward(
            self,
            x: Tensor,
            length: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        x = self.conv(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        x = F.gelu(x)

        if length is not None:
            length = torch.div(length - self.kernel_size, self.stride, rounding_mode='floor') + 1
            length = torch.max(torch.zeros_like(length), length)

        return x, length

from torch import nn, Tensor
from torch.functional import F


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(-2, -1)
        x = F.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps
        )
        x = x.transpose(-2, -1)
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, bias=bias)
        self.layer_norm = LayerNorm(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.layer_norm(x)
        x = F.gelu(x)

        return x


class FeatureExtractor(nn.Module):
    def __init__(self, conv_layers):
        super().__init__()
        self.conv_layers = conv_layers

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x


class Quantizer(nn.Module):
    def forward(self, x):
        x = F.gumbel_softmax(x, hard=True, dim=1)
        return x


class ConvolutionalPositionalEmbedding(nn.Module):
    """Positional embedding which is placed at the beginning of Transformer.
    Args:
        embed_dim (int): Feature dimension of the input Tensor.
        kernel_size (int): The number of frames to be use.
        groups (int): The number of groups in feature dimensions.
    """

    def __init__(
            self,
            embed_dim: int,
            kernel_size: int,
            groups: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
        )
        self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)
        self.num_remove: int = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        """
        Args:
            x (Tensor): shape ``[batch, frame, feature]``.
        Returns:
            Tensor: The resulting feature. Shape ``[batch, frame, feature]``.
        """
        x = x.transpose(-2, -1)
        x = self.conv(x)
        if self.num_remove > 0:
            x = x[..., :-self.num_remove]
        x = F.gelu(x)
        x = x.transpose(-2, -1)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        trf_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.trf_encoder = nn.TransformerEncoder(trf_encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = x.transpose(-2, -1)
        x = self.trf_encoder(x)
        x = x.transpose(-2, -1)
        return x


class Projection(nn.Module):
    def __init__(self, embed_size, output_size):
        super(Projection, self).__init__()
        self.embed_size = embed_size
        self.output_size = output_size
        self.dense = nn.Linear(embed_size * output_size, output_size)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = F.adaptive_avg_pool1d(x, self.output_size)
        x = self.flatten(x)
        x = self.dense(x)
        return x

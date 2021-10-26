from typing import Optional, Tuple

import pytorch_lightning as pl
from torch import nn, Tensor


class TransformerModel(pl.LightningModule):
    feature_extractor: nn.Module
    quantizer: nn.Module
    encoder: nn.Module
    projector: nn.Module
    aux: nn.Module

    def __init__(
            self,
            feature_extractor: nn.Module,
            quantizer: nn.Module,
            encoder: nn.Module,
            projector: nn.Module,
            aux: Optional[nn.Module] = None
    ):
        super(TransformerModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.quantizer = quantizer
        self.encoder = encoder
        self.projector = projector
        self.aux = aux

    def forward(
            self,
            waveforms: Tensor,
            lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        x, lengths = self.feature_extractor(waveforms, lengths)
        x = self.quantizer(x, lengths)
        x = self.encoder(x)
        x = self.projector(x)
        if self.aux is not None:
            x = self.aux(x)

        return x, lengths

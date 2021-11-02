from typing import Any

import pytorch_lightning as pl
from torch import nn


class BaseModel(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1 = self.__create_block(1, 3)
        self.l2 = self.__create_block(3, 16)
        self.l3 = self.__create_block(16, 32)
        self.l4 = self.__create_block(32, 64)

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)

        return out

    @staticmethod
    def __create_block(input_channels, output_channels):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(output_channels),
            nn.Conv2d(output_channels, output_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(output_channels)
        )


class Decoder(nn.Module):
    pass

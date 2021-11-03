from typing import Optional, Tuple, Any

import pytorch_lightning as pl
import torch.optim
from pytorch_lightning.metrics.functional import accuracy
from torch import nn, Tensor


class BaseModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-03, weight_decay=1e-03, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.criterion = nn.NLLLoss()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y_truth = train_batch

        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y_truth)

        pred = torch.argmax(y_pred, dim=1)
        acc = accuracy(pred, y_truth)

        logs = {
            'train_loss': loss,
            'train_acc': acc
        }

        batch_dict = {
            'loss': loss,
            'log': logs,
            'correct': pred.eq(y_truth).sum().item(),
            'total': len(y_truth)
        }

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return batch_dict

    def validation_step(self, val_batch, batch_idx):
        print(val_batch)

        x, y_truth = val_batch

        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y_truth)

        pred = torch.argmax(y_pred, dim=1)
        acc = accuracy(pred, y_truth)

        logs = {
            'train_loss': loss,
            'train_acc': acc
        }

        batch_dict = {
            'loss': loss,
            'log': logs,
            'correct': pred.eq(y_truth).sum().item(),
            'total': len(y_truth),
            'preds': y_pred,
            'target': y_truth
        }

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return batch_dict


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


class CNNModel(BaseModel):

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.sparse_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=15, stride=7, padding=1, bias=False),
            nn.BatchNorm1d(1),
            nn.ReLU(inplace=False),
            nn.Conv1d(in_channels=1, out_channels=3, kernel_size=5, stride=3, padding=1, bias=False),
            nn.BatchNorm1d(3),
            nn.ReLU(inplace=False),
            nn.Conv1d(in_channels=3, out_channels=3, kernel_size=3, stride=3, padding=1, bias=False),
            nn.BatchNorm1d(3),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3)
        )

        self.dense_conv = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
            nn.Conv1d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(6),
            nn.GELU()
        )

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1024),
            nn.Linear(1024, 256),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        out = self.sparse_conv(x)
        out = self.dense_conv(out)
        out = self.fc(out)
        return out

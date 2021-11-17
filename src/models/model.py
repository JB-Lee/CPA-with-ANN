from typing import Optional, Tuple, Any

import pytorch_lightning as pl
import torch.optim
from pytorch_lightning.metrics.functional import accuracy
from torch import nn, Tensor
from torch.nn import functional as F

from . import components


class LayerNorm(nn.LayerNorm):
    def forward(self, input: Tensor) -> Tensor:
        x = input.transpose(-2, -1)
        x = F.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps
        )
        x = x.transpose(-2, -1)
        return x


class BaseModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-03, weight_decay=1e-03, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.criterion = nn.CrossEntropyLoss()

        self.train_log = []
        self.val_log = []
        self.best_epoch = 0
        self.best_val = 10.0

        self.train_loss_log = []
        self.train_acc_log = []
        self.val_loss_log = []
        self.val_acc_log = []

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

    def training_epoch_end(self, outputs) -> None:
        correct = sum([x['correct'] for x in outputs])
        total = sum([x['total'] for x in outputs])
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = correct / total

        val_loss, val_acc = self.val_log[self.current_epoch]

        if val_loss < self.best_val:
            self.best_val = val_loss
            self.best_epoch = self.current_epoch

            self.log('best/val_loss', val_loss, on_epoch=True, prog_bar=True)
            self.log('best/val_acc', val_acc, on_epoch=True, prog_bar=True)

            self.log('best/train_loss', avg_loss, on_epoch=True, prog_bar=True)
            self.log('best/train_acc', avg_acc, on_epoch=True, prog_bar=True)

            self.log('best/epoch', self.current_epoch, on_epoch=True, prog_bar=True)

        self.logger.experiment.add_scalar('Loss/Train', avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar('Accuracy/Train', avg_acc, self.current_epoch)

        self.train_loss_log.append(avg_loss)
        self.train_acc_log.append(avg_acc)

        tb_logs = {'loss': avg_loss, 'Accuracy': avg_acc}

    def validation_epoch_end(self, outputs):
        correct = sum([x['correct'] for x in outputs])
        total = sum([x['total'] for x in outputs])
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = correct / total

        preds = torch.exp(torch.cat([x['preds'] for x in outputs])).cpu()
        targets = torch.cat([x['target'] for x in outputs]).cpu()

        self.val_log.append((avg_loss, avg_acc))

        self.logger.experiment.add_scalar('Loss/Validation', avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar('Accuracy/Validation', avg_acc, self.current_epoch)

        self.val_loss_log.append(avg_loss)
        self.val_acc_log.append(avg_acc)

        tb_logs = {'loss': avg_loss, 'Accuracy': avg_acc}

        return {
            'loss': avg_loss,
            'log': tb_logs
        }

    def get_history(self):
        return {'train_loss': self.train_loss_log,
                'train_acc': self.train_acc_log,
                'val_loss': self.val_loss_log,
                'val_acc': self.val_acc_log}


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
            nn.Conv1d(in_channels=1, out_channels=2, kernel_size=11, stride=4, padding=1, bias=False),
            nn.BatchNorm1d(2),
            nn.ReLU(inplace=False),

            nn.Conv1d(in_channels=2, out_channels=4, kernel_size=11, stride=4, padding=1, bias=False),
            nn.BatchNorm1d(4),
            nn.ReLU(inplace=False),

            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=11, stride=4, padding=1, bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=False),

            # nn.Conv1d(in_channels=8, out_channels=16, kernel_size=11, stride=4, padding=1, bias=False),
            # nn.BatchNorm1d(16),
            # nn.ReLU(inplace=False),

            nn.Dropout(0.1)
        )

        self.dense_conv = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),

            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),

            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GELU(),
            # nn.BatchNorm1d(16),

        )

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1024),
            nn.Flatten(),
            nn.Linear(1024 * 16, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256)
        )

    def forward(self, x):
        out = self.sparse_conv(x)
        out = self.dense_conv(out)
        out = self.fc(out)
        return out


class TestCNNModel(BaseModel):

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.sparse_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=2, kernel_size=11, stride=4, padding=1, bias=False),
            nn.BatchNorm1d(2),
            nn.ReLU(inplace=False),

            nn.Conv1d(in_channels=2, out_channels=4, kernel_size=11, stride=4, padding=1, bias=False),
            nn.BatchNorm1d(4),
            nn.ReLU(inplace=False),

            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=False),

            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=False),

            nn.Conv1d(in_channels=16, out_channels=64, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=False),

            nn.Conv1d(in_channels=64, out_channels=256, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=False),

            nn.Dropout(0.1)
        )

        self.dense_conv = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),

            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),

            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),

        )

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

    def forward(self, x):
        out = self.sparse_conv(x)
        out = self.dense_conv(out)
        out = self.gap(out)
        return out


class TestTransformerModel(BaseModel):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.sparse_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=2, kernel_size=11, stride=4, padding=1, bias=False),
            nn.BatchNorm1d(2),
            nn.ReLU(inplace=False),

            nn.Conv1d(in_channels=2, out_channels=4, kernel_size=11, stride=4, padding=1, bias=False),
            nn.BatchNorm1d(4),
            nn.ReLU(inplace=False),

            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=11, stride=4, padding=1, bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=False),

            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=False),

            nn.Dropout(0.1)
        )

        self.feature_projection = nn.Sequential(
            nn.AdaptiveAvgPool1d(16),
            nn.Linear(16, 16),
        )

        self.CPE = components.ConvolutionalPositionalEmbedding(embed_dim=16, kernel_size=3, groups=1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=16, nhead=8)
        self.TRF = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.FP = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 16, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )

    def forward(self, x) -> Any:
        out = self.sparse_conv(x)

        out = self.feature_projection(out)

        out = self.CPE(out)

        out = self.TRF(out)
        out = self.FP(out)

        return out


class TestModel(BaseModel):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        blocks = [components.ConvLayer(1, 8, 11, 5, True), components.ConvLayer(8, 16, 5, 3, True),
                  components.ConvLayer(16, 16, 5, 3, True), components.ConvLayer(16, 32, 3, 2, True),
                  components.ConvLayer(32, 256, 3, 2, True)]

        self.feature_extractor = components.FeatureExtractor(nn.ModuleList(blocks))
        # self.quantizer = components.Quantizer()
        self.dropout = nn.Dropout(0.1)
        self.pe = components.ConvolutionalPositionalEmbedding(embed_dim=256, kernel_size=3, groups=1)
        self.encoder = components.Encoder(d_model=256, nhead=8, num_layers=2)
        # self.down_sample = components.ConvLayer(256, 128, 1, 1, False)
        self.feature_projector = components.Projection(embed_size=256, output_size=256)

    def forward(self, x) -> Any:
        x = self.feature_extractor(x)
        # x = self.quantizer(x)
        x = self.dropout(x)
        x = self.pe(x)
        x = self.encoder(x)
        # x = self.dropout(x)
        # x = self.down_sample(x)
        x = self.feature_projector(x)

        return x


class FE(nn.Module):
    def __init__(self):
        super(FE, self).__init__()

        blocks = [components.ConvLayer(1, 2, 11, 5, True), components.ConvLayer(2, 4, 11, 5, True),
                  components.ConvLayer(4, 8, 5, 3, True), components.ConvLayer(8, 16, 5, 3, True),
                  components.ConvLayer(16, 32, 3, 2, True)]

        self.feature_extractor = components.FeatureExtractor(nn.ModuleList(blocks))
        # self.quantizer = components.Quantizer()
        # self.dense = nn.Linear(32, 32)

    def forward(self, x):
        x = self.feature_extractor(x)
        # x = self.quantizer(x)
        # x = F.adaptive_avg_pool1d(x, 32)
        # x = self.dense(x)
        return x


class CodebookModel(nn.Module):
    def __init__(self):
        super(CodebookModel, self).__init__()
        self.projector = components.Projection(embed_size=32, output_size=256)

    def forward(self, x):
        x = self.projector(x)
        return x


class TRFClassifier(nn.Module):
    def __init__(self):
        super(TRFClassifier, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.pe = components.ConvolutionalPositionalEmbedding(embed_dim=32, kernel_size=3, groups=1)
        self.encoder = components.Encoder(d_model=32, nhead=8, num_layers=2)
        # self.down_sample = components.ConvLayer(256, 128, 1, 1, False)
        self.feature_projector = components.Projection(embed_size=32, output_size=256)

    def forward(self, x):
        x = self.dropout(x)
        x = self.pe(x)
        x = self.encoder(x)
        x = self.feature_projector(x)

        return x


class TestModel2(BaseModel):
    def __init__(self, alpha=2, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.fe = FE()
        self.cb = CodebookModel()
        self.trf = TRFClassifier()

        self.alpha = alpha

    def forward(self, x) -> Any:
        x = self.fe(x)
        x = self.trf(x)
        return x

    def training_step(self, train_batch, batch_idx):
        x, y_truth = train_batch

        q = self.fe(x)

        y_pred = self.trf(q)
        y_cb = self.cb(q)

        loss_pred = self.criterion(y_pred, y_truth)
        loss_cb = self.criterion(y_cb, y_truth)

        loss = self.alpha * loss_pred + loss_cb

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

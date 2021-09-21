from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import seaborn as sb

from olsen.metrics.metrics import ClassificationMetrics
from olsen.module.classifier import Classifier
from olsen.dataunit.labelblock import LabelBlock
from olsen.module.transformer import BertSentenceEncoder
from olsen.module.wrapper import AttnWrapper
from olsen.module.activation import Activation


class BertSlider(pl.LightningModule):
    def __init__(self, transformer: str, embedding_method: str,
                 window_size: int, dilation_gap: int, num_classes: int, learning_rate: float,
                 labels: List[str] = None):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.labels = labels

        # models
        self.encoder = BertSentenceEncoder(embedding_method=embedding_method,
                                           transformer=transformer)
        self.wrapper = AttnWrapper(model=self.encoder,
                                   embed_dim=self.encoder.get_dimension(),
                                   window_size=window_size,
                                   dilation_gap=dilation_gap)
        self.activation = Activation(encoding_dim=self.wrapper.get_dimension())
        self.classifier = Classifier(encoding_dim=self.wrapper.get_dimension(),
                                     num_classes=num_classes)
        self.model = nn.Sequential(self.wrapper, self.activation, self.classifier)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.train_metrics = ClassificationMetrics(self.num_classes, "train")
        self.val_metrics = ClassificationMetrics(self.num_classes, "val")
        self.test_metrics = ClassificationMetrics(self.num_classes, "test")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = list(zip(*batch))
        logits = self.forward(x)
        label_tensor = self.label2tensor(y).type_as(logits).long()
        loss = self.cross_entropy_loss(logits, label_tensor)
        self.train_metrics(logits, label_tensor)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = list(zip(*val_batch))
        logits = self.forward(x)
        label_tensor = self.label2tensor(y).type_as(logits).long()
        loss = self.cross_entropy_loss(logits, label_tensor)
        self.val_metrics(logits, label_tensor)
        self.log("val_loss", loss)
        self.log_scalars(self.val_metrics)

    def validation_epoch_end(self, outputs):
        self.log_nonscalars(self.val_metrics)

    def test_step(self, test_batch, batch_idx):
        x, y = list(zip(*test_batch))
        logits = self.forward(x)
        label_tensor = self.label2tensor(y).type_as(logits).long()
        loss = self.cross_entropy_loss(logits, label_tensor)
        self.test_metrics(logits, label_tensor)
        self.log("test_loss", loss)
        self.log_scalars(self.test_metrics)

    def test_epoch_end(self, outputs):
        self.log_nonscalars(self.test_metrics)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        logits = self.forward(batch)
        normalized_probs = self.softmax(logits)
        _, pred_indices = torch.topk(normalized_probs, k=1, dim=1)
        pred_indices = pred_indices.squeeze(1).tolist()
        labels = LabelBlock.to_labels(pred_indices, self.labels)
        return labels

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def label2tensor(self, labels: List[LabelBlock]) -> torch.Tensor:
        return torch.cat([label.idxs for label in labels])

    def log_scalars(self, metric: ClassificationMetrics):
        self.log(f"{metric.mode}_precision_macro", metric.macro_pc)
        self.log(f"{metric.mode}_precision_micro", metric.micro_pc)
        self.log(f"{metric.mode}_precision_weighted", metric.weigh_pc)

        self.log(f"{metric.mode}_recall_macro", metric.macro_rc)
        self.log(f"{metric.mode}_recall_micro", metric.micro_rc)
        self.log(f"{metric.mode}_recall_weighted", metric.weigh_rc)

        self.log(f"{metric.mode}_f1_macro", metric.macro_f1)
        self.log(f"{metric.mode}_f1_micro", metric.micro_f1)
        self.log(f"{metric.mode}_f1_weighted", metric.weigh_f1)

        self.log(f"{metric.mode}_mcc", metric.mcc)

    def log_nonscalars(self, metric: ClassificationMetrics):
        fig = plt.figure(figsize=(24, 24))
        cf_matrix = metric.cnfs_mat.compute().cpu().numpy()
        sb.heatmap(cf_matrix, annot=True, xticklabels=self.labels, yticklabels=self.labels, fmt='.1%')
        self.logger.experiment.add_figure(f"{metric.mode}_cnfs_mat", fig, global_step=self.current_epoch)

        categ_pc = metric.categ_pc.compute().cpu().tolist()
        pc_map = dict(zip(self.labels, categ_pc))
        self.logger.experiment.add_scalars(f"{metric.mode}_precision_categ", pc_map, global_step=self.current_epoch)

        categ_rc = metric.categ_pc.compute().cpu().tolist()
        rc_map = dict(zip(self.labels, categ_rc))
        self.logger.experiment.add_scalars(f"{metric.mode}_recall_categ", rc_map, global_step=self.current_epoch)

        categ_f1 = metric.categ_pc.compute().cpu().tolist()
        f1_map = dict(zip(self.labels, categ_f1))
        self.logger.experiment.add_scalars(f"{metric.mode}_f1_categ", f1_map, global_step=self.current_epoch)


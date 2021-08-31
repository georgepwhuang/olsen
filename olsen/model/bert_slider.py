from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn

from olsen.metrics.metrics import ClassificationMetrics
from olsen.module.classifier import Classifier
from olsen.data.labelblock import LabelBlock
from olsen.module.tokenizer import SentenceTokenizer
from olsen.module.transformer import BertSentenceEncoder
from olsen.module.wrapper import AttnWrapper


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
        self.tokenizer = SentenceTokenizer(transformer=transformer)
        self.encoder = BertSentenceEncoder(embedding_method=embedding_method,
                                           transformer=transformer)
        self.litemod = nn.Sequential(self.tokenizer, self.encoder)
        self.wrapper = AttnWrapper(model=self.litemod,
                                   embed_dim=self.encoder.get_dimension(),
                                   window_size=window_size,
                                   dilation_gap=dilation_gap)
        self.classifier = Classifier(encoding_dim=self.wrapper.embed_dim,
                                     num_classes=num_classes)
        self.model = nn.Sequential(self.wrapper, self.classifier)
        self.loss = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.train_metrics = ClassificationMetrics(self.num_classes)
        self.val_metrics = ClassificationMetrics(self.num_classes)
        self.test_metrics = ClassificationMetrics(self.num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = list(zip(*batch))
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log("train_loss", loss)
        self.log_metrics("train", self.train_metrics)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = list(zip(*val_batch))
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log("val_loss", loss)
        self.log_metrics("val", self.val_metrics)

    def test_step(self, test_batch, batch_idx):
        x, y = list(zip(*test_batch))
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log("test_loss", loss)
        self.log_metrics("test", self.test_metrics)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        logits = self.forward(batch)
        normalized_probs = self.softmax(logits)
        _, pred_indices = torch.topk(normalized_probs, k=1, dim=1)
        pred_indices = pred_indices.squeeze(1).tolist()
        labels = LabelBlock.from_indices(pred_indices, self.labels)
        return labels.labels

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def log_metrics(self, mode: str, metric: ClassificationMetrics):
        self.log(f"{mode}_precision", metric.categ_pc)
        self.log(f"{mode}_precision_marco", metric.macro_pc)
        self.log(f"{mode}_precision_micro", metric.micro_pc)
        self.log(f"{mode}_precision_weighted", metric.weigh_pc)
        self.log(f"{mode}_recall", metric.categ_rc)
        self.log(f"{mode}_recall_marco", metric.macro_rc)
        self.log(f"{mode}_recall_micro", metric.micro_rc)
        self.log(f"{mode}_recall_weighted", metric.weigh_rc)
        self.log(f"{mode}_f1", metric.categ_f1)
        self.log(f"{mode}_f1_marco", metric.macro_f1)
        self.log(f"{mode}_f1_micro", metric.micro_f1)
        self.log(f"{mode}_f1_weighted", metric.weigh_f1)
        if mode != "train":
            self.log(f"{mode}_confusion_mat", metric.cnfs_mat)
            self.log(f"{mode}_mcc", metric.mcc)

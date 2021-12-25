from typing import List

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sb

from olsen.model.bert_slider import BertSlider
from olsen.util.uda_util import get_tsa_thresh
from olsen.metrics.metrics import ClassificationMetrics


class BertSliderWithUDA(BertSlider):
    def __init__(self, transformer: str, embedding_method: str,
                 window_size: int, dilation_gap: int, num_classes: int, learning_rate: float,
                 labels: List[str] = None, unsup_coeff: float = 1.0, tsa_schedule: str = "exp_schedule",
                 uda_confidence_thresh: float = 0.8, uda_softmax_temp: float = 0.4):
        super().__init__(transformer, embedding_method, window_size, dilation_gap, num_classes, learning_rate, labels)
        self.sup_loss_funct = nn.CrossEntropyLoss(reduction="none")
        self.unsup_loss_funct = nn.KLDivLoss(reduction="none")
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.unsup_coeff = unsup_coeff
        self.tsa_schedule = tsa_schedule
        self.uda_confidence_thresh = uda_confidence_thresh
        self.uda_softmax_temp = uda_softmax_temp
        self.save_hyperparameters()

        self.deploy = nn.Sequential(self.activation, self.classifier)

    def forward(self, x):
        x = torch.utils.checkpoint.checkpoint(self.wrapper, x)
        return self.deploy(x)

    def on_train_start(self) -> None:
        self.period = self.trainer.datamodule.period
        self.cur_step = 0

        if self.trainer.max_steps != -1:
            self.total_steps = self.trainer.max_steps
        else:
            limit_batches = self.trainer.limit_train_batches
            batches = len(self.trainer.datamodule.train_dataloader())
            batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)
            effective_accum = self.period

            self.total_steps = (batches // effective_accum) * self.trainer.max_epochs

    def training_step(self, batch, batch_idx):
        if batch_idx % self.period == 0:
            text, label = list(zip(*batch))
            logits = self.forward(text)
            label_tensor = self.concat_label_tensor(label).type_as(logits).long()
            sup_loss = self.sup_loss_funct(logits, label_tensor)

            # training signal annealing
            tsa_thresh = get_tsa_thresh(self.tsa_schedule, self.cur_step, self.total_steps, self.num_classes)
            self.cur_step += 1
            loss_mask = torch.lt(torch.exp(-sup_loss), tsa_thresh)
            sup_loss = torch.div(torch.sum(torch.mul(sup_loss, loss_mask), dim=-1),
                                 torch.clamp(torch.sum(loss_mask, dim=-1), min=1))
            self.log("train_cross_entropy_loss", sup_loss)
            return torch.mul(sup_loss, self.period)

        else:
            original, augmented = list(zip(*batch))
            with torch.no_grad():
                original_tensor = self.forward(original)
                original_prob = self.softmax(original_tensor)

                # confidence-based masking
                unsup_loss_mask = torch.gt(torch.max(original_prob, dim=-1)[0], self.uda_confidence_thresh)

            augmented_tensor = self.forward(augmented)
            # sharpening predictions
            augmented_prob = self.log_softmax(torch.div(augmented_tensor, self.uda_softmax_temp))

            unsup_loss = self.unsup_loss_funct(augmented_prob, original_prob)
            unsup_loss = torch.sum(unsup_loss, dim=-1)
            unsup_loss = torch.div(torch.sum(torch.mul(unsup_loss, unsup_loss_mask), dim=-1),
                                   torch.clamp(torch.sum(unsup_loss_mask, dim=-1), min=1))
            self.log("train_consistency_loss", unsup_loss)
            return torch.mul(unsup_loss, self.unsup_coeff * self.period / (self.period - 1))

    def validation_step(self, val_batch, batch_idx):
        x, y = list(zip(*val_batch))
        logits = self.forward(x)
        label_tensor = self.concat_label_tensor(y).type_as(logits).long()
        loss = self.cross_entropy_loss(logits, label_tensor)
        self.val_metrics(logits, label_tensor)
        self.log("val_loss", loss, sync_dist=True)
        self.log_scalars(self.val_metrics)

    def test_step(self, test_batch, batch_idx):
        x, y = list(zip(*test_batch))
        logits = self.forward(x)
        label_tensor = self.concat_label_tensor(y).type_as(logits).long()
        loss = self.cross_entropy_loss(logits, label_tensor)
        self.test_metrics(logits, label_tensor)
        self.log("test_loss", loss, sync_dist=True)
        self.log_scalars(self.test_metrics)


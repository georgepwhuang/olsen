from typing import List

import torch
import torch.nn as nn

from olsen.model.bert_slider import BertSlider
from olsen.util.uda_util import get_tsa_thresh


class BertSliderWithTSA(BertSlider):
    def __init__(self, transformer: str, embedding_method: str,
                 window_size: int, dilation_gap: int, num_classes: int, learning_rate: float,
                 labels: List[str] = None, tsa_schedule: str = "linear_schedule"):
        super().__init__(transformer, embedding_method, window_size, dilation_gap, num_classes, learning_rate, labels)
        self.sup_loss_funct = nn.CrossEntropyLoss(reduction="none")
        self.tsa_schedule = tsa_schedule
        self.save_hyperparameters()

    def setup(self, stage=None) -> None:
        super().setup(stage)
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/5449
        if stage == 'fit':
            if self.trainer.max_steps:
                self.total_steps = self.trainer.max_steps
            else:
                limit_batches = self.trainer.limit_train_batches
                batches = len(self.train_dataloader())
                batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(
                    limit_batches * batches)

                num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
                if self.trainer.tpu_cores:
                    num_devices = max(num_devices, self.trainer.tpu_cores)

                effective_accum = self.trainer.accumulate_grad_batches * num_devices
                self.total_steps = (batches // effective_accum) * self.trainer.max_epochs

    def training_step(self, batch, batch_idx):
        text, label = list(zip(*batch))
        logits = self.model(text)
        label_tensor = self.concat_label_tensor(label).type_as(logits).long()
        sup_loss = self.sup_loss_funct(logits, label_tensor)

        # training signal annealing
        tsa_thresh = get_tsa_thresh(self.tsa_schedule, self.global_step, self.total_steps, self.num_classes)
        larger_than_threshold = torch.exp(-sup_loss) > tsa_thresh
        loss_mask = torch.ones_like(label_tensor, dtype=torch.float32) * (1 - larger_than_threshold.type(torch.float32))
        sup_loss = torch.sum(sup_loss * loss_mask, dim=-1) / torch.max(torch.sum(loss_mask, dim=-1), torch.tensor(1.))

        self.log("train_loss", sup_loss)

        return sup_loss

from typing import List

import torch
import torch.nn as nn

from olsen.model.bert_slider import BertSlider
from olsen.util.uda_util import get_tsa_thresh


class BertSliderWithSupervisedUDA(BertSlider):
    def __init__(self, transformer: str, embedding_method: str,
                 window_size: int, dilation_gap: int, num_classes: int, learning_rate: float,
                 labels: List[str] = None, unsup_coeff: int = 1, tsa_schedule: str = "exp_schedule",
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
        text, aug, label = list(zip(*batch))
        merged_text = text + aug
        logits = self.model(merged_text)
        length = logits.size()[0]
        assert length % 2 == 0, "Merged tensor error"
        assert length == 2 * sum([block.distribute[1] for block in text])
        original_tensor, augmented_tensor = torch.split(logits, int(length / 2))
        label_tensor = self.concat_label_tensor(label).type_as(logits).long()
        sup_loss = self.sup_loss_funct(original_tensor, label_tensor)

        # training signal annealing
        tsa_thresh = get_tsa_thresh(self.tsa_schedule, self.global_step, self.total_steps, self.num_classes)
        larger_than_threshold = torch.exp(-sup_loss) > tsa_thresh
        loss_mask = torch.ones_like(label_tensor, dtype=torch.float32) * (1 - larger_than_threshold.type(torch.float32))
        sup_loss = torch.sum(sup_loss * loss_mask, dim=-1) / torch.max(torch.sum(loss_mask, dim=-1), torch.tensor(1.))

        original_prob = self.softmax(original_tensor)

        # confidence-based masking
        unsup_loss_mask = torch.max(original_prob, dim=-1)[0] > self.uda_confidence_thresh
        unsup_loss_mask = unsup_loss_mask.type(torch.float32)

        # sharpening predictions
        augmented_prob = self.log_softmax(augmented_tensor / self.uda_softmax_temp)

        unsup_loss = self.unsup_loss_funct(augmented_prob, original_prob)
        unsup_loss = torch.sum(unsup_loss, dim=-1)
        unsup_loss = torch.sum(unsup_loss * unsup_loss_mask, dim=-1) / torch.max(torch.sum(unsup_loss_mask, dim=-1),
                                                                                 torch.tensor(1.))

        total_loss = sup_loss + self.unsup_coeff * unsup_loss

        self.log("train_cross_entropy_loss", sup_loss)
        self.log("train_consistency_loss", unsup_loss)
        self.log("train_loss", total_loss)

        return total_loss

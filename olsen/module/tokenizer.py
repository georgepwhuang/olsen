from typing import List

import torch
import torch.nn as nn
from transformers import AutoTokenizer


class SentenceTokenizer(nn.Module):
    def __init__(
            self,
            transformer: str,
    ):
        super(SentenceTokenizer, self).__init__()
        self.transformer = transformer
        self.tokenizer = AutoTokenizer.from_pretrained(self.transformer,
                                                       use_fast=False,
                                                       model_max_length=512,
                                                       do_basic_tokenize=False)
        self.dt = {
            self.tokenizer.bos_token: "bos",
            self.tokenizer.eos_token: "eos",
            self.tokenizer.cls_token: "cls",
            self.tokenizer.sep_token: "sep",
            self.tokenizer.mask_token: "mask",
            self.tokenizer.unk_token: "unk",
        }

    def forward(self, lines: List[str]) -> torch.Tensor:
        for line in lines:
            for key, value in self.dt.items():
                if bool(key):
                    line = line.replace(key, value)
        return self.tokenizer(lines, padding=True, truncation=True, return_tensors="pt")

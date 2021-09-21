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
        self.tokenizer = AutoTokenizer.from_pretrained(self.transformer)
        self.dt = {
            self.tokenizer.bos_token: "bos",
            self.tokenizer.eos_token: "eos",
            self.tokenizer.cls_token: "cls",
            self.tokenizer.sep_token: "sep",
            self.tokenizer.mask_token: "mask",
            self.tokenizer.unk_token: "unk",
        }

    def forward(self, lines: List[str]) -> torch.Tensor:
        new_lines = []
        for line in lines:
            new_line = line
            for key, value in self.dt.items():
                if bool(key):
                    new_line = new_line.replace(key, value)
            new_lines.append(new_line)
        return self.tokenizer(new_lines, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

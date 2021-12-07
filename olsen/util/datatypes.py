from collections import UserDict
from typing import List

import torch
from transformers.tokenization_utils_base import BatchEncoding


class EncodingDict(UserDict):
    def __init__(self, data: dict):
        super().__init__(data)

    def __getattr__(self, item: str):
        try:
            return self.data[item]
        except KeyError:
            raise AttributeError

    @staticmethod
    def merge(data: List[BatchEncoding]):
        input_ids = [encoding.input_ids for encoding in data]
        input_ids = torch.cat(input_ids, dim=0)
        attention_mask = [encoding.attention_mask for encoding in data]
        attention_mask = torch.cat(attention_mask, dim=0)
        dt = {"input_ids": input_ids, "attention_mask": attention_mask}
        return EncodingDict(dt)

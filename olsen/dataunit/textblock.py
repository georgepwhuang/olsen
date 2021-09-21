from typing import List
from torch import nn
import torch


class TextBlock:
    def __init__(
            self, lines: List[str], begin: List[str], end: List[str], overlap: int, tokenizer: nn.Module
    ):
        assert len(lines) >= overlap, f"Main length is {len(lines)} !>= overlap"
        begin_ext = [""] * (overlap - len(begin))
        begin_ext.extend(begin)
        end_ext = end
        end_ext.extend([""] * (overlap - len(end)))
        self.begin = tokenizer(begin_ext)
        self.main = tokenizer(lines)
        self.end = tokenizer(end_ext)
        assert len(begin_ext) == overlap, f"Begin length is {len(begin_ext)} != overlap"
        assert len(end_ext) == overlap, f"End length is {len(end_ext)} != overlap"
        self.distribute = [len(begin_ext), len(lines), len(end_ext)]

    def to(self, device: torch.device):
        self.begin.to(device)
        self.end.to(device)
        self.main.to(device)

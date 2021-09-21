from typing import Any, Dict

import torch
import torch.nn as nn


class Activation(nn.Module):
    def __init__(
            self,
            encoding_dim: int,
    ):
        super(Activation, self).__init__()
        self.encoding_dim = encoding_dim
        self.dense = nn.Linear(self.encoding_dim, self.encoding_dim)
        self.activation = nn.Tanh()

    def forward(self, inputs: torch.Tensor) -> Dict[str, Any]:
        encoding = self.dense(inputs)
        output = self.activation(encoding)
        return output

import itertools
from typing import List

import torch
import torch.nn as nn

from olsen.dataunit.textblock import TextBlock
from olsen.util.datatypes import EncodingDict


class AttnWrapper(nn.Module):
    def __init__(
            self,
            model: nn.Module,
            embed_dim: int,
            window_size: int,
            dilation_gap: int,
    ):
        super(AttnWrapper, self).__init__()
        self.model = model
        self.embed_dim = embed_dim
        self.attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=8, batch_first=True)
        self.window_size = int((window_size - 1) / 2.0)
        self.dilation_gap = dilation_gap + 1
        self.overlap = self.window_size * self.dilation_gap

    def forward(self, text: List[TextBlock]):
        input_list = [[block.begin, block.main, block.end] for block in text]
        flattened_list = list(itertools.chain.from_iterable(input_list))
        input_encoding = EncodingDict.merge(flattened_list)

        size = [block.distribute for block in text]
        all_size = [sum(block.distribute) for block in text]

        output_encoding = self.model(input_encoding)

        input_by_item = torch.split(output_encoding, all_size)
        output_list = [torch.split(input_by_item[i], size[i]) for i in range(len(size))]

        encodings = []
        for begin_encoding, main_encoding, end_encoding in output_list:
            key_list = []
            for i in range(-self.overlap, 0, self.dilation_gap):
                encoding = torch.cat([begin_encoding[i:], main_encoding[:i]], dim=0)
                key_list.append(encoding)
            for i in range(self.dilation_gap, self.overlap + self.dilation_gap, self.dilation_gap):
                encoding = torch.cat([main_encoding[i:], end_encoding[:i]], dim=0)
                key_list.append(encoding)
            key = torch.stack(key_list, dim=1)
            query = main_encoding.unsqueeze(1)
            attn = self.attention(query=query, key=key, value=key)[0]
            attn_encoding = attn.squeeze(1)
            encoding = torch.cat([main_encoding, attn_encoding], dim=-1)
            encodings.append(encoding)
        final_encoding = torch.cat(encodings, dim=0)
        return final_encoding

    def get_dimension(self):
        return self.embed_dim * 2

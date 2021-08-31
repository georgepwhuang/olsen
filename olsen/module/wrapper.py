import itertools
from typing import List

import torch
import torch.nn as nn

from olsen.data.textblock import TextBlock


class AttnWrapper(nn.Module):
    def __init__(
            self,
            model: nn.Sequential,
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

    def forward(self, lines: List[TextBlock]):
        main_size = [len(doc.lines) for doc in lines]
        begin_size = [len(doc.begin) for doc in lines]
        end_size = [len(doc.end) for doc in lines]

        main_lines = [doc.lines for doc in lines]
        begin_lines = [doc.begin for doc in lines]
        end_lines = [doc.end for doc in lines]

        main_lines = list(itertools.chain.from_iterable(main_lines))
        begin_lines = list(itertools.chain.from_iterable(begin_lines))
        end_lines = list(itertools.chain.from_iterable(end_lines))

        all_size = [len(main_lines), len(begin_lines), len(end_lines)]
        all_lines = list(itertools.chain.from_iterable([main_lines, begin_lines, end_lines]))

        all_encodings = self.model(all_lines)

        main_encodings, begin_encodings, end_encodings = torch.split(all_encodings, all_size)

        main_encodings_list = torch.split(main_encodings, main_size)
        begin_encodings_list = torch.split(begin_encodings, begin_size)
        end_encodings_list = torch.split(end_encodings, end_size)

        encodings = []
        for main_encoding, begin_encoding, end_encoding in \
                zip(main_encodings_list, begin_encodings_list, end_encodings_list):
            key_list = []
            for i in range(-self.overlap, 0, self.dilation_gap):
                encoding = torch.cat([begin_encoding[i:], main_encoding[:i]], dim=0)
                key_list.append(encoding)
            for i in range(self.dilation_gap, self.overlap + self.dilation_gap, self.dilation_gap):
                encoding = torch.cat([main_encoding[i:], end_encoding[:i]], dim=0)
                key_list.append(encoding)
            key = torch.stack(key_list, dim=-2)
            query = main_encoding.unsqueeze(1)
            attn = self.attention(query=query, key=key, value=key)
            attn_encoding = attn.squeeze(-2)
            encoding = torch.cat([main_encoding, attn_encoding], dim=-1)
            encodings.append(encoding)
        final_encoding = torch.cat(encodings, dim=0)
        return final_encoding

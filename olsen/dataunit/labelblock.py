from typing import List
import torch


class LabelBlock:
    def __init__(self, labels: List[str], label_map: List[str]):
        dt = {k: v for v, k in enumerate(label_map)}
        idxs = []
        for label in labels:
            assert label in dt.keys(), f"{label} is not a valid label"
            idxs.append(dt[label])
        self.idxs = torch.tensor(idxs)

    def to(self, device: torch.device):
        self.idxs.to(device)

    @staticmethod
    def to_labels(idxs: List[int], label_map: List[str]):
        return [label_map[idx] for idx in idxs]

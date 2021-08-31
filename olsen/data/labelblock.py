from typing import List


class LabelBlock:
    def __init__(self, idxs: List[int], labels: List[str], label_map: List[str]):
        self.idxs = idxs
        self.label_map = label_map
        self.labels = labels

    @classmethod
    def from_labels(cls, labels: List[str], label_map: List[str]):
        dt = {k: v for v, k in enumerate(label_map)}
        idxs = [dt[label] for label in labels]
        return cls(idxs=idxs, labels=labels, label_map=label_map)

    @classmethod
    def from_indices(cls, idxs: List[int], label_map: List[str]):
        labels = [label_map[idx] for idx in idxs]
        return cls(idxs=idxs, labels=labels, label_map=label_map)

import torch
import torchmetrics as tm
from torch import nn


class ClassificationMetrics(nn.Module):
    def __init__(self, num_classes: int, mode: str):
        super().__init__()
        self.num_classes = num_classes
        self.mode = mode

        self.categ_pc = tm.Precision(num_classes=self.num_classes, average="none")
        self.macro_pc = tm.Precision(num_classes=self.num_classes, average="macro")
        self.micro_pc = tm.Precision(num_classes=self.num_classes, average="micro")
        self.weigh_pc = tm.Precision(num_classes=self.num_classes, average="weighted")

        self.categ_rc = tm.Recall(num_classes=self.num_classes, average="none")
        self.macro_rc = tm.Recall(num_classes=self.num_classes, average="macro")
        self.micro_rc = tm.Recall(num_classes=self.num_classes, average="micro")
        self.weigh_rc = tm.Recall(num_classes=self.num_classes, average="weighted")

        self.categ_f1 = tm.F1(num_classes=self.num_classes, average="none")
        self.macro_f1 = tm.F1(num_classes=self.num_classes, average="macro")
        self.micro_f1 = tm.F1(num_classes=self.num_classes, average="micro")
        self.weigh_f1 = tm.F1(num_classes=self.num_classes, average="weighted")

        self.cnfs_mat = tm.ConfusionMatrix(num_classes=self.num_classes, normalize="true")

        self.mcc = tm.MatthewsCorrcoef(num_classes=self.num_classes)

    def to(self, device: torch.device):
        self.categ_pc.to(device)
        self.macro_pc.to(device)
        self.micro_pc.to(device)
        self.weigh_pc.to(device)

        self.categ_rc.to(device)
        self.macro_rc.to(device)
        self.micro_rc.to(device)
        self.weigh_rc.to(device)

        self.categ_f1.to(device)
        self.macro_f1.to(device)
        self.micro_f1.to(device)
        self.weigh_f1.to(device)

        self.cnfs_mat.to(device)

        self.mcc.to(device)

    def forward(self, x, y):
        self.categ_pc(x, y)
        self.macro_pc(x, y)
        self.micro_pc(x, y)
        self.weigh_pc(x, y)

        self.categ_rc(x, y)
        self.macro_rc(x, y)
        self.micro_rc(x, y)
        self.weigh_rc(x, y)

        self.categ_f1(x, y)
        self.macro_f1(x, y)
        self.micro_f1(x, y)
        self.weigh_f1(x, y)

        self.cnfs_mat(x, y)
        self.mcc(x, y)

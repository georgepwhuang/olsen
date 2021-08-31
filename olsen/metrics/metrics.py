import torchmetrics as tm


class ClassificationMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes

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
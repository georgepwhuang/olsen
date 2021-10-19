from typing import Optional, List

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from olsen.datamod.dataset import DocumentClassificationDataset, DocumentInferenceDataset, UnlabelledDocumentDataset, \
    AugmentedDocumentClassificationDataset


class DocumentClassificationData(pl.LightningDataModule):
    def __init__(
            self,
            inner_batch_size: int,
            batch_size: int,
            window_size: int,
            dilation_gap: int,
            labels: List[str],
            transformer: str,
            train_filename: Optional[str] = None,
            dev_filename: Optional[str] = None,
            test_filename: Optional[str] = None,
            pred_filenames: Optional[List[str]] = None,
    ):
        super().__init__()
        self.train_filename = train_filename
        self.dev_filename = dev_filename
        self.test_filename = test_filename
        self.pred_filenames = pred_filenames
        self.inner_batch_size = inner_batch_size
        self.batch_size = batch_size
        self.window_size = window_size
        self.dilation_gap = dilation_gap
        self.labels = labels
        self.transformer = transformer

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit' or stage is None:
            self.train_dataset = DocumentClassificationDataset(filename=self.train_filename,
                                                               batch_size=self.inner_batch_size,
                                                               window_size=self.window_size,
                                                               dilation_gap=self.dilation_gap,
                                                               defined_labels=self.labels,
                                                               transformer=self.transformer)
            self.dev_dataset = DocumentClassificationDataset(filename=self.dev_filename,
                                                             batch_size=self.inner_batch_size,
                                                             window_size=self.window_size,
                                                             dilation_gap=self.dilation_gap,
                                                             defined_labels=self.labels,
                                                             transformer=self.transformer)
        if stage == 'test' or stage is None:
            self.test_dataset = DocumentClassificationDataset(filename=self.test_filename,
                                                              batch_size=self.inner_batch_size,
                                                              window_size=self.window_size,
                                                              dilation_gap=self.dilation_gap,
                                                              defined_labels=self.labels,
                                                              transformer=self.transformer)
        if stage == 'predict':
            datasets = []
            for pred_filename in self.pred_filenames:
                dataset = DocumentInferenceDataset(filename=pred_filename,
                                                   batch_size=self.inner_batch_size,
                                                   window_size=self.window_size,
                                                   dilation_gap=self.dilation_gap,
                                                   transformer=self.transformer)
                datasets.append(dataset)
            self.pred_datasets = datasets

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, collate_fn=list, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.dev_dataset, collate_fn=list, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset, collate_fn=list, batch_size=self.batch_size)

    def predict_dataloader(self):
        return [DataLoader(dataset=pred_dataset, collate_fn=list, batch_size=self.batch_size) for pred_dataset
                in self.pred_datasets]



class DocumentClassificationDataWithSupervisedUDA(DocumentClassificationData):
    def __init__(
            self,
            inner_batch_size: int,
            batch_size: int,
            window_size: int,
            dilation_gap: int,
            labels: List[str],
            transformer: str,
            train_filename: Optional[str] = None,
            dev_filename: Optional[str] = None,
            test_filename: Optional[str] = None,
            pred_filenames: Optional[List[str]] = None,
    ):
        super().__init__(inner_batch_size, batch_size, window_size, dilation_gap, labels, transformer, train_filename,
                         dev_filename, test_filename, pred_filenames)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit' or stage is None:
            self.train_dataset = AugmentedDocumentClassificationDataset(filename=self.train_filename,
                                                                        batch_size=self.inner_batch_size,
                                                                        window_size=self.window_size,
                                                                        dilation_gap=self.dilation_gap,
                                                                        defined_labels=self.labels,
                                                                        transformer=self.transformer)
            self.dev_dataset = DocumentClassificationDataset(filename=self.dev_filename,
                                                             batch_size=self.inner_batch_size,
                                                             window_size=self.window_size,
                                                             dilation_gap=self.dilation_gap,
                                                             defined_labels=self.labels,
                                                             transformer=self.transformer)
        if stage == 'test' or stage is None:
            self.test_dataset = DocumentClassificationDataset(filename=self.test_filename,
                                                              batch_size=self.inner_batch_size,
                                                              window_size=self.window_size,
                                                              dilation_gap=self.dilation_gap,
                                                              defined_labels=self.labels,
                                                              transformer=self.transformer)
        if stage == 'predict':
            datasets = []
            for pred_filename in self.pred_filenames:
                dataset = DocumentInferenceDataset(filename=pred_filename,
                                                   batch_size=self.inner_batch_size,
                                                   window_size=self.window_size,
                                                   dilation_gap=self.dilation_gap,
                                                   transformer=self.transformer)
                datasets.append(dataset)
            self.pred_datasets = datasets

class DocumentClassificationDataWithUDA(DocumentClassificationData):
    def __init__(self,
                 inner_batch_size: int,
                 sup_batch_size: int,
                 unsup_batch_size: int,
                 window_size: int,
                 dilation_gap: int,
                 labels: List[str],
                 transformer: str,
                 train_filename: Optional[str] = None,
                 dev_filename: Optional[str] = None,
                 test_filename: Optional[str] = None,
                 pred_filenames: Optional[List[str]] = None,
                 unlabelled_filenames: Optional[str] = None):
        super().__init__(inner_batch_size, sup_batch_size, window_size, dilation_gap, labels, transformer,
                         train_filename, dev_filename, test_filename, pred_filenames)
        self.sup_batch_size = sup_batch_size
        self.unsup_batch_size = unsup_batch_size
        self.unlabelled_filenames = unlabelled_filenames

    def setup(self, stage: Optional[str] = None) -> None:
        super().setup(stage)
        if stage == 'fit' or stage is None:
            self.unsup_dataset = UnlabelledDocumentDataset(filename=self.train_filename,
                                                           batch_size=self.inner_batch_size,
                                                           window_size=self.window_size,
                                                           dilation_gap=self.dilation_gap,
                                                           transformer=self.transformer)

    def train_dataloader(self):
        sup_data = DataLoader(dataset=self.train_dataset, collate_fn=list, batch_size=self.sup_batch_size, shuffle=True)
        unsup_data = DataLoader(dataset=self.unsup_dataset, collate_fn=list, batch_size=self.unsup_batch_size,
                                shuffle=True)

        return {"sup": sup_data, "unsup": unsup_data}

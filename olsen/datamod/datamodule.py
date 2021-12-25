from typing import Optional, List

import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from olsen.datamod.dataset import DocumentClassificationDataset, DocumentInferenceDataset, UnlabelledDocumentDataset, \
    AugmentedDocumentClassificationDataset
from olsen.datamod.sampler import SyncedDistributedSampler, SyncedDistributedBatchSampler


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


class DocumentClassificationDataWithDataAugmentation(DocumentClassificationData):
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


class DocumentClassificationDataForSSL(DocumentClassificationData):
    def __init__(self,
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
                 unlabelled_dirname: Optional[str] = None,
                 ssl_training_mode: str = "overflow",
                 drop_unfill_batch: bool = False,
                 data_ratio: Optional[List[int]] = None):
        super().__init__(inner_batch_size, batch_size, window_size, dilation_gap, labels, transformer,
                         train_filename, dev_filename, test_filename, pred_filenames)
        self.unlabelled_dirname = unlabelled_dirname
        self.drop_last = drop_unfill_batch
        self.mode = ssl_training_mode
        self.ratio = data_ratio

    def setup(self, stage: Optional[str] = None) -> None:
        super().setup(stage)
        if stage == 'fit' or stage is None:
            self.unsup_dataset = UnlabelledDocumentDataset(dirname=self.unlabelled_dirname,
                                                           batch_size=self.inner_batch_size,
                                                           window_size=self.window_size,
                                                           dilation_gap=self.dilation_gap,
                                                           transformer=self.transformer)
            self.cat_dataset = ConcatDataset([self.train_dataset, self.unsup_dataset])


    def train_dataloader(self):
        self.batch_sampler = SyncedDistributedBatchSampler(SyncedDistributedSampler(self.cat_dataset),
                                                           batch_size=self.batch_size, drop_last=self.drop_last,
                                                           mode=self.mode, ratio=self.ratio)
        self.period = self.batch_sampler.period
        return DataLoader(dataset=self.cat_dataset, collate_fn=list, batch_sampler=self.batch_sampler)

    def val_dataloader(self):
        sampler = DistributedSampler(self.dev_dataset, shuffle=False)
        return DataLoader(dataset=self.dev_dataset, collate_fn=list, batch_size=self.batch_size,
                          sampler=sampler)

    def test_dataloader(self):
        sampler = DistributedSampler(self.dev_dataset, shuffle=False)
        return DataLoader(dataset=self.test_dataset, collate_fn=list, batch_size=self.batch_size,
                          sampler=sampler)

    def predict_dataloader(self):
        return [DataLoader(dataset=pred_dataset, collate_fn=list, batch_size=self.batch_size,
                           sampler=DistributedSampler(self.test_dataset, shuffle=False)) for pred_dataset
                in self.pred_datasets]

import random
from abc import ABC, abstractmethod
from typing import List, Sized

from torch.utils.data import Dataset

from olsen.dataunit.labelblock import LabelBlock
from olsen.dataunit.textblock import TextBlock
from olsen.module.tokenizer import SentenceTokenizer


class SizedDataset(Dataset, ABC, Sized):
    @abstractmethod
    def __len__(self):
        pass


class DocumentClassificationDataset(SizedDataset):
    def __init__(
            self, filename: str, defined_labels: List[str], batch_size: int,
            window_size: int, dilation_gap: int, transformer: str
    ):
        super().__init__()
        self.filename = filename
        self.batch_size = batch_size
        self.overlap = int((window_size - 1) / 2.0) * (dilation_gap + 1)
        self.defined_labels = defined_labels
        self.tokenizer = SentenceTokenizer(transformer)
        self.lines, self.labels = self.get_lines_labels()

    def get_lines_labels(self) -> (List[TextBlock], List[LabelBlock]):
        lines: List[str] = []
        labels: List[str] = []
        textblocks: List[TextBlock] = []
        labelblocks: List[LabelBlock] = []
        with open(self.filename) as fp:
            for line in fp:
                if bool(line.strip()):
                    text, label = line.split("###")
                    text = text.strip()
                    label = label.strip()
                    lines.append(text)
                    labels.append(label)
                else:
                    textblock, labelblock = self.create_block(lines, labels)
                    textblocks.extend(textblock)
                    labelblocks.extend(labelblock)
                    lines = []
                    labels = []
        if len(lines) > 0 and len(labels) > 0:
            textblock, labelblock = self.create_block(lines, labels)
            textblocks.extend(textblock)
            labelblocks.extend(labelblock)
        return textblocks, labelblocks

    def create_block(self, lines: List[str], labels: List[str]) -> (List[TextBlock], List[LabelBlock]):
        counter = 0
        while True:
            offset = random.randint(self.overlap, self.batch_size)
            if (len(lines) - offset) % self.batch_size >= self.overlap:
                break
            counter += 1
            assert counter < 10000, "Unable to partition batch"
        textblock = []
        labelblock = []
        line_batch = lines[0:offset]
        label_batch = labels[0:offset]
        begin = []
        end = lines[offset:offset + self.overlap]
        textblock.append(TextBlock(lines=line_batch, begin=begin, end=end, overlap=self.overlap,
                                   tokenizer=self.tokenizer))
        labelblock.append(LabelBlock(labels=label_batch, label_map=self.defined_labels))
        for idx in range(offset, len(lines), self.batch_size):
            line_batch = lines[idx: idx + self.batch_size]
            label_batch = labels[idx: idx + self.batch_size]
            begin = lines[idx - self.overlap:idx]
            if idx > len(lines) - self.batch_size - self.overlap:
                end = lines[idx + self.batch_size: len(lines)]
            else:
                end = lines[idx + self.batch_size: idx + self.batch_size + self.overlap]
            textblock.append(TextBlock(lines=line_batch, begin=begin, end=end, overlap=self.overlap,
                                       tokenizer=self.tokenizer))
            labelblock.append(LabelBlock(labels=label_batch, label_map=self.defined_labels))
        return textblock, labelblock

    def __len__(self):
        assert len(self.lines) == len(self.labels)
        return len(self.lines)

    def __getitem__(self, idx) -> (TextBlock, LabelBlock):
        return self.lines[idx], self.labels[idx]


class AugmentedDocumentClassificationDataset(SizedDataset):
    def __init__(
            self, filename: str, defined_labels: List[str], batch_size: int,
            window_size: int, dilation_gap: int, transformer: str
    ):
        super().__init__()
        self.filename = filename
        self.batch_size = batch_size
        self.overlap = int((window_size - 1) / 2.0) * (dilation_gap + 1)
        self.defined_labels = defined_labels
        self.tokenizer = SentenceTokenizer(transformer)
        self.lines, self.augs, self.labels = self.get_lines_labels()

    def get_lines_labels(self) -> (List[TextBlock], List[TextBlock], List[LabelBlock]):
        lines: List[str] = []
        augs: List[str] = []
        labels: List[str] = []
        textblocks: List[TextBlock] = []
        augblocks: List[TextBlock] = []
        labelblocks: List[LabelBlock] = []
        with open(self.filename) as fp:
            for line in fp:
                if bool(line.strip()):
                    text, aug, label = line.split("###")
                    text = text.strip()
                    aug = aug.strip()
                    label = label.strip()
                    lines.append(text)
                    augs.append(aug)
                    labels.append(label)
                else:
                    textblock, augblock, labelblock = self.create_block(lines, augs, labels)
                    textblocks.extend(textblock)
                    augblocks.extend(augblock)
                    labelblocks.extend(labelblock)
                    lines = []
                    augs = []
                    labels = []
        if len(lines) > 0 and len(labels) > 0:
            textblock, augblock, labelblock = self.create_block(lines, augs, labels)
            textblocks.extend(textblock)
            augblocks.extend(augblock)
            labelblocks.extend(labelblock)
        return textblocks, augblocks, labelblocks

    def create_block(self, lines: List[str], augs: List[str], labels: List[str]) -> \
            (List[TextBlock], List[LabelBlock], List[LabelBlock]):
        counter = 0
        while True:
            offset = random.randint(self.overlap, self.batch_size)
            if (len(lines) - offset) % self.batch_size >= self.overlap:
                break
            counter += 1
            assert counter < 10000, "Unable to partition batch"
        textblock = []
        augblock = []
        labelblock = []
        line_batch = lines[0:offset]
        aug_batch = augs[0:offset]
        label_batch = labels[0:offset]
        begin = []
        end = lines[offset:offset + self.overlap]
        textblock.append(TextBlock(lines=line_batch, begin=begin, end=end, overlap=self.overlap,
                                   tokenizer=self.tokenizer))
        aug_begin = []
        aug_end = augs[offset:offset + self.overlap]
        augblock.append(TextBlock(lines=aug_batch, begin=aug_begin, end=aug_end, overlap=self.overlap,
                                  tokenizer=self.tokenizer))
        labelblock.append(LabelBlock(labels=label_batch, label_map=self.defined_labels))
        for idx in range(offset, len(lines), self.batch_size):
            line_batch = lines[idx: idx + self.batch_size]
            aug_batch = augs[idx: idx + self.batch_size]
            label_batch = labels[idx: idx + self.batch_size]
            begin = lines[idx - self.overlap:idx]
            if idx > len(lines) - self.batch_size - self.overlap:
                end = lines[idx + self.batch_size: len(lines)]
            else:
                end = lines[idx + self.batch_size: idx + self.batch_size + self.overlap]
            textblock.append(TextBlock(lines=line_batch, begin=begin, end=end, overlap=self.overlap,
                                       tokenizer=self.tokenizer))
            aug_begin = augs[idx - self.overlap:idx]
            if idx > len(augs) - self.batch_size - self.overlap:
                aug_end = augs[idx + self.batch_size: len(augs)]
            else:
                aug_end = augs[idx + self.batch_size: idx + self.batch_size + self.overlap]
            augblock.append(TextBlock(lines=aug_batch, begin=aug_begin, end=aug_end, overlap=self.overlap,
                                      tokenizer=self.tokenizer))
            labelblock.append(LabelBlock(labels=label_batch, label_map=self.defined_labels))
        return textblock, augblock, labelblock

    def __len__(self):
        assert len(self.lines) == len(self.labels) == len(self.augs)
        return len(self.lines)

    def __getitem__(self, idx) -> (TextBlock, LabelBlock):
        return self.lines[idx], self.augs[idx], self.labels[idx]


class DocumentInferenceDataset(SizedDataset):
    def __init__(
            self, filename: str, batch_size: int, window_size: int, dilation_gap: int, transformer: str
    ):
        super().__init__()
        self.filename = filename
        self.batch_size = batch_size
        self.overlap = int((window_size - 1) / 2.0) * (dilation_gap + 1)
        self.tokenizer = SentenceTokenizer(transformer)
        self.lines = self.get_lines()

    def get_lines(self) -> List[TextBlock]:
        with open(self.filename) as fp:
            lines = [line.strip() for line in fp if bool(line.strip())]
        textblock = self.create_block(lines)
        return textblock

    def create_block(self, lines: List[str]) -> List[TextBlock]:
        counter = 0
        while True:
            offset = random.randint(self.overlap, self.batch_size)
            if (len(lines) - offset) % self.batch_size >= self.overlap:
                break
            counter += 1
            assert counter < 10000, "Unable to partition batch"
        textblock = []
        line_batch = lines[0:offset]
        begin = []
        end = lines[offset:offset + self.overlap]
        textblock.append(TextBlock(lines=line_batch, begin=begin, end=end, overlap=self.overlap,
                                   tokenizer=self.tokenizer))
        for idx in range(offset, len(lines), self.batch_size):
            line_batch = lines[idx: idx + self.batch_size]
            begin = lines[idx - self.overlap:idx]
            if idx > len(lines) - self.batch_size - self.overlap:
                end = lines[idx + self.batch_size: len(lines)]
            else:
                end = lines[idx + self.batch_size: idx + self.batch_size + self.overlap]
            textblock.append(TextBlock(lines=line_batch, begin=begin, end=end, overlap=self.overlap,
                                       tokenizer=self.tokenizer))
        return textblock

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx) -> TextBlock:
        return self.lines[idx]


class UnlabelledDocumentDataset(SizedDataset):
    def __init__(
            self, filename: str, batch_size: int,
            window_size: int, dilation_gap: int, transformer: str
    ):
        super().__init__()
        self.filename = filename
        self.batch_size = batch_size
        self.overlap = int((window_size - 1) / 2.0) * (dilation_gap + 1)
        self.tokenizer = SentenceTokenizer(transformer)
        self.original, self.backtran = self.get_lines()

    def get_lines(self) -> (List[TextBlock], List[TextBlock]):
        original: List[str] = []
        backtran: List[str] = []
        origblocks: List[TextBlock] = []
        backblocks: List[TextBlock] = []
        with open(self.filename) as fp:
            for line in fp:
                if bool(line.strip()):
                    orig, back = line.split("###")
                    orig = orig.strip()
                    back = back.strip()
                    original.append(orig)
                    backtran.append(back)
                else:
                    origblock, backblock = self.create_block(original, backtran)
                    origblocks.extend(origblock)
                    backblocks.extend(backblock)
                    original = []
                    backtran = []
        if len(original) > 0 and len(backtran) > 0:
            origblock, backblock = self.create_block(original, backtran)
            origblocks.extend(origblock)
            backblocks.extend(backblock)
        return original, backtran

    def create_block(self, original: List[str], backtran: List[str]) -> (List[TextBlock], List[TextBlock]):
        counter = 0
        while True:
            offset = random.randint(self.overlap, self.batch_size)
            if (len(original) - offset) % self.batch_size >= self.overlap:
                break
            counter += 1
            assert counter < 10000, "Unable to partition batch"
        length = len(original)
        assert length == len(backtran), "Augmented data inconsistent with original: line number difference"
        origblock = []
        backblock = []
        orig_batch = original[0:offset]
        back_batch = backtran[0:offset]
        orig_begin = []
        back_begin = []
        orig_end = original[offset:offset + self.overlap]
        back_end = backtran[offset:offset + self.overlap]
        origblock.append(TextBlock(lines=orig_batch, begin=orig_begin, end=orig_end, overlap=self.overlap,
                                   tokenizer=self.tokenizer))
        backblock.append(TextBlock(lines=back_batch, begin=back_begin, end=back_end, overlap=self.overlap,
                                   tokenizer=self.tokenizer))
        for idx in range(offset, length, self.batch_size):
            orig_batch = original[idx: idx + self.batch_size]
            back_batch = backtran[idx: idx + self.batch_size]
            orig_begin = original[idx - self.overlap:idx]
            back_begin = backtran[idx - self.overlap:idx]
            if idx > length - self.batch_size - self.overlap:
                orig_end = original[idx + self.batch_size: length]
                back_end = backtran[idx + self.batch_size: length]
            else:
                orig_end = original[idx + self.batch_size: idx + self.batch_size + self.overlap]
                back_end = backtran[idx + self.batch_size: idx + self.batch_size + self.overlap]
            origblock.append(TextBlock(lines=orig_batch, begin=orig_begin, end=orig_end, overlap=self.overlap,
                                       tokenizer=self.tokenizer))
            backblock.append(TextBlock(lines=back_batch, begin=back_begin, end=back_end, overlap=self.overlap,
                                       tokenizer=self.tokenizer))
        return origblock, backblock

    def __len__(self):
        assert len(self.original) == len(self.backtran)
        return len(self.original)

    def __getitem__(self, idx) -> (TextBlock, TextBlock):
        return self.original[idx], self.backtran[idx]

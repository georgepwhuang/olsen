import random
from typing import List

from torch.utils.data import Dataset

from olsen.dataunit.labelblock import LabelBlock
from olsen.dataunit.textblock import TextBlock
from olsen.module.tokenizer import SentenceTokenizer


class DocumentClassificationDataset(Dataset):
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


class DocumentInferenceDataset(Dataset):
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

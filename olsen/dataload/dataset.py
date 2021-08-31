import random
from typing import List

from torch.utils.data import Dataset

from olsen.data.labelblock import LabelBlock
from olsen.data.textblock import TextBlock


class DocumentClassificationDataset(Dataset):
    def __init__(
            self, filename: str, defined_labels: List[str], batch_size: int,
            window_size: int, dilation_gap: int
    ):
        super().__init__()
        self.filename = filename
        self.batch_size = batch_size
        self.overlap = int((window_size - 1) / 2.0) * (dilation_gap + 1)
        self.defined_labels = defined_labels
        self.lines, self.labels = self.get_lines_labels()

    def get_lines_labels(self) -> (List[TextBlock], List[LabelBlock]):
        lines: List[str] = []
        labels: List[str] = []
        textblocks: List[TextBlock] = []
        labelblocks: List[LabelBlock] = []
        with open(self.filename) as fp:
            for line in fp:
                if bool(line.strip()):
                    line, label = line.split("###")
                    line = line.strip()
                    label = label.strip()
                    lines.append(line)
                    labels.append(label)
                else:
                    lines_read = len(lines)
                    textblock, labelblock = self.create_block(lines, labels)
                    textblocks.extend(textblock)
                    labelblocks.extend(labelblock)
                    lines_batched = sum([len(block.lines) for block in textblock])
                    assert lines_read == lines_batched, \
                        f"Lines read ({lines_read}) and lines batched ({lines_batched})do not match"
                    lines = []
                    labels = []
        if len(lines) > 0 and len(labels) > 0:
            lines_read = len(lines)
            textblock, labelblock = self.create_block(lines, labels)
            textblocks.extend(textblock)
            labelblocks.extend(labelblock)
            lines_batched = sum([len(block.lines) for block in textblock])
            assert lines_read == lines_batched, \
                f"Lines read ({lines_read}) and lines batched ({lines_batched})do not match"
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
        textblock.append(TextBlock(lines=line_batch, begin=begin, end=end, overlap=self.overlap))
        labelblock.append(LabelBlock.from_labels(labels=label_batch, label_map=self.defined_labels))
        for idx in range(offset, len(lines), self.batch_size):
            line_batch = lines[idx: idx + self.batch_size]
            label_batch = labels[idx: idx + self.batch_size]
            begin = lines[idx - self.overlap:idx]
            if idx > len(lines) - self.batch_size - self.overlap:
                end = lines[idx + self.batch_size: len(lines)]
            else:
                end = lines[idx + self.batch_size: idx + self.batch_size + self.overlap]
            textblock.append(TextBlock(lines=line_batch, begin=begin, end=end, overlap=self.overlap))
            labelblock.append(LabelBlock.from_labels(labels=label_batch, label_map=self.defined_labels))
        return textblock, labelblock

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx) -> (TextBlock, LabelBlock):
        return self.lines[idx], self.labels[idx]


class DocumentInferenceDataset(Dataset):
    def __init__(
            self, filename: str, batch_size: int = 16,
            window_size: int = 3, dilation_gap: int = 0
    ):
        super().__init__()
        self.filename = filename
        self.batch_size = batch_size
        self.overlap = int((window_size - 1) / 2.0) * (dilation_gap + 1)
        self.lines = self.get_lines()

    def get_lines(self) -> List[TextBlock]:
        lines: List[str] = []
        textblock: List[TextBlock] = []
        with open(self.filename) as fp:
            for line in fp:
                if bool(line.strip()):
                    line = line.strip()
                    lines.append(line)
                else:
                    lines_read = len(lines)
                    file_batch = self.create_block(lines)
                    textblock.extend(file_batch)
                    lines_batched = sum([len(file.lines) for file in file_batch])
                    assert lines_read == lines_batched, \
                        f"Lines read ({lines_read}) and lines batched ({lines_batched})do not match"
                    lines = []
        if len(lines) > 0:
            lines_read = len(lines)
            file_batch = self.create_block(lines)
            textblock.extend(file_batch)
            lines_batched = sum([len(file.lines) for file in file_batch])
            assert lines_read == lines_batched, \
                f"Lines read ({lines_read}) and lines batched ({lines_batched})do not match"
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
        textblock.append(TextBlock(lines=line_batch, begin=begin, end=end, overlap=self.overlap))
        for idx in range(offset, len(lines), self.batch_size):
            line_batch = lines[idx: idx + self.batch_size]
            begin = lines[idx - self.overlap:idx]
            if idx > len(lines) - self.batch_size - self.overlap:
                end = lines[idx + self.batch_size: len(lines)]
            else:
                end = lines[idx + self.batch_size: idx + self.batch_size + self.overlap]
            textblock.append(TextBlock(lines=line_batch, begin=begin, end=end, overlap=self.overlap))
        return textblock

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx) -> (TextBlock):
        return (self.lines[idx])

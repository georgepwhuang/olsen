from functools import reduce
from math import gcd, ceil
from typing import Optional, Iterator, List

import numpy as np
from torch.utils.data import Sampler, BatchSampler, ConcatDataset
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import math


def __cum_to_ind__(cum_distr):
    prev = 0
    ind_distr = []
    for item in cum_distr:
        ind_distr.append(item - prev)
        prev = item
    return ind_distr


def __ind_to_cum__(ind_distr):
    l = len(ind_distr)
    cum_distr = [0]
    cum_distr[0] = ind_distr[0]
    for i in range(1, l):
        cum_distr.append(0)
        cum_distr[i] = cum_distr[i - 1] + ind_distr[i]
    return cum_distr


class SyncedSampler(Sampler[int]):
    def __init__(self, dataset: ConcatDataset, seed: Optional[int] = None):
        self.cum_distr = dataset.cumulative_sizes
        self.dataset_count = len(dataset.cumulative_sizes)
        self.seed = seed

    def __iter__(self):
        if self.seed:
            np.random.seed(self.seed)
        start = 0
        indices = []
        for end in self.cum_distr:
            arr = np.arange(start, end)
            np.random.shuffle(arr)
            indices.append(arr)
            start = end + 1
        return iter(np.concatenate(indices))

    def __len__(self):
        return self.cum_distr[-1]


class SyncedBatchSampler(BatchSampler):
    def __init__(self, sampler: SyncedSampler, batch_size: int, drop_last: bool,
                 mode: str, ratio: List[int] = None):
        super().__init__(sampler, batch_size, drop_last)
        self.cum_distr = self.sampler.cum_distr
        self.dataset_count = self.sampler.dataset_count
        self.ind_distr = __cum_to_ind__(self.cum_distr)

        self.mode = mode
        if self.mode == "truncate":
            self.ratio = [int(item / self.ind_distr[0]) for item in self.ind_distr]
        elif self.mode == "overflow":
            self.ratio = [int(ceil(item / self.ind_distr[0])) for item in self.ind_distr]
        else:
            raise ValueError("Not a valid mode")

        if ratio is not None:
            self.ratio = [int(item / reduce(gcd, ratio)) for item in ratio]

        self.period = sum(self.ratio)
        self.order = self.__generate_order__()

    def __generate_order__(self):
        batch_list = []
        for i in range(self.dataset_count):
            batch_list.append([])

        dataset_idx = 0
        tmp = []
        for idx, item in enumerate(self.sampler):
            tmp.append(item)
            if len(tmp) == self.batch_size:
                batch_list[dataset_idx].append(tmp)
                tmp = []
            if idx + 1 == self.cum_distr[dataset_idx]:
                if len(tmp) > 0 and not self.drop_last:
                    batch_list[dataset_idx].append(tmp)
                    tmp = []
                dataset_idx += 1

        order = []
        counter = 0
        while True:
            for j in range(0, self.dataset_count):
                for k in range(self.ratio[j]):
                    idx = (counter * self.ratio[j] + k) % len(batch_list[j])
                    order.append(batch_list[j][idx])
            if self.mode == "truncate":
                for j in range(0, self.dataset_count):
                    if (counter + 1) * self.ratio[j] >= len(batch_list[j]):
                        break
            elif self.mode == "overflow":
                check = True
                for j in range(0, self.dataset_count):
                    if (counter + 1) * self.ratio[j] < len(batch_list[j]):
                        check = False
                if check:
                    break
            counter += 1
        return order

    def __iter__(self):
        return iter(self.order)

    def __len__(self):
        return len(self.order)


class SyncedDistributedSampler(DistributedSampler, SyncedSampler):
    def __init__(self, dataset: ConcatDataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.cum_distr = dataset.cumulative_sizes
        self.dataset_count = len(dataset.cumulative_sizes)
        self.seed = seed
        self.ind_distr = __cum_to_ind__(self.cum_distr)
        self.num_samples = []
        for i in range(self.dataset_count):
            if self.drop_last and self.ind_distr[i] % self.num_replicas != 0:
                self.num_samples.append(math.ceil(
                    (self.ind_distr[i] - self.num_replicas) / self.num_replicas
                ))
            else:
                self.num_samples.append(math.ceil((self.ind_distr[i]) / self.num_replicas))
        self.each_size = [self.num_samples[i] * self.num_replicas for i in range(self.dataset_count)]
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            np.random.seed(self.seed + self.epoch)
        start = 0
        indices = []
        for end in self.cum_distr:
            arr = np.arange(start, end)
            if self.shuffle:
                np.random.shuffle(arr)
            indices.append(arr)
            start = end + 1
        for i in range(self.dataset_count):
            if not self.drop_last:
                self.each_size[i] - len(indices[i])
                indices[i] = np.tile(indices[i], math.ceil(self.each_size[i]/ len(indices[i])))[:self.each_size[i]]
            else:
                indices[i] = indices[i][:self.each_size[i]]
            assert len(indices[i]) == self.each_size[i]

        for i in range(self.dataset_count):
            assert len(indices[i]) % self.num_replicas == 0
            indices[i] = indices[i][self.rank:self.each_size[i]:self.num_replicas]
            assert len(indices[i]) == self.num_samples[i]

        return iter(np.concatenate(indices))

    def __len__(self) -> int:
        return sum(self.num_samples)


class SyncedDistributedBatchSampler(SyncedBatchSampler):
    def __init__(self, sampler: SyncedDistributedSampler, batch_size: int, drop_last: bool,
                 mode: str, ratio: List[int]):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.ind_distr = self.sampler.num_samples
        self.cum_distr = __ind_to_cum__(self.ind_distr)
        self.dataset_count = self.sampler.dataset_count

        self.mode = mode
        '''
        if self.mode == "truncate":
            self.ratio = [int(item / self.ind_distr[0]) for item in self.ind_distr]
        elif self.mode == "overflow":
            self.ratio = [int(ceil(item / self.ind_distr[0])) for item in self.ind_distr]
        else:
            raise ValueError("Not a valid mode")

        if ratio is not None:
            self.ratio = [int(item / reduce(gcd, ratio)) for item in ratio]
        '''

        self.ratio = [int(item / reduce(gcd, ratio)) for item in ratio]

        self.period = sum(self.ratio)
        self.order = self.__generate_order__()

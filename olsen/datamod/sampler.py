from functools import reduce
from math import gcd, ceil
from typing import Optional, List

import numpy as np
from torch.utils.data import Sampler, BatchSampler, ConcatDataset


class DistributedRandomSampler(Sampler[int]):
    def __init__(self, data_source: ConcatDataset, seed: Optional[int] = None):
        super().__init__(data_source)
        self.cum_distr = data_source.cumulative_sizes
        self.dataset_count = len(data_source.cumulative_sizes)
        self.seed = seed
        self.order = self.__generate__()

    def __generate__(self):
        if self.seed:
            np.random.seed(self.seed)
        start = 0
        ls = []
        for end in self.cum_distr:
            arr = np.arange(start, end)
            np.random.shuffle(arr)
            ls.append(arr)
            start = end + 1
        return np.concatenate(ls)

    def __iter__(self):
        return iter(self.order)


class SyncedBatchSampler(BatchSampler):
    def __init__(self, sampler: DistributedRandomSampler, batch_size: int, drop_last: bool,
                 mode: str, ratio: List[int] = None):
        super().__init__(sampler, batch_size, drop_last)
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.cum_distr = self.sampler.cum_distr
        self.dataset_count = self.sampler.dataset_count
        self.ind_distr = self.__cum_to_ind__()

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
                    idx = (counter * self.ratio[j] + k) % self.ind_distr[j]
                    order.append(batch_list[j][idx])
            if self.mode == "truncate":
                for j in range(0, self.dataset_count):
                    if (counter + 1) * self.ratio[j] >= self.ind_distr[j]:
                        break
            elif self.mode == "overflow":
                check = True
                for j in range(0, self.dataset_count):
                    if (counter + 1) * self.ratio[j] < self.ind_distr[j]:
                        check = False
                if check:
                    break
        return order

    def __cum_to_ind__(self):
        prev = 0
        ind_distr = []
        for item in self.cum_distr:
            ind_distr.append(item - prev)
            prev = item
        return ind_distr

    def __iter__(self):
        return iter(self.order)

import random
from typing import List

from torch.utils.data import IterableDataset

from .laion import LaionIterJsonDataset
from .openimage import OpenImageBLIPaug_Dataset


class ProbPickingDataset(IterableDataset):
    """A dataset wrapper for picking dataset with probability."""

    def __init__(self, datasets: List[dict]):
        super().__init__()
        assert sum([dataset["prob"] for dataset in datasets]) == 1

        self.dataset_list = []
        self.range_list = []

        start_idx = 0
        for dataset_prob in datasets:
            dataset = dataset_prob["dataset"]
            prob = dataset_prob["prob"]
            end_idx = start_idx + prob
            self.dataset_list.append(iter(dataset))
            self.range_list.append([start_idx, end_idx])
            start_idx = end_idx

    def __iter__(self):
        while True:
            rand_num = random.random()
            for idx, (s, e) in enumerate(self.range_list):
                if s <= rand_num < e:
                    iterator = self.dataset_list[idx]
                    try:
                        data = next(iterator)
                    except StopIteration:
                        iterator = iter(self.dataset_list[idx])
                        self.dataset_list[idx] = iterator
                        data = next(iterator)
                    yield data

    def __len__(self):
        # pesudo length
        return 999_999_999


__all__ = ["OpenImageBLIPaug_Dataset", "LaionIterJsonDataset", "ProbPickingDataset"]

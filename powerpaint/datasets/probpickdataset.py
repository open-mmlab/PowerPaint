import random
from collections.abc import Iterable, Mapping
from typing import List

from torch.utils.data import Dataset, IterableDataset


class MappingDatasetWrapper(IterableDataset):
    """This class is used to wrap a mapping dataset to an iterable dataset."""

    def __init__(self, dataset: Dataset):
        # assert isinstance(dataset, Mapping)
        self.length = len(dataset)
        self.dataset = dataset

    def __iter__(self):
        while True:
            for idx in range(self.length):
                yield self.dataset[idx]


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

            if not isinstance(dataset, Dataset):
                dataset = DATASETS.build(dataset)
            if isinstance(dataset, Mapping):
                dataset = MappingDatasetWrapper(dataset)
            if not isinstance(dataset, Iterable):
                dataset = MappingDatasetWrapper(dataset)

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


if __name__ == "__main__":
    register_all_modules()

    mdj_pipeline = [
        dict(type="LoadImageFromFile", key="img", channel_order="rgb"),
        dict(type="CenterCropLongEdge", keys="img"),
        dict(type="Resize", scale=(512, 512), keys="img", backend="pillow"),
        dict(type="PackInputs", keys=["img"], data_keys="prompt"),
    ]

    mdj_dataset = dict(
        type="Midjourney_AIGCMM_VAL", root="/mnt/petrelfs/liuwenran/datasets/aigcmm/AIGCMM_VAL", pipeline=mdj_pipeline
    )

    # define laion dataset
    backend_args = dict(backend="petrel", path_mapping={"/": "laion:s3://laion5b/"})
    pipeline = [
        dict(type="LoadImageFromFile", key="img", channel_order="rgb", backend_args=backend_args),
        dict(type="CenterCropLongEdge", keys="img"),
        dict(type="Resize", scale=(512, 512), keys="img", backend="pillow"),
        dict(
            type="PackInputs",
            keys=["img"],
            data_keys="prompt",
            meta_keys=["width", "height", "similarity_score", "aesthetic_score"],
        ),
    ]
    anno_root = "laion:s3://llm-process/laion-5b/format/v020/laion2B-en/"
    laion_dataset = dict(type="LaionIterJsonDataset", anno_root=anno_root, pipeline=pipeline, bufsize=100)

    # mix the dataset
    dataset = dict(
        type="ProbPickingDataset",
        datasets=[
            dict(dataset=mdj_dataset, prob=1.0),
        ],
    )

    dataset = DATASETS.build(dataset)
    iterator = iter(dataset)
    idx = 0
    while idx < 5000000:
        data = next(iterator)
        # print('=============== idx ==================')
        if idx % 1000 == 0:
            print(idx)
        idx += 1
        print("data ind " + str(idx))

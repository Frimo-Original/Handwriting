import math
import random

from torch.utils.data import Sampler, Subset


def dataset_lengths(dataset, max_seq_len=None):
    if isinstance(dataset, Subset):
        parent_lengths = dataset_lengths(dataset.dataset, max_seq_len)
        return [parent_lengths[idx] for idx in dataset.indices]

    if hasattr(dataset, "points"):
        lengths = [len(points) for points in dataset.points]
    else:
        lengths = [int(dataset[idx]["length"]) for idx in range(len(dataset))]

    if max_seq_len is not None:
        lengths = [min(length, max_seq_len) for length in lengths]
    return lengths


class LengthBucketBatchSampler(Sampler):
    def __init__(
        self,
        lengths,
        batch_size,
        shuffle=True,
        drop_last=False,
        bucket_size_multiplier=50,
        seed=0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.lengths = list(lengths)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.bucket_size = max(self.batch_size, self.batch_size * int(bucket_size_multiplier))
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        indices = list(range(len(self.lengths)))

        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            self.epoch += 1
            rng.shuffle(indices)
            batches = []
            for start in range(0, len(indices), self.bucket_size):
                bucket = indices[start : start + self.bucket_size]
                bucket.sort(key=lambda idx: self.lengths[idx], reverse=True)
                batches.extend(self._split_batches(bucket))
            rng.shuffle(batches)
        else:
            indices.sort(key=lambda idx: self.lengths[idx], reverse=True)
            batches = self._split_batches(indices)

        yield from batches

    def _split_batches(self, indices):
        batches = []
        for start in range(0, len(indices), self.batch_size):
            batch = indices[start : start + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)
        return batches

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return math.ceil(len(self.lengths) / self.batch_size)

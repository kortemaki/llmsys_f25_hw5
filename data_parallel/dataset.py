from random import Random

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist


def saferound(fracs: np.array) -> np.array:
    """
    Simplified version of iteround.saferound that works for this case.

    Ensures that sum(assignments) == sum(fracs)
    """
    assignments = fracs.astype(int)
    to_assign = fracs.sum() - assignments.sum()
    if to_assign != int(to_assign):
        raise ValueError("fracs must sum to an integer!")
    i = 0
    while to_assign:
        to_assign -= 1
        assignments[i] += 1
        i += 1
    return assignments


class Partition():
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        '''Given index, get the data according to the partitioned index'''
        # BEGIN ASSIGN5_1_1
        return self.data[self.index[index]]
        # END ASSIGN5_1_1


class DataPartitioner():
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        ''' Create indices for different partitions
        1. Create indices and use `rng` to shuffle indices
        2. Create different partitions of indices according to `sizes` and store in `self.partitions`
        '''
        # BEGIN ASSIGN5_1_1
        # ensure sizes sums to 1
        sizes = np.array(sizes)
        sizes = sizes / np.sum(sizes)

        # map sizes to integer split points of the datapoints
        n = len(self.data)
        sizes = saferound(sizes * n)
        splits = np.cumsum(sizes)

        # split index according to sizes
        index = list(range(n))
        rng.shuffle(index)
        index = np.array(index)
        indices = np.split(index, np.cumsum(splits))

        # create partitions
        self.partitions = [
            Partition(
                data=self.data,
                index=indices[i]
            )
            for i in range(len(sizes)) # use sizes here because it is the correct length
        ]
        # END ASSIGN5_1_1

    def use(self, partition):
        ''' Return a simple dataset class `Partiton` by original data and partitioned indices

        Just one line of code. Think it simply.
        '''
        # BEGIN ASSIGN5_1_1
        return self.partitions[partition]
        # END ASSIGN5_1_1


def partition_dataset(rank, world_size, dataset, batch_size=128, collate_fn=None):
    """ Partitioning training dataset of the Machine Translation

    Returns:
        DataLoader: partitioned dataloader

    Hint:
    1. Calculate the partitioned batch size
    2. Create a partitioner class `DataPartitioner` with dataset and the list of partitioned sizes
    3. Get the current partition dataset given `rank`, use the `use` function in DataPartitioner
    4. Wrap the dataset with `DataLoader`, remember to customize the `collate_fn`
    """
    # BEGIN ASSIGN5_1
    partitioned_sizes = world_size * [1 / world_size]
    batch_size = saferound(np.array(world_size * [batch_size / world_size]))[rank]

    parts = DataPartitioner(dataset, partitioned_sizes)
    return DataLoader(parts.use(rank), batch_size=batch_size, collate_fn=collate_fn)
    # END ASSIGN5_1

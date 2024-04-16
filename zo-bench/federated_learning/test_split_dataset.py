import numpy as np
import torch
from torch.utils.data import Dataset
from split_dataset import split_dataset
class DummyDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return index
    
    def shuffle(self, seed):
        return self
    
    def shard(self, num_shards, shard_index):
        shard_size = self.num_samples // num_shards
        start = shard_index * shard_size
        end = (shard_index + 1) * shard_size
        return DummyDataset(end - start)
    def select(self, indices):
        return DummyDataset(len(indices))

def test_split_dataset_iid():
    fed_args = type("FedArgs", (), {"split_strategy": "iid", "num_clients": 4})
    script_args = type("ScriptArgs", (), {"seed": 42})
    dataset = DummyDataset(100)
    dataset = dataset.select(range(20))
    local_datasets, _ = split_dataset(fed_args, script_args, dataset)

    assert len(local_datasets) == 4
    assert sum(len(local_dataset) for local_dataset in local_datasets) == len(dataset)
    print([len(local_dataset) for local_dataset in local_datasets])

def test_split_dataset_noniid():
    fed_args = type("FedArgs", (), {"split_strategy": "noniid", "num_clients": 3, "alpha": 0.5})
    script_args = type("ScriptArgs", (), {"seed": 42})
    dataset = DummyDataset(100)
    dataset = dataset[:20]

    local_datasets, _ = split_dataset(fed_args, script_args, dataset)

    assert len(local_datasets) == 3
    print([len(local_dataset) for local_dataset in local_datasets])

def test_split_dataset_noniid_with_min_partition_size():
    fed_args = type("FedArgs", (), {"split_strategy": "noniid", "num_clients": 5, "alpha": 0.2, "min_partition_size": 10})
    script_args = type("ScriptArgs", (), {"seed": 42})
    dataset = DummyDataset(100)

    local_datasets, _ = split_dataset(fed_args, script_args, dataset)

    assert len(local_datasets) == 5
    print([len(local_dataset) for local_dataset in local_datasets])
    print(sum(len(local_dataset) for local_dataset in local_datasets))
    for local_dataset in local_datasets:
        assert len(local_dataset) >= 10

test_split_dataset_iid()
test_split_dataset_noniid()
test_split_dataset_noniid_with_min_partition_size()
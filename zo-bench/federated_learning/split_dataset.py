import random
import numpy as np

def split_dataset(fed_args, seed, dataset):
    dataset = dataset.shuffle(seed=seed)        # Shuffle the dataset
    print(f"Total number of samples: {len(dataset)}")
    local_datasets = []
    local_num_samples = [len(dataset) // fed_args.num_clients] * fed_args.num_clients
    if fed_args.split_strategy == "iid":
        for i in range(fed_args.num_clients):
            local_datasets.append(dataset.shard(fed_args.num_clients, i))
    elif fed_args.split_strategy == "noniid":
        # Dirichlet distribution for number of examples per client
        # assert parameters for dirichlet distribution
        assert hasattr(fed_args, "alpha") and fed_args.alpha > 0
        if hasattr(fed_args, "min_partition_size") and fed_args.min_partition_size > 0:
            min_partition_size = fed_args.min_partition_size
            min_size =0
            while min_size < min_partition_size:
                proportions = np.random.dirichlet(np.repeat(fed_args.alpha, fed_args.num_clients))
                proportions = proportions/proportions.sum()
                min_size = min(proportions * len(dataset))   
        else:
            proportions = np.random.dirichlet(np.repeat(fed_args.alpha, fed_args.num_clients))
        # local datasets
        for i in range(fed_args.num_clients):
            num_samples = int(proportions[i] * len(dataset))
            local_num_samples[i] = num_samples
            local_datasets.append(dataset.select(range(num_samples)))
            # print(f"Client {i} has {num_samples} samples.")
            
    return local_datasets, local_num_samples

def get_dataset_this_round(dataset, round, fed_args, script_args):
    num2sample = script_args.batch_size * script_args.gradient_accumulation_steps * script_args.max_steps
    num2sample = min(num2sample, len(dataset))
    random.seed(round)
    random_idx = random.sample(range(0, len(dataset)), num2sample)
    dataset_this_round = dataset.select(random_idx)

    return dataset_this_round
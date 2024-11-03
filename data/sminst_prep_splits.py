import os
import random
from collections import defaultdict

import numpy as np
import tonic
import torch
from torch.utils.data import TensorDataset
import tonic.transforms as transforms


def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility across numpy, random, and torch.

    Args:
        seed (int): Seed value for reproducibility.
    """
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def convert_dataset(events_dataset: torch.utils.data.Dataset) -> TensorDataset:
    """
    Convert an event-based dataset into a TensorDataset of event tensors and labels.

    Args:
        events_dataset (Dataset): A dataset of event-based data with labels.

    Returns:
        TensorDataset: Converted dataset in tensor format (events, labels).
    """
    x_data, y_data = zip(*((events, label) for events, label in events_dataset))
    return TensorDataset(torch.as_tensor(np.array(x_data)), torch.as_tensor(np.array(y_data)))


def stratified_split(
        dataset: TensorDataset,
        fraction: float,
        multiple: bool = False,
        n_split: int = None,
        save: bool = False,
        save_path: str = "."
) -> tuple:
    """
    Perform a stratified split on the dataset to ensure label distribution is consistent across splits.

    Args:
        dataset (TensorDataset): Input dataset to be split.
        fraction (float): Fraction of data to include in the first split.
        multiple (bool): If True, saves multiple splits with suffixes. Default is False.
        n_split (int): Index for multiple splits. Used if `multiple` is True. Default is None.
        save (bool): If True, saves splits to the specified path. Default is False.
        save_path (str): Directory path to save the splits. Default is the current directory.

    Returns:
        tuple: Tuple of (split1, split2) TensorDatasets if save=False; otherwise None.
    """

    if isinstance(dataset, list) and len(dataset) > 1:
        dataset = torch.utils.data.ConcatDataset(dataset)
        data = torch.cat([ds.tensors[0] for ds in dataset.datasets])
        labels = torch.cat([ds.tensors[1] for ds in dataset.datasets])
        dataset = TensorDataset(data, labels)
    else:
        labels = dataset.tensors[1]

    # Group indices by label
    indices_per_label = defaultdict(list)
    for index, label in enumerate(labels):
        indices_per_label[label.item()].append(index)

    first_indices, second_indices = [], []

    # Split each label group
    for label, indices in indices_per_label.items():
        n_samples = round(len(indices) * fraction)
        selected_indices = random.sample(indices, n_samples)
        first_indices.extend(selected_indices)
        second_indices.extend(set(indices) - set(selected_indices))

    random.shuffle(first_indices)
    random.shuffle(second_indices)

    # Create tensor datasets for splits
    split1 = TensorDataset(dataset[first_indices][0], dataset[first_indices][1])
    split2 = TensorDataset(dataset[second_indices][0], dataset[second_indices][1])

    # Display summary of the split
    print("Stratified split completed with the following results:")
    print(f"\t Original dataset length: {len(dataset)}")
    print(f"\t Requested split fractions: {np.round(fraction * 100, 2)}%:{100 - np.round(fraction * 100, 2)}%")
    for idx, split in enumerate([split1, split2], 1):
        labels, counts = torch.unique(split.tensors[1], return_counts=True)
        print(f"\t Split {idx} - length: {len(split)}")
        print(f"\t\t Labels: {labels.tolist()} with counts: {counts.tolist()}")

    # Save the splits if requested
    if save:
        os.makedirs(save_path, exist_ok=True)
        train_filename = f"ds_train_{n_split}.pt" if multiple else "ds_train.pt"
        val_filename = f"ds_val_{n_split}.pt" if multiple else "ds_val.pt"
        torch.save(split1, os.path.join(save_path, train_filename))
        torch.save(split2, os.path.join(save_path, val_filename))
    else:
        return split1, split2


if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(42)

    # Define parameters
    n_splits = 10
    save = True
    data_dir = ""
    neurons = 11
    sensor_size = (neurons, 1, 1)

    # Define transformation pipeline
    transform_pipeline = transforms.Compose([
        transforms.ToFrame(sensor_size=sensor_size, n_time_bins=784),
        transforms.NumpyAsType('float32'),
    ])

    # Load datasets
    train_dataset = tonic.datasets.SMNIST(save_to=data_dir, num_neurons=neurons, train=True, duplicate=False, transform=transform_pipeline)
    test_dataset = tonic.datasets.SMNIST(save_to=data_dir, num_neurons=neurons, train=False, duplicate=False, transform=transform_pipeline)

    # Convert and save the test dataset
    data_dir = os.path.join(data_dir, "smnist_splitted")
    os.makedirs(data_dir, exist_ok=True)
    ds_test = convert_dataset(test_dataset)
    torch.save(ds_test, os.path.join(data_dir, "ds_test.pt"))

    # Calculate split fraction for 60:20:20 split (train/validation/test)
    ds_trainval = convert_dataset(train_dataset)
    split_fraction = (len(ds_trainval) - len(ds_test)) / len(ds_trainval)

    # Single train-validation split
    stratified_split(ds_trainval, split_fraction, save=save, save_path=data_dir)

    # Multiple train-validation splits
    for i in range(n_splits):
        stratified_split(ds_trainval, split_fraction, multiple=True, n_split=i, save=save, save_path=data_dir)
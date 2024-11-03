from collections import defaultdict
import numpy as np
import os
import random
import tonic
import torch
from torch.utils.data import TensorDataset
import tonic.transforms as transforms

def convert_dataset(events_dataset):

    x_dataset = []
    y_dataset = []

    for events,label in iter(events_dataset):

        x_dataset.append(events)
        y_dataset.append(label)

    return TensorDataset(torch.as_tensor(np.array(x_dataset)), torch.as_tensor(np.array(y_dataset)))


def stratified_split(dataset : list, fraction, multiple=False, n_split=None, save=False):

    if (type(dataset) == list) & (len(dataset) > 1):
        dataset = torch.utils.data.ConcatDataset(dataset)
        data = torch.cat([ds.tensors[0] for ds in dataset.datasets])
        labels = torch.cat([ds.tensors[1] for ds in dataset.datasets])
        dataset = torch.utils.data.TensorDataset(data, labels)
    else:
        labels = dataset.tensors[1]

    indices_per_label = defaultdict(list)

    for index, label in enumerate(labels):
        indices_per_label[label.item()].append(index)

    first_set_indices, second_set_indices = list(), list()

    for label, indices in indices_per_label.items():
        n_samples_for_label = round(len(indices) * fraction)
        random_indices_sample = random.sample(indices, n_samples_for_label)
        first_set_indices.extend(random_indices_sample)
        second_set_indices.extend(set(indices) - set(random_indices_sample))

    random.shuffle(first_set_indices)
    random.shuffle(second_set_indices)

    first_set = dataset[first_set_indices]
    data_sp1 = first_set[:][0]
    labels_sp1 = first_set[:][1]
    second_set = dataset[second_set_indices]
    data_sp2 = second_set[:][0]
    labels_sp2 = second_set[:][1]

    split1 = torch.utils.data.TensorDataset(data_sp1, labels_sp1)
    split2 = torch.utils.data.TensorDataset(data_sp2, labels_sp2)

    print("Stratified split done with the following results:")
    print(f"\t --- original dataset length: {len(dataset)} ---")
    print(f"\t --- required fractions: {np.round(fraction*100,2)}:{100-np.round(fraction*100,2)}")
    print(f"\t split 1:")
    print(f"\t\t length: {len(split1)}")
    print(f"\t\t labels {torch.unique(split1.tensors[1], return_counts=True)[0].tolist()} with distribution {torch.unique(split1.tensors[1], return_counts=True)[1].tolist()}")
    print(f"\t split 2:")
    print(f"\t\t length: {len(split2)}")
    print(f"\t\t labels {torch.unique(split2.tensors[1], return_counts=True)[0].tolist()} with distribution {torch.unique(split2.tensors[1], return_counts=True)[1].tolist()}")
    print(f"Total number of samples: {len(split1)+len(split2)}")

    if save:

        if not multiple:
            torch.save(split1, "./ds_train.pt")
            torch.save(split2, "./ds_val.pt")

        else:
            torch.save(split1, f"./ds_train_{n_split}.pt")
            torch.save(split2, f"./ds_val_{n_split}.pt")

    else:

        return split1, split2

if __name__ == "__main__":

    seed = 42
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    n_splits = 10
    save = True
    data_path = "./data"
    neurons = 11
    sensor_size = (neurons, 1, 1)
    transform = transforms.Compose([
        transforms.ToFrame(sensor_size=sensor_size, n_time_bins=784),
        transforms.NumpyAsType('float32'),
    ])
    train = tonic.datasets.SMNIST(save_to=data_path, num_neurons=neurons, train=True, duplicate=False, transform=transform) # 1077 tuples (events, label)
    test = tonic.datasets.SMNIST(save_to=data_path, num_neurons=neurons, train=False, duplicate=False, transform=transform) # 264 tuples (events, label) [about 20% of training-test]

    ### Test
    ds_test = convert_dataset(test)
    torch.save(ds_test, "./ds_test.pt")
    ### Training-validation
    ds_trainval = convert_dataset(train)
    fraction = (len(ds_trainval)-len(ds_test)) / len(ds_trainval) # so that the validation set has the same number as the test set --> [60:20:20 split]
    # Single training-validation
    stratified_split(ds_trainval, fraction, save=save)
    # Multiple training-validation
    for ii in range(n_splits):
        stratified_split(ds_trainval, fraction, multiple=True, n_split=ii, save=save)


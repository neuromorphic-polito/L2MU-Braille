import matplotlib.pyplot as plt
import tonic
import torch
from torch.nn.functional import batch_norm
from torch.utils.data import DataLoader, random_split
from lightning import LightningDataModule
import tonic.transforms as transforms


class SMNIST(LightningDataModule):
    def __init__(self, data_dir: str = "../data/smnist", batch_size: int = 32, num_workers: int = 4, n_time_bins: int = 40, seed: int = 42):
        super().__init__()
        self.val_dataset = None
        self.train_dataset = None
        self.test_dataset = None
        self.data_dir = data_dir
        self.batch_size = int(batch_size)
        self.num_workers = num_workers
        self.n_time_bins = n_time_bins
        self.val_split = 0.2
        self.seed = seed
        self.num_inputs = 99
        self.num_outputs = 10
        self.sensor_size= (99, 1, 1)

        self.transform = transforms.Compose([
            transforms.ToFrame(sensor_size=self.sensor_size, n_event_bins=768),
            transforms.NumpyAsType('float32'),
        ])
        #self.transform = None
        self.target_transform = None

    def prepare_data(self):
        tonic.datasets.SMNIST(save_to=self.data_dir, train=True)
        tonic.datasets.SMNIST(save_to=self.data_dir, train=False)

    def setup(self, stage):
        match stage:
            case 'fit':
                full_train_dataset = tonic.datasets.SMNIST(save_to=self.data_dir, train=True,
                                                        transform=self.transform, num_neurons=99, duplicate=False)
                train_size = int((1 - self.val_split) * len(full_train_dataset))
                val_size = len(full_train_dataset) - train_size

                self.train_dataset, self.val_dataset = random_split(full_train_dataset, [train_size, val_size],
                                                                    generator=torch.Generator().manual_seed(self.seed))
            case 'test':
                self.test_dataset = tonic.datasets.SMNIST(save_to=self.data_dir, train=False, num_neurons=99, duplicate=False,
                                                       transform=self.transform, target_transform=self.target_transform)
            case _:
                raise ValueError(f"Stage {stage} not supported")



    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)
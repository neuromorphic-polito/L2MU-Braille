import time
import torch
from torch.utils.data import DataLoader
from lightning import LightningDataModule
import random


class SMNIST(LightningDataModule):
    def __init__(self, data_dir: str = "../data/smnist_splitted", batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.val_dataset = None
        self.train_dataset = None
        self.test_dataset = None
        self.data_dir = data_dir
        self.batch_size = int(batch_size)
        self.num_workers = num_workers
        dataset = torch.load(self.data_dir + '/ds_train.pt', weights_only=False)
        self.num_inputs = dataset[0][0].shape[2]
        self.num_outputs = dataset[:][1].unique().shape[0]

    def setup(self, stage):
        match stage:
            case 'fit':
                # Store the current state
                original_state = random.getstate()
                random.seed(time.time())
                random_number = random.randint(0, 9)
                print(f'Dataset split selected: {random_number}')
                random.setstate(original_state)
                self.train_dataset, self.val_dataset = torch.load(self.data_dir + f'/ds_train_{random_number}.pt', weights_only=False), torch.load(self.data_dir + f'/ds_val_{random_number}.pt', weights_only=False)
            case 'test':
                self.test_dataset = torch.load(self.data_dir + '/ds_test.pt', weights_only=False)
            case _:
                raise ValueError(f"Stage {stage} not supported")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)
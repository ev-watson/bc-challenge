from mlb_requests import Server
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import torch
import random

import config

def one_hot_encode(answer):
    one_hot = np.zeros(len(config.ARCHITECTURES), dtype=np.uint8)
    one_hot[config.LABELS[answer]] = 1
    return one_hot


class BCDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # scale and ensure data is float32 (MPS)
        x = torch.tensor(self.data[idx] / 255.0, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

class BCDataModule(LightningDataModule):
    def __init__(self, batch_size=config.BATCH_SIZE):
        super().__init__()
        self.batch_size = batch_size

        self.data = np.load(config.DATA_PATH, mmap_mode='r')
        self.labels = np.load(config.LABELS_PATH, mmap_mode='r')
        n = len(self.data)

        self.train_data = self.data[:int(config.TRAIN_SPLIT * n)]
        self.train_labels = self.labels[:int(config.TRAIN_SPLIT * n)]

        self.val_data = self.data[int(config.TRAIN_SPLIT * n):int(config.VAL_SPLIT * n)]
        self.val_labels = self.labels[int(config.TRAIN_SPLIT * n):int(config.VAL_SPLIT * n)]

        self.test_data = self.data[int(config.VAL_SPLIT * n):]
        self.test_labels = self.labels[int(config.VAL_SPLIT * n):]

        self.train_dataset = BCDataset(self.train_data, self.train_labels)
        self.val_dataset = BCDataset(self.val_data, self.val_labels)
        self.test_dataset = BCDataset(self.test_data, self.test_labels)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=0,
                          pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=0,
                          pin_memory=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=0,
                          pin_memory=False)


if __name__ == "__main__":
    # s = Server()

    # num_samples = config.NUM_SAMPLES

    # bdata = np.empty((num_samples, 64), dtype=np.uint8)
    # labels = np.empty((num_samples, 12), dtype=np.uint8)

    # for i in tqdm(range(num_samples), desc="Generating data"):
    #     s.get()

    #     bdata[i] = np.frombuffer(s.binary, dtype=np.uint8)

    #     # choose random target just to get the label
    #     target = random.choice(s.targets)
    #     s.post(target)

    #     labels[i] = one_hot_encode(s.ans)


    # np.save('bdata.npy', bdata)
    # np.save('labels.npy', labels)

    data_module = BCDataModule()
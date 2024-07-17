import torch
import pandas as pd
import numpy as np
import tables
from torch.utils.data import Dataset, TensorDataset

class HDF5Dataset(Dataset):
    def __init__(self, table, transform=None):
        self.table = table
        self.transform = transform

        # Retrieve min and max values from table attributes
        self.min_value = self.table.attrs.min_value
        self.max_value = self.table.attrs.max_value

    def __len__(self):
        return self.table.nrows

    def __getitem__(self, idx):
        data = self.table[idx]['data']
        sample = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def set_transform(self, transform):
        self.transform = transform

    def __del__(self):
        self.table._v_file.close()

class Normalize:
    def __init__(self, min_value, max_value, new_min=-1, new_max=1):
        self.min_value = min_value
        self.max_value = max_value
        self.new_min = new_min
        self.new_max = new_max
        self.range = max_value - min_value

    def __call__(self, tensor):
        # Scale tensor to [0, 1]
        tensor = (tensor - self.min_value) / self.range
        # Scale to [new_min, new_max]
        tensor = tensor * (self.new_max - self.new_min) + self.new_min
        return tensor
    
def spec_dataset(dataset, train_table="/train"):
    db = tables.open_file(dataset, mode='r')
    table = db.get_node(train_table + '/data')
    dataset = HDF5Dataset(table, transform=None)
    
    return dataset

def line_dataset(n=8000):
    rng = np.random.default_rng(42)
    x = rng.uniform(-0.5, 0.5, n)
    y = rng.uniform(-1, 1, n)
    X = np.stack((x, y), axis=1)
    X *= 4
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))

def dino_dataset(n=8000):
    df = pd.read_csv("deepechoes/diffusion/DatasaurusDozen.tsv", sep="\t")
    df = df[df["dataset"] == "dino"]

    rng = np.random.default_rng(42)
    ix = rng.integers(0, len(df), n)
    x = df["x"].iloc[ix].tolist()
    x = np.array(x) + rng.normal(size=len(x)) * 0.15
    y = df["y"].iloc[ix].tolist()
    y = np.array(y) + rng.normal(size=len(x)) * 0.15
    x = (x/54 - 1) * 4
    y = (y/48 - 1) * 4
    X = np.stack((x, y), axis=1)
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


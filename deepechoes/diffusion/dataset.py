import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, TensorDataset

class HDF5Dataset(Dataset):
    def __init__(self, table, transform=None):
        self.table = table
        self.transform = transform

    def __len__(self):
        return self.table.nrows

    def __getitem__(self, idx):
        data = self.table[idx]['data']
        sample = torch.tensor(data, dtype=torch.float32)
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

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
import torch
import pandas as pd
import numpy as np
import tables
from torch.utils.data import Dataset, TensorDataset
from torchvision import transforms

def get_leaf_paths(hdf5_file, table_path):
    with tables.open_file(hdf5_file, mode='r') as file:
        group = file.get_node(table_path)
        leaf_paths = []

        if isinstance(group, tables.Leaf):
            leaf_paths.append(table_path)
        elif isinstance(group, tables.Group):
            for node in file.walk_nodes(group, classname='Leaf'):
                leaf_paths.append(node._v_pathname)
                
        return leaf_paths
    
class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file, table_name, transform=None):
        self.hdf5_file = hdf5_file
        self.table_name = table_name
        self.transform = transform
        self.file = tables.open_file(hdf5_file, mode='r')
        self.table = self.file.get_node(table_name)
        self.data = self.table.col('data')
        self.labels = self.table.col('label')

        # Retrieve min and max values from table attributes
        self.min_value = self.table.attrs.min_value if hasattr(self.table.attrs, 'min_value') else None
        self.max_value = self.table.attrs.max_value if hasattr(self.table.attrs, 'max_value') else None
        self.mean_value = self.table.attrs.mean_value if hasattr(self.table.attrs, 'mean_value') else None
        self.std_value = self.table.attrs.std_value if hasattr(self.table.attrs, 'std_value') else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        spectrogram = self.data[idx]
        label = self.labels[idx]        
        
        # Convert to tensor
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            spectrogram = self.transform(spectrogram)

        return spectrogram

    def __del__(self):
        self.file.close()

    def set_transform(self, transform):
        self.transform = transform
    
class NormalizeToRange:
    def __init__(self, min_value=None, max_value=None, new_min=0, new_max=1):
        self.min_value = min_value
        self.max_value = max_value
        self.new_min = new_min
        self.new_max = new_max

    def __call__(self, tensor):
        # If min_value or max_value is not provided, use the tensor's min and max
        min_value = self.min_value if self.min_value is not None else tensor.min()
        max_value = self.max_value if self.max_value is not None else tensor.max()
        range_value = max_value - min_value
        
        # Scale tensor to [0, 1]
        tensor = (tensor - min_value) / range_value
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


def apply_transforms(representation_data_tensor, shape):
    resize_transform = transforms.Resize(shape)
    normalization = NormalizeToRange(new_min=-1, new_max=1)
    representation_data_tensor = resize_transform(representation_data_tensor)
    representation_data_tensor = normalization(representation_data_tensor)
    return representation_data_tensor

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class _BaseTraceDataset(Dataset):
    df: pd.DataFrame
    trace_size: int
    transform = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        plain, key, cipher, trace = self.df.iloc[index, :4]

        if trace.size < self.trace_size:
            trace = np.pad(trace, (0, self.trace_size), constant_values=0)
        else:
            trace = trace[:self.trace_size]

        trace = np.expand_dims(trace, 0)

        if self.transform:
            trace = self.transform(trace)

        sample = {
            'plain': plain,
            'key': key,
            'cipher': cipher,
            'trace': trace
        }

        return sample


class FeatherTraceDataset(_BaseTraceDataset):
    def __init__(self, feather_file, trace_size):
        self.df = pd.read_feather(feather_file)
        self.trace_size = trace_size


class PickleTraceDataset(_BaseTraceDataset):
    def __init__(self, pickle_file, trace_size):
        self.df = pd.read_pickle(pickle_file)
        self.trace_size = trace_size

import os
import os.path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class _BaseTraceDataset(Dataset):
    df: pd.DataFrame
    trace_size: int
    transform = None
    scaler = None

    def __init__(self, trace_size, transform=None, scaler=None):
        self.trace_size = trace_size
        self.transform = transform
        self.scaler = scaler

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        plain, key, cipher, trace = self.df.iloc[index, :4]

        if self.transform:
            trace = self.transform(trace)

        if trace.size < self.trace_size:
            trace = np.pad(trace, (0, self.trace_size - trace.size), constant_values=0)
        else:
            trace = trace[:self.trace_size]

        if self.scaler:
            trace = self.scaler.fit_transform(trace.reshape(-1, 1)).flatten()

        trace = np.expand_dims(trace, 0)

        # key = np.eye(256, dtype=np.uint8)[key[0]]
        key = key[0].astype('int64')

        # sample = {
        #     'plain': plain,
        #     'key': key,
        #     'cipher': cipher,
        #     'trace': trace
        # }
        #
        # return sample

        return trace, key


class FeatherTraceDataset(_BaseTraceDataset):
    def __init__(self, feather_file, *args, **kwargs):
        super(FeatherTraceDataset, self).__init__(*args, **kwargs)
        self.df = pd.read_feather(feather_file)


class PickleTraceDataset(_BaseTraceDataset):
    def __init__(self, pickle_file, *args, **kwargs):
        super(PickleTraceDataset, self).__init__(*args, **kwargs)
        self.df = pd.read_pickle(pickle_file)


class TraceDataset(_BaseTraceDataset):
    def __init__(self, data_folder, *args, **kwargs):
        super(TraceDataset, self).__init__(*args, **kwargs)
        files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.pkl')]
        self.df = pd.concat([pd.read_pickle(f) for f in files])

import os
import numpy as np
import torch
from typing import Tuple
from scipy.signal import butter, filtfilt

def create_lowpass_filter(cutoff: float, fs: float, order: int = 4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def filter_signal(channel_data, b, a):
    return filtfilt(b, a, channel_data)

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", baseline_window: Tuple[int, int] = (0, 100), fs: float = 200.0) -> None:
        super().__init__()

        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        self.baseline_window = baseline_window

        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))

        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

        self.apply_baseline_correction()
        self.filter_data(fs)
        self.apply_standardization()

    def apply_baseline_correction(self):
        baseline = self.X[:, :, self.baseline_window[0]:self.baseline_window[1]].mean(dim=2, keepdim=True)
        self.X -= baseline

    def filter_data(self, fs: float):
        cutoff = 40.0
        b, a = create_lowpass_filter(cutoff, fs)

        for i in range(self.X.shape[1]):
            filtered_channel = filter_signal(self.X[:, i, :].numpy(), b, a)
            self.X[:, i, :] = torch.from_numpy(filtered_channel.copy())

    def apply_standardization(self):
        mean = self.X.mean(dim=2, keepdim=True)
        std = self.X.std(dim=2, keepdim=True)
        self.X = (self.X - mean) / std

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]

    @property
    def num_channels(self) -> int:
        return self.X.shape[1]

    @property
    def seq_len(self) -> int:
        return self.X.shape[2]

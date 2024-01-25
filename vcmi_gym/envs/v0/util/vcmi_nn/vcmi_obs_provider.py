import random
import glob
import torch
import numpy as np

from ..vcmi_env import VcmiEnv

#
#  WIP -- idea was to collect live observations
#  (instead of relying on already stored in files)
#


class Dataset:
    def __init__(self, files):
        self.it_files = iter(files)

    def __iter__(self):
        return self

    def __next__(self):
        return torch.as_tensor(np.load(next(self.it_files)))


class VcmiObsProviderFile:
    def __init__(self, train_test_ratio, deterministic):
        assert isinstance(train_test_ratio, float)
        assert train_test_ratio > 0 and train_test_ratio < 1

        files = glob.glob("data/observations/*.npy")
        if not deterministic:
            random.shuffle(files)
        n_train_files = int(train_test_ratio * len(files))
        self.train_files = files[:n_train_files]
        self.test_files = files[n_train_files:]

    def train_data(self):
        return Dataset(self.train_files)

    def test_data(self):
        return Dataset(self.test_files)


class VcmiObsProviderLive:
    maps_basedir = "/Users/simo/Library/Application Support/vcmi/Maps"

    def __init__(self, mapmask=""):
        env

    def train_data(self):
        return Dataset(self.train_files)

    def test_data(self):
        return Dataset(self.test_files)


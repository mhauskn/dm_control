import torch
import h5py
import numpy as np
from torch.utils.data import Dataset

OBS_KEYS = [
    'walker/joints_pos',
    'walker/joints_vel',
    'walker/appendages_pos',
    'walker/actuator_activation',
    'walker/sensors_accelerometer',
    'walker/sensors_gyro',
    'walker/sensors_torque',
    'walker/sensors_touch',
    'walker/sensors_velocimeter',
    'walker/reference_rel_joints',
]

class TrajectoryDataset(Dataset):
    def __init__(self, h5py_file, block_size):
        dset = h5py.File(h5py_file, 'r')
        self.block_size = block_size
        # Copy the dataset into memory
        # self.observations = np.concatenate([dset['observables/{}'.format(k)][...] for k in OBS_KEYS], axis=1)
        self.actions = dset['actions'][...]
        self.observations = np.eye(self.actions.shape[0], dtype=np.float32)

    def __len__(self):
        return self.actions.shape[0]

    def __getitem__(self, idx):
        x = self.observations[idx]
        y = self.actions[idx]
        return x, y


if __name__ == "__main__":
    d = TrajectoryDataset('small_trajectory_dataset.hdf5', block_size=16)
    print(len(d))
    print(d[0][0].shape)
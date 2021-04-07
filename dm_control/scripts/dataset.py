import torch
import h5py
import numpy as np
from torch.utils.data import Dataset

OBS_KEYS = [
    'walker/actuator_activation',
    'walker/appendages_pos',
    'walker/body_height',
    'walker/end_effectors_pos',
    'walker/joints_pos',
    'walker/joints_vel',
    'walker/sensors_accelerometer',
    'walker/sensors_gyro',
    'walker/sensors_torque',
    'walker/sensors_touch',
    'walker/sensors_velocimeter',
    'walker/world_zaxis',
    # 'walker/clip_id',
    'walker/reference_rel_joints',
    'walker/reference_rel_bodies_pos_global',
    'walker/reference_rel_bodies_quats',
    'walker/reference_rel_bodies_pos_local',
    'walker/reference_ego_bodies_quats',
    'walker/reference_rel_root_quat',
    'walker/reference_rel_root_pos_local',
    'walker/reference_appendages_pos',
    'walker/velocimeter_control',
    'walker/gyro_control',
    'walker/joints_vel_control',
    # 'walker/time_in_clip',
]

class TrajectoryDataset(Dataset):
    def __init__(self, h5py_file, block_size):
        dset = h5py.File(h5py_file, 'r')
        self.block_size = block_size
        # Copy the dataset into memory
        self.observations = np.concatenate([dset['observables/{}'.format(k)][...] for k in OBS_KEYS], axis=1)
        self.actions = dset['actions'][...]
        # self.observations = np.eye(self.actions.shape[0], dtype=np.float32)

    def __len__(self):
        return self.actions.shape[0]

    def __getitem__(self, idx):
        x = self.observations[idx]
        y = self.actions[idx]
        return x, y


if __name__ == "__main__":
    d = TrajectoryDataset('data/complete.hdf5', block_size=16)
    print(len(d))
    print(d[0][0].shape)
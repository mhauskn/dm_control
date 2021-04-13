import torch
import h5py
import numpy as np
from absl import logging
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
    'walker/clip_id',
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
    'walker/time_in_clip',
]

class TrajectoryDataset(Dataset):
    def __init__(self, h5py_file, block_size, observables):
        logging.info(f'Loading dataset from: {h5py_file}')
        dset = h5py.File(h5py_file, 'r')
        self.block_size = block_size
        # Assemble the observables
        self.observables = []
        for o in observables:
            if not o.strip():
                continue
            if not o.startswith('walker/'):
                o = 'walker/' + o
            assert o in OBS_KEYS, f"Unrecognized Observable: {o}"
            self.observables.append(o)
        # Sort the list of observables so that we are robust against changes in order
        sorted(self.observables)
        logging.info(f'Observables: {self.observables}')

        # Copy the dataset into memory
        self.observations = np.concatenate([dset[f'observables/{k}'][...] for k in self.observables], axis=1)
        self.actions = dset['actions'][...]
        # self.observations = np.eye(self.actions.shape[0], dtype=np.float32)

    @property
    def observation_size(self):
        """ Dimension of each observation vector. """
        return self.observations.shape[1]

    @property
    def action_size(self):
        """ Dimension of each action vector. """
        return self.actions.shape[1]

    def __len__(self):
        return self.actions.shape[0] - self.block_size

    def __getitem__(self, idx):
        x = self.observations[idx:idx + self.block_size]
        y = self.actions[idx:idx + self.block_size]
        return x, y


if __name__ == "__main__":
    d = TrajectoryDataset('data/complete.hdf5', block_size=4)
    print(len(d))
    print(d[0])
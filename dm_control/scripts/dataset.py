import torch
import h5py
import numpy as np
import bisect
import random
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
        self.dones = dset['dones'][...]

        self._remove_short_episodes()
        self._create_logical_offset()


    def _remove_short_episodes(self):
        """ Removes all episodes shorter than block_size. """
        all_obs = []
        all_acts = []
        all_dones = []
        episode_ends = np.nonzero(self.dones)[0]
        episode_start = 0
        episodes_removed = 0
        for episode_end in episode_ends:
            ep_length = episode_end - episode_start + 1
            if ep_length >= self.block_size:
                all_obs.append(self.observations[episode_start: episode_end+1])
                all_acts.append(self.actions[episode_start: episode_end+1])
                all_dones.append(self.dones[episode_start: episode_end+1])
            else:
                episodes_removed += 1
            episode_start = episode_end + 1
        self.observations = np.concatenate(all_obs)
        self.actions = np.concatenate(all_acts)
        self.dones = np.concatenate(all_dones)
        logging.info(f"Removed {episodes_removed} episodes shorter than {self.block_size} steps.")


    def _create_logical_offset(self):
        """ The idea behind the logical offset is to avoid sampling data that crosses episode 
        boundaries. The strategy is to avoid sampling a datapoint in the tail of an episode
        (denoted by |ooooo|) as it would cross an episode boundary when adding context.

        Actual Dataset: here shown with 4 episodes, heads + tails.
        |-----------|ooooo| |-----|ooooo| |---------------|ooooo| |oooo|
        
        Logical Dataset: contains only the heads of episodes - so that we never sample from the 
        tail of an episode (and cross an episode boundary).
        |-----------|       |-----|       |---------------|       ||
        
        The logical offset tells us for an index into the logical dataset, the corresponding
        index in the actual dataset.

        For example, if we wanted to retrieve the first timestep of Episode 2, we would need to 
        offset the logical index by the tail length of Episode 1 to arrive at an index into the
        actual dataset.
        
        """
        self.logical_index, self.logical_offset = [-1], [0, 0]
        episode_ends = np.nonzero(self.dones)[0]
        episode_start = 0
        head_sum, tail_sum = 0, 0
        for idx, episode_end in enumerate(episode_ends):
            ep_length = episode_end - episode_start + 1
            assert ep_length >= self.block_size
            tail_start = (episode_end+1) - self.block_size + 1
            head_steps = tail_start - episode_start
            tail_steps = (episode_end+1) - tail_start
            assert tail_steps == self.block_size - 1
            assert head_steps + tail_steps == ep_length
            head_sum += head_steps
            tail_sum += tail_steps
            self.logical_index.append(head_sum-1)
            self.logical_offset.append(tail_sum)
            episode_start = episode_end + 1
        assert head_sum + tail_sum == self.dones.shape[0]
        self.total_len = head_sum


    @property
    def observation_size(self):
        """ Dimension of each observation vector. """
        return self.observations.shape[1]

    @property
    def action_size(self):
        """ Dimension of each action vector. """
        return self.actions.shape[1]

    def __len__(self):
        return self.total_len
        # return self.actions.shape[0] - self.block_size

    def __getitem__(self, idx):
        """ Given the logical idx, we need to find the offset to arrive at the 
        actual index into the dataset.

        """
        z = bisect.bisect_left(self.logical_index, idx)
        offset = self.logical_offset[z]
        start_idx = idx + offset
        end_idx = start_idx + self.block_size

        # If we've sampled an episode termination, ensure it's at the final step
        dones = self.dones[start_idx: end_idx]
        s = sum(dones)
        assert s == 0 or (s == 1 and dones[-1] == True)

        x = self.observations[start_idx: end_idx]
        y = self.actions[start_idx: end_idx]
        return x, y


if __name__ == "__main__":
    block_size = 64
    d = TrajectoryDataset('data/complete.hdf5', block_size=block_size, observables=['joints_pos', 'joints_vel'])
    # d[193]

    N = len(d)
    for idx in range(N):
        # n = random.randint(0, N-1)
        x, y = d[idx]
        assert x.shape[0] == block_size
        assert y.shape[0] == block_size
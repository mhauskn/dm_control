import os
import numpy as np
import h5py
from os.path import join as pjoin
from absl import app
from absl import flags
from absl import logging
from solver import build_env

# This script processes saved actions sequences into a (state, action) 
# dataset for downstream consumption.

logging.set_verbosity(logging.INFO)

FLAGS = flags.FLAGS
flags.DEFINE_list("input_dirs", ".", "List of directories to gather actions from.")
flags.DEFINE_string("output_path", "trajectory_dataset.hdf5", "Output file for the dataset.")

OBS_KEYS = [
    'walker/joints_pos',
    'walker/joints_vel',
    'walker/appendages_pos',
    'walker/body_height',
    'walker/actuator_activation',
    'walker/sensors_accelerometer',
    'walker/sensors_gyro',
    'walker/sensors_torque',
    'walker/sensors_touch',
    'walker/sensors_velocimeter',
]

def parse_clip_name(stdout_file):
    """ Parses the name of the clip from the stdout.txt of the job's run. """
    with open(stdout_file, 'r') as f:
        for line in f.readlines():
            if 'FLAGS.clip_name' in line:
                return line.split()[-1]
    raise Exception("Unable to identify clip_name in: '{}'".format(stdout_file))


def extract_data(job_dir):
    """ Extracts (observations, actions) from a single job directory. """
    try:
        actions = np.load(pjoin(job_dir, 'opt_acts_0.npy'))
        clip_name = parse_clip_name(pjoin(job_dir, 'stdout.txt'))
        logging.debug('Parsed clip name: {}'.format(clip_name))
    except:
        raise

    env = build_env(
        reward_type="termination",
        clip_name=clip_name
    )
    return run_episode(env, actions)


def run_episode(env, actions):
    """ Re-runs the episode, recording the observations and actions. """
    time_step = env.reset()
    observables = {key: [] for key in OBS_KEYS}
    for idx, act in enumerate(actions):
        for k in OBS_KEYS:
            features = time_step.observation[k].astype(np.float32)
            observables[k].append(features)
        time_step = env.step(act)
        if time_step.last():
            break
    concat_obs = { k: np.concatenate(v) for k,v in observables.items() }
    concat_act = actions[:idx+1].astype(np.float32)
    return concat_obs, concat_act


def create_dataset(argv):
    """ To create a dataset:
    
    1) Look through individual jobs in the experiment folder to find 
    optimized_actions.npy files.
    2) Build the corresponding environment based on the intended clip 
    and starting position.
    3) Record states and actions into a ? datastructure.
    4) Save the dataset.

    """
    all_observations, all_actions, all_dones = [], [], []
    for exp_dir in FLAGS.input_dirs:
        for job_dir in next(os.walk(exp_dir))[1]:
            try:
                observations, actions = extract_data(pjoin(exp_dir, job_dir))
                # Record episode terminations in dones.
                dones = np.zeros((len(actions)), dtype=np.bool)
                dones[-1] = True
                all_observations.append(observations)
                all_actions.append(actions)
                all_dones.append(dones)
            except Exception as e:
                logging.warning(e)

    # Concatenate everything
    all_actions_np = np.concatenate(all_actions)
    all_dones_np = np.concatenate(all_dones)
    tmp = {k: [] for k in all_observations[0].keys()}
    for obs_dict in all_observations:
        for k, v in obs_dict.items():
            tmp[k].append(v)
    all_observations_np = {k: np.concatenate(v) for k,v in tmp.items()}

    # Write the dataset to a hdf5 file
    f = h5py.File(FLAGS.output_path, "w")
    act_dset = f.create_dataset("actions", all_actions_np.shape, all_actions_np.dtype)
    act_dset[...] = all_actions_np
    done_dset = f.create_dataset("dones", all_dones_np.shape, all_dones_np.dtype)
    done_dset[...] = all_dones_np
    grp = f.create_group("observables")
    for k,v in all_observations_np.items():
        dset = grp.create_dataset(k, v.shape, v.dtype)
        dset[...] = v


def load_dataset(argv):
    f = h5py.File(FLAGS.output_path, 'r')
    f.keys()

if __name__ == "__main__":
    app.run(create_dataset)
    # app.run(load_dataset)
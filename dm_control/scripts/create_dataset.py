import os
import math
import numpy as np
import h5py
from os.path import join as pjoin
from absl import app
from absl import flags
from absl import logging
from solver import build_env, evaluate
from tqdm import tqdm

# This script processes saved actions sequences into a (state, action) 
# dataset for downstream consumption.

logging.set_verbosity(logging.INFO)

FLAGS = flags.FLAGS
flags.DEFINE_list("input_dirs", ".", "List of directories to gather actions from.")
flags.DEFINE_string("output_path", "trajectory_dataset.hdf5", "Output file for the dataset.")

TERMINATION_ERROR_THRESHOLD=0.3

def parse_clip_name(stdout_file):
    """ Parses the name of the clip from the stdout.txt of the job's run. """
    with open(stdout_file, 'r') as f:
        for line in f.readlines():
            if 'FLAGS.clip_name' in line:
                return line.split()[-1]
    raise Exception("Unable to identify clip_name in: '{}'".format(stdout_file))

def parse_start_step(stdout_file):
    """ Parses the starting_step of the clip from the stdout.txt of the job's run. """
    with open(stdout_file, 'r') as f:
        for line in f.readlines():
            if 'FLAGS.start_step' in line:
                return int(line.split()[-1])
    # Otherwise return a default start_step of 0 for jobs that were run before we added this flag
    return 0

def parse_final_performance(stdout_file):
    """ Parses the Jfin of the clip from the stdout.txt of the job's run. """
    with open(stdout_file, 'r') as f:
        for line in f.readlines():
            if 'Optimization Pass Complete: Jini=' in line:
                s = line.split()[-3]
                return float(s.split("=")[-1])
    logging.warning(f"Unable to identify Jfin in: {stdout_file}")
    return 0.


def extract_data(job_dir):
    """ Extracts (observations, actions) from a single job directory. """
    try:
        stdout_path = pjoin(job_dir, 'stdout.txt')
        actions = np.load(pjoin(job_dir, 'opt_acts_0.npy'))
        clip_name = parse_clip_name(stdout_path)
        start_step = parse_start_step(stdout_path)
        expected_JFin = parse_final_performance(stdout_path)
        logging.debug(f'Parsed {job_dir}: clip_name: {clip_name} start_step: {start_step} Jfin: {expected_JFin:.3f}')
    except:
        raise

    env = build_env(
        reward_type="termination",
        clip_name=clip_name,
        start_step=start_step,
        termination_error_threshold=TERMINATION_ERROR_THRESHOLD,
        disable_observables=False,
    )
    # Jfin = evaluate(env, actions)
    # if not math.isclose(Jfin, expected_JFin, abs_tol=.001):
    #     logging.info(f'Divergence {job_dir}: Expected {expected_JFin} Actual {Jfin}')
    # return None, None
    episode_success, Jfin, concat_obs, concat_act = run_episode(env, actions)
    if episode_success:
        if expected_JFin > 0:
            assert math.isclose(Jfin, expected_JFin, abs_tol=.001)
        return concat_obs, concat_act
    else:
        logging.debug(f'Detected early termination: {stdout_path}')
        return None, None


def run_episode(env, actions):
    """ Runs the episode, recording the observations and actions. 
        Returns: Success (boolean), observations, actions.
    """
    time_step = env.reset()
    observables = {key: [] for key, v in time_step.observation.items() if v.size > 0}
    J = 0
    for idx, act in enumerate(actions):
        for k,v in time_step.observation.items():
            if v.size > 0:
                features = np.array(v, dtype=np.float32, copy=True)
                # Expand to make sure all observables have at least 2 dimensions
                if features.ndim < 2:
                    features = features[:, np.newaxis]
                observables[k].append(features)
        time_step = env.step(act)
        J += time_step.reward
        if env._task._termination_error >= TERMINATION_ERROR_THRESHOLD:
            return False, J, None, None
        if time_step.last():
            break
    concat_obs = { k: np.concatenate(v) for k,v in observables.items() }
    concat_act = np.array(actions[:idx+1], dtype=np.float32, copy=True)
    return True, J, concat_obs, concat_act

def check_for_finished_jobs(input_dirs):
    """ Checks each experiment directory for a stdout.txt file and a opt_acts_0.npy action file.
        Returns a list of experiment directories that are missing either of these.
    """
    finished_jobs, failed_jobs = [], []
    for exp_dir in input_dirs:
        for job_dir in next(os.walk(exp_dir))[1]:
            full_path = pjoin(exp_dir, job_dir)
            if os.path.exists(pjoin(full_path, 'stdout.txt')) and \
                os.path.exists(pjoin(full_path, 'opt_acts_0.npy')):
                finished_jobs.append(full_path)
            else:
                # logging.info(f'Unfinished job: {full_path} is missing required files.')
                failed_jobs.append(full_path)
    return finished_jobs, failed_jobs

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
    early_terminations = []
    finished_jobs, failed_jobs = check_for_finished_jobs(FLAGS.input_dirs)
    # logging.info(f'Detected {len(finished_jobs)} finished jobs')
    pbar = tqdm(finished_jobs)
    for job_dir in pbar:
        pbar.set_description(f"{job_dir}")
        observations, actions = extract_data(job_dir)
        # Ignore clips that resulted in early termination / failure
        if observations == None:
            early_terminations.append(job_dir)
            continue
        # Record episode terminations in dones.
        dones = np.zeros((len(actions)), dtype=np.bool)
        dones[-1] = True
        all_observations.append(observations)
        all_actions.append(actions)
        all_dones.append(dones)

    for job_dir in failed_jobs:
        logging.info(f'[Failed Job] {job_dir}')
    for job_dir in early_terminations:
        logging.info(f'[Early Termination] {job_dir}')

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
    logging.info(f'[Finished!] Jobs: {len(finished_jobs)}/{len(finished_jobs)+len(failed_jobs)} successful; {len(early_terminations)} early terminations;')
    logging.info(f'Action Shape: {all_actions_np.shape}')
    for k,v in all_observations_np.items():
        logging.info(f'Observation {k} Shape: {v.shape}')

def load_dataset(argv):
    f = h5py.File(FLAGS.output_path, 'r')
    print(f.keys())
    print(f['actions'].shape)

if __name__ == "__main__":
    app.run(create_dataset)
    # app.run(load_dataset)
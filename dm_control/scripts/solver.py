import os
import tree
import numpy as np
import functools
import math
import time
from google.protobuf import descriptor
from google.protobuf import text_format
from dm_control import composer
from dm_control.locomotion.mocap import loader
from dm_control.locomotion.mocap import mocap_pb2
from dm_control.locomotion.mocap import trajectory
from dm_control import mjcf
from dm_control.utils import io as resources
from dm_control.locomotion.mocap import cmu_mocap_data
from dm_control.locomotion.walkers import cmu_humanoid
from dm_control.locomotion.arenas import floors
from dm_control.locomotion.tasks.reference_pose import tracking
from dm_control.locomotion.tasks.reference_pose import types
from dm_control.locomotion.tasks.reference_pose import utils
from dm_control.mujoco.wrapper import mjbindings
from dm_control.viewer import application
from dm_control.viewer import renderer
from dm_control import _render
from dm_control import viewer
import matplotlib.pyplot as plt
from tqdm import trange
from scipy import optimize
import multiprocessing
from collections import namedtuple
from absl import app
from absl import flags
from absl import logging

mjlib = mjbindings.mjlib

TERMINATION_ERROR_THRESHOLD = 0.3

OUTPUT_DIR = os.environ.get('PT_OUTPUT_DIR', '.')
DATA_DIR = os.environ.get('PT_DATA_DIR', '.')

CustomInit = namedtuple('CustomInit', ['clip_index', 'start_step', 'physics_state'])

FLAGS = flags.FLAGS
flags.DEFINE_integer("additional_segs", 0, "Additional trajectory segments to add.")
flags.DEFINE_string("reward_type", "kumquat", "Name of reward function.")
flags.DEFINE_integer("seg_size", 8, "Size of segments to optimize.")
flags.DEFINE_string("clip_name", "CMU_016_22", "Name of reference clip. See cmu_subsets.py")
flags.DEFINE_string("load_actions_path", None, "Path relative to DATA_DIR to load actions from.")
flags.DEFINE_integer("optimizer_iters", 1, "Max iterations of Scipy optimizer.")
flags.DEFINE_integer("optimization_passes", 1, "Number of optimization passes to perform.")


def get_trajectory_guess(env, custom_init):
    """ Returns an trajectory based on the provided actions and target reference poses. """
    env.task.set_custom_init(custom_init)
    time_step = env.reset()
    actions = []
    act_minimum = env.action_spec().minimum
    act_maximum = env.action_spec().maximum
    physics_states = []
    J = 0
    while True:
        state = env._physics.get_state().copy()
        physics_states.append(state)
        target_joints = env.task.get_reference_features()['joints']
        act = env.task._walker.cmu_pose_to_actuation(target_joints)
        act = np.minimum(np.maximum(act, act_minimum), act_maximum)
        actions.append(np.copy(act))
        time_step = env.step(act)
        J += time_step.reward
        if time_step.last():
            break
    return np.array(actions), physics_states, J


def evaluate(env, actions, custom_init):
    """ Resets the environment and executes the provided actions. """
    J = 0
    env.task.set_custom_init(custom_init)
    env.reset_to_physics_state() # env.reset()
    for act in actions:
        time_step = env.step(act)
        J += time_step.reward
        if time_step.last():
            break
    return J


def evaluate_with_physics_states(env, actions, custom_init):
    """ Resets the environment and executes the provided actions. """
    J = 0
    env.task.set_custom_init(custom_init)
    env.reset_to_physics_state() # env.reset()
    physics_states = []
    for act in actions:
        physics_states.append(env._physics.get_state().copy())
        time_step = env.step(act)
        J += time_step.reward
        if time_step.last():
            break
    return J, physics_states


def episode_failed(env, actions, custom_init):
    env.task.set_custom_init(custom_init)
    env.reset_to_physics_state()
    for act in actions:
        time_step = env.step(act)
        if env._task._termination_error >= TERMINATION_ERROR_THRESHOLD:
            return True
        if time_step.last():
            break
    return False
    

# def optimize_clip_segment(env, actions, custom_init, optimizer_iters, additional_actions=None):
#     start_step = custom_init.start_step
#     logging.info('Optimized Actions {}-{}'.format(start_step, start_step+len(actions)))
#     return actions+.01

def optimize_clip_segment(env, actions, custom_init, optimizer_iters, additional_actions=None):
    """ Optimizes the provided actions.

    Args:
        actions: Numpy ndarray of shape (n_steps x action_dim) of actions to optimize.
        custom_init: Specifier 
        optimizer_iters: Number of iterations of Scipy optimzier to run.
        additional_actions: (Optional) extra actions appended to the original
            actions for purposes of evaluation, but not optimized.

    Returns:
        Numpy ndarray of optimized actions.
    """
    start = time.time()
    full_actions = np.concatenate((actions, additional_actions)) \
        if additional_actions is not None else actions
    J_init = evaluate(env, full_actions, custom_init)
    x0 = actions.flatten()
    if additional_actions is not None:
        fun = lambda x: evaluate(env, 
                                  np.concatenate((x.reshape(actions.shape), additional_actions)), 
                                  custom_init)
    else:
        fun = lambda x: evaluate(env, x.reshape(actions.shape), custom_init)

    res = optimize.minimize(
        fun,
        x0, 
        method='powell',
        bounds=optimize.Bounds(lb=-np.ones_like(x0), ub=np.ones_like(x0)),
        options={
            'disp': True,
            'maxiter': optimizer_iters,
        }
    )
    opt_actions = res.x.reshape(actions.shape)
    J_fin = res.fun
    end = time.time()
    logging.info('Optimized Actions {}-{}. Jini={:.3f} Jfin={:.3f} elapsedTime(s)={:.1f}'.format(
        custom_init.start_step, custom_init.start_step+len(actions), J_init, J_fin, end-start))
    return opt_actions


def build_env(reward_type, ghost_offset=0, clip_name='CMU_016_22'):
    walker = cmu_humanoid.CMUHumanoidPositionControlledV2020
    arena = floors.Floor()
    task = tracking.MultiClipMocapTracking(
        walker=walker,
        arena=arena,
        ref_path=cmu_mocap_data.get_path_for_cmu_2020(),
        dataset=types.ClipCollection(ids=[clip_name]),
        ref_steps=(1, 2, 3, 4, 5),
        min_steps=10,
        reward_type=reward_type,
        always_init_at_clip_start=True,
        termination_error_threshold=1e10,
        ghost_offset=ghost_offset,
    )
    env = composer.Environment(time_limit=30,
                                task=task,
                                random_state=None,
                                strip_singleton_obs_buffer_dim=True)
    return env


def singlethreaded_optimize(env, actions, custom_init, optimizer_iters, seg_size, additional_segs):
    """ Optimizes a sequence of actions starting at step start_step in the clip.
        Breaks actions it into subsequences of size seg_size and iteratively optimizes
        each subsequence.

        Args:
            env: the environment.
            actions: ndarray of (n_steps x action_dim) of actions to optimize.
            start_step: offset from beginning of clip to start optimizing.
            optimizer_iters: Number of optimizer iterations to run.
            init_physics_state: mujoco physics state associated with clip[start_step].
            seg_size: optimize segments of size seg_size.

        Returns:
            The return J of the optimized sequences and the optimized actions.
    """
    start = time.time()
    optimized_actions = np.copy(actions)
    Jini, physics_states = evaluate_with_physics_states(env, optimized_actions, custom_init)

    n_segs = math.ceil(len(actions) / seg_size)
    for seg_idx in range(n_segs):
        ts = seg_idx * seg_size
        start_step = custom_init.start_step + ts
        cInit = CustomInit(0, start_step, physics_states[ts])
        add_acts = optimized_actions[ts + seg_size: ts + (additional_segs+1)*seg_size] \
            if additional_segs > 0 else None
        opt_actions = optimize_clip_segment(
            env,
            actions=optimized_actions[ts: ts + seg_size],
            optimizer_iters=optimizer_iters,
            custom_init=cInit,
            additional_actions=add_acts)
        optimized_actions[ts: ts + seg_size] = opt_actions
        J, physics_states = evaluate_with_physics_states(env, optimized_actions, custom_init)
        if episode_failed(env, opt_actions, cInit):
            logging.info('Exiting early due to termination.')
            break

    Jfin = evaluate(env, optimized_actions, custom_init)
    end = time.time()
    logging.info('Optimization Pass Complete: Jini={:.3f} Jfin={:.3f} len(Actions)={} elapsedTime(s)={:.1f}'.format(
        Jini, Jfin, len(actions), end-start))

    return J, optimized_actions


def log_flags(flags):
    """ Logs the value of each of the flags. """
    for k in dir(flags):
        if k != '?':
            flag = 'FLAGS.{}'.format(k)
            logging.info('{}: {}'.format(flag, eval(flag)))


def main(argv):
    log_flags(FLAGS)
    env = build_env(reward_type=FLAGS.reward_type,
                    clip_name=FLAGS.clip_name)
    env.reset()
    init_physics_state = env._physics.get_state().copy()
    cInit = CustomInit(clip_index=0, start_step=0, physics_state=init_physics_state)

    if FLAGS.load_actions_path:
        fname = os.path.join(DATA_DIR, FLAGS.load_actions_path)
        logging.info('Loading actions from: {}'.format(fname))
        actions = np.load(fname)
        Jinit, physics_states = evaluate_with_physics_states(env, actions, cInit)
    else:
        actions, physics_states, Jinit = get_trajectory_guess(env, cInit)

    for idx in range(FLAGS.optimization_passes):
        logging.info('Starting optimization pass {}: Jinit={:.3f} Length={}'.format(idx, Jinit, len(actions)))

        Jfin, optimized_actions = singlethreaded_optimize(
            env, actions, cInit, FLAGS.optimizer_iters, FLAGS.seg_size, FLAGS.additional_segs)

        fname = "opt_acts_{}.npy".format(idx)
        logging.info('Saving actions to {}'.format(fname))
        np.save(os.path.join(OUTPUT_DIR, fname), optimized_actions)

        actions = np.copy(optimized_actions)
        Jinit, physics_states = evaluate_with_physics_states(env, actions, cInit)


if __name__ == "__main__":
    app.run(main)
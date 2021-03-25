import os
import tree
import numpy as np
import functools
import math
import time
import resource
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
from dm_control.mujoco.wrapper.mjbindings import mjlib
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

CustomInit = namedtuple('CustomInit', ['start_step', 'physics_data'])

FLAGS = flags.FLAGS
flags.DEFINE_integer("additional_segs", 0, "Additional trajectory segments to add.")
flags.DEFINE_string("reward_type", "kumquat", "Name of reward function.")
flags.DEFINE_integer("seg_size", 8, "Size of segments to optimize.")
flags.DEFINE_string("clip_name", "CMU_016_22", "Name of reference clip. See cmu_subsets.py")
flags.DEFINE_string("load_actions_path", None, "Path relative to DATA_DIR to load actions from.")
flags.DEFINE_integer("optimizer_iters", 1, "Max iterations of Scipy optimizer.")
flags.DEFINE_integer("optimization_passes", 1, "Number of optimization passes to perform.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_float("force_magnitude", 0, "Magnitude of force applied to walker.")
flags.DEFINE_boolean("disable_observables", True, "Disables observations for faster run speed.")


def set_task_state(env, start_step: int, physics_data: 'wrapper.MjData'):
    """ Sets the state of the tracking task to a physics state & timestep. """
    env._physics.free()
    # Set the physics state of the environment
    env._physics._reload_from_data(physics_data.deepcopy())
    env._hooks._episode_step_count = 0
    env._reset_next_step = False
    # Set the tracking task to a particular timestep and update features
    env.task.set_tracking_state_and_update(
        physics=env.physics, clip_index=0, start_step=start_step)


def evaluate_and_get_physics_data(env, actions, custom_init=None):
    """ Resets the environment and executes the provided actions. 
        Returns the total reward and the sequence of physics_data.
    """
    J = 0
    if custom_init:
        set_task_state(env, custom_init.start_step, custom_init.physics_data)
    else:
        env.reset()
    physics_data = []
    for act in actions:
        physics_data.append(env.physics.data.deepcopy())
        time_step = env.step(act)
        J += time_step.reward
        if time_step.last():
            break
    return J, physics_data


def get_trajectory_guess(env, custom_init=None):
    """ Returns an initial guess at an action trajectory based on inverse dynamics. """
    if custom_init:
        set_task_state(env, custom_init.start_step, custom_init.physics_data)
    else:
        env.reset()
    actions = []
    act_minimum = env.action_spec().minimum
    act_maximum = env.action_spec().maximum
    while True:
        target_joints = env.task.get_reference_features()['joints']
        act = env.task._walker.cmu_pose_to_actuation(target_joints)
        act = np.minimum(np.maximum(act, act_minimum), act_maximum)
        actions.append(np.copy(act))
        time_step = env.step(act)
        if time_step.last():
            break
    return np.array(actions)


def evaluate(env, actions, custom_init=None):
    """ Resets the environment and executes the provided actions. """
    if custom_init:
        set_task_state(env, custom_init.start_step, custom_init.physics_data)
    else:
        env.reset()
    J = 0
    for act in actions:
        time_step = env.step(act)
        J += time_step.reward
        if time_step.last():
            break
    return J


def episode_failed(env, actions, custom_init=None):
    """ Returns True if error exceeds termination threshold on any step. """
    if custom_init:
        set_task_state(env, custom_init.start_step, custom_init.physics_data)
    else:
        env.reset()
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
#     return actions+.01, custom_init.physics_data.deepcopy()

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
            'disp': False,
            'maxiter': optimizer_iters,
        }
    )
    opt_actions = res.x.reshape(actions.shape)
    J_fin = res.fun
    end = time.time()
    logging.info('Optimized Actions {}-{}. Jini={:.3f} Jfin={:.3f} elapsedTime(s)={:.1f}'.format(
        custom_init.start_step, custom_init.start_step+len(actions), J_init, J_fin, end-start))
    evaluate(env, opt_actions, custom_init)
    final_state = env.physics.data.deepcopy()
    return opt_actions, final_state


def build_env(reward_type, ghost_offset=0, clip_name='CMU_016_22', proto_modifier=None,
              force_magnitude=0, disable_observables=True):
    walker = cmu_humanoid.CMUHumanoidPositionControlledV2020
    arena = floors.Floor()
    task = tracking.MultiClipMocapTracking(
        walker=walker,
        arena=arena,
        ref_path=cmu_mocap_data.get_path_for_cmu_2020(),
        dataset=types.ClipCollection(ids=[clip_name]),
        ref_steps=(1, 2, 3, 4, 5),
        min_steps=10,
        max_steps=256,
        reward_type=reward_type,
        always_init_at_clip_start=True,
        termination_error_threshold=1e10,
        proto_modifier=proto_modifier,
        ghost_offset=ghost_offset,
        force_magnitude=force_magnitude,
        disable_observables=disable_observables,
    )
    env = composer.Environment(
        task=task, 
        random_state=np.random.RandomState(seed=FLAGS.seed)
    )
    return env


def singlethreaded_optimize(env, actions, optimizer_iters, seg_size, additional_segs):
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
    Jini = evaluate(env, actions)
    optimized_actions = np.copy(actions)
    time_step = env.reset()
    physics_state = env.physics.data.deepcopy()
    n_segs = math.ceil(len(actions) / seg_size)
    
    for seg_idx in range(n_segs):
        logging.info('Memory Usage: {:.1f} Mb'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.))
        ts = seg_idx * seg_size
        cInit = CustomInit(ts, physics_state)
        add_acts = optimized_actions[ts + seg_size: ts + (additional_segs+1)*seg_size] \
            if additional_segs > 0 else None
        opt_actions, new_physics_state = optimize_clip_segment(
            env,
            actions=optimized_actions[ts: ts + seg_size],
            optimizer_iters=optimizer_iters,
            custom_init=cInit,
            additional_actions=add_acts)
        optimized_actions[ts: ts + seg_size] = opt_actions
        if episode_failed(env, opt_actions, cInit):
            logging.info('Exiting early due to termination.')
            break
        physics_state.free()
        physics_state = new_physics_state

    physics_state.free()
    Jfin = evaluate(env, optimized_actions)
    end = time.time()
    logging.info('Optimization Pass Complete: Jini={:.3f} Jfin={:.3f} len(Actions)={} elapsedTime(s)={:.1f}'.format(
        Jini, Jfin, len(actions), end-start))

    return optimized_actions


def log_flags(flags):
    """ Logs the value of each of the flags. """
    for k in dir(flags):
        if k != '?':
            flag = 'FLAGS.{}'.format(k)
            logging.info('{}: {}'.format(flag, eval(flag)))


def main(argv):
    log_flags(FLAGS)
    env = build_env(
        reward_type=FLAGS.reward_type,
        clip_name=FLAGS.clip_name,
        force_magnitude=FLAGS.force_magnitude,
        disable_observables=FLAGS.disable_observables,
    )

    if FLAGS.load_actions_path:
        fname = os.path.join(DATA_DIR, FLAGS.load_actions_path)
        logging.info('Loading actions from: {}'.format(fname))
        actions = np.load(fname)
    else:
        actions = get_trajectory_guess(env)

    for idx in range(FLAGS.optimization_passes):
        J = evaluate(env, actions)
        logging.info('Starting optimization pass {}: Jinit={:.3f} Length={}'.format(idx, J, len(actions)))

        optimized_actions = singlethreaded_optimize(
            env, actions, FLAGS.optimizer_iters, FLAGS.seg_size, FLAGS.additional_segs)

        fname = "opt_acts_{}.npy".format(idx)
        logging.info('Saving actions to {}'.format(fname))
        np.save(os.path.join(OUTPUT_DIR, fname), optimized_actions)

        actions = np.copy(optimized_actions)


if __name__ == "__main__":
    app.run(main)
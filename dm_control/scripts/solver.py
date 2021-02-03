import os
import tree
import numpy as np
import functools
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

mjlib = mjbindings.mjlib

OUTPUT_DIR = os.environ.get('PT_OUTPUT_DIR', '.')

def get_initial_trajectory(env, max_steps):
    """ Returns an initial trajectory based on target reference poses. """
    time_step = env.reset()
    actions = []
    act_minimum = env.action_spec().minimum
    act_maximum = env.action_spec().maximum
    # reference_features = []
    physics_states = []
    J = 0
    for step in range(max_steps):
        # ref_feats = utils.get_features(env.physics, env.task._walker)
        # copied_feats = { k: v.copy() for k, v in ref_feats.items() }
        state = env._physics.get_state().copy()
        physics_states.append(state)
        target_joints = env.task.get_reference_features()['joints']
        act = env.task._walker.cmu_pose_to_actuation(target_joints)
        act = np.minimum(np.maximum(act, act_minimum), act_maximum)
        actions.append(np.copy(act))
        # reference_features.append(ref_feats)
        time_step = env.step(act)
        J += time_step.reward
        if time_step.last():
            break
    # ref_feats = utils.get_features(env.physics, env.task._walker)
    # state = env._physics.get_state()
    return np.array(actions), physics_states, J

def evaluate(env, actions):
    """ Resets the environment and executes the provided actions. """
    J = 0
    env.reset()
    for act in actions:
        time_step = env.step(act)
        J += time_step.reward
        if time_step.last():
            break
    return J

def evaluate_with_physics_states(env, actions):
    """ Resets the environment and executes the provided actions. """
    J = 0
    env.reset()
    physics_states = []
    for act in actions:
        physics_states.append(env._physics.get_state().copy())
        time_step = env.step(act)
        J += time_step.reward
        if time_step.last():
            break
    return J, physics_states

def optimize_clip_segment(start_step, init_physics_state, actions):
    env = build_env()
    env.task.set_custom_init(0, start_step, init_physics_state)    
    # print('Initial Val:', evaluate(env, actions))
    # return actions
    x0 = actions.flatten()
    res = optimize.minimize(
        lambda x: -evaluate(env, x.reshape(actions.shape)), 
        x0, 
        method='powell',
        bounds=optimize.Bounds(lb=-np.ones_like(x0), ub=np.ones_like(x0)),
        options={
            'disp': True,
            'maxiter': 1,
        }
    )
    opt_actions = res.x.reshape(actions.shape)
    # final_perf = res.fun #evaluate(env, opt_actions)
    # final_state = env._physics.get_state().copy()
    return opt_actions

def build_env():
    walker = cmu_humanoid.CMUHumanoidPositionControlledV2020
    arena = floors.Floor()
    task = tracking.MultiClipMocapTracking(
        walker=walker,
        arena=arena,
        ref_path=cmu_mocap_data.get_path_for_cmu_2020(),
        dataset=types.ClipCollection(ids=['CMU_016_22']),
        ref_steps=(1, 2, 3, 4, 5),
        min_steps=10,
        reward_type='comic',
        always_init_at_clip_start=True,
        # termination_error_threshold=1e10,
        # body_error_multiplier=0.,
        # ghost_offset=0,
    )
    env = composer.Environment(time_limit=30,
                                task=task,
                                random_state=None,
                                strip_singleton_obs_buffer_dim=True)
    return env

def main():    
    STEPS = 32
    SEGMENT_SIZE = 4
    OPT_ITERS = 4

    env = build_env()
    actions, physics_states, J = get_initial_trajectory(env, max_steps=STEPS)
    print('Initial Trajectory Return:', J)

    # newJ = 0
    # for ts in range(steps):
    #     env.task.set_custom_init(0, ts, physics_states[ts])
    #     newJ += evaluate(env, np.expand_dims(actions[ts], axis=0))
    # print(J, newJ)
    # opt_act, final_state, J = optimize_clip_segment(env, 0, physics_states[0], actions)
    # print('Final error', J)

    # Single Threaded Version
    # for iteration in range(1):
    #     new_states = []
    #     total_err = 0
    #     for ts in range(steps//SEGMENT_SIZE):
    #         #act = np.expand_dims(actions[ts], axis=0)
    #         act = actions[SEGMENT_SIZE*ts: SEGMENT_SIZE*(ts+1)]
    #         opt_act, final_state, J = optimize_clip_segment(env, SEGMENT_SIZE*ts, physics_states[ts], act)
    #         actions[SEGMENT_SIZE*ts: SEGMENT_SIZE*(ts + 1)] = opt_act
    #         new_states.append(final_state)
    #         total_err += J
    #         print('Step {} optimized to {}'.format(ts, J))
    #     print('Iteration {} ended with total error {}'.format(iteration, total_err))
    #     physics_states[1:] = new_states[:-1]

    # Multithreaded Version
    print('Detected {} cpus.'.format(multiprocessing.cpu_count()))
    optimized_actions = actions
    for iteration in range(OPT_ITERS):
        args_list = []
        for ts in range(STEPS // SEGMENT_SIZE):
            start_step = SEGMENT_SIZE * ts
            start_state = physics_states[ts]
            actions_segment = optimized_actions[SEGMENT_SIZE*ts: SEGMENT_SIZE*(ts+1)]
            args_list.append((start_step, start_state, actions_segment))
            
        num_workers = len(args_list)
        with multiprocessing.Pool(processes=num_workers) as pool:
            outputs = pool.starmap(optimize_clip_segment, args_list)
            optimized_actions = np.concatenate(outputs)
            
        J, physics_states = evaluate_with_physics_states(env, optimized_actions)
        print('Iteration {} Return: {}'.format(iteration, J))

    np.save(os.path.join(OUTPUT_DIR, 'optimized_actions.npy'), actions)


def visualize_trajectory(actions):
    env = build_env()
    print(evaluate(env, actions))

    def policy(time_step):
        global step
        if time_step.first():
            step = 0
        else:
            step += 1
        if step < len(actions):
            return actions[step]
        else:
            return np.zeros_like(actions[0])

    viewer.launch(env, policy=policy)


if __name__ == "__main__":
    # main()
    visualize_trajectory(np.load('pt/exp1/simple_job/optimized_actions.npy'))
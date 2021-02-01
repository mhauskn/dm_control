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

mjlib = mjbindings.mjlib

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
    J = 0
    time_step = env.reset()
    for step in range(actions.shape[0]):
        ref_feats = utils.get_features(env.physics, env.task._walker)
        act = actions[step]
        time_step = env.step(act)
        J += time_step.reward
        if time_step.last():
            break
    return -J

def optimize_clip_segment(env, start_step, init_physics_state, actions):
    env.task.set_custom_init(0, start_step, init_physics_state)    
    # print('Initial Val:', evaluate(env, actions))
    x0 = actions.flatten()
    res = optimize.minimize(
        lambda x: evaluate(env, x.reshape(actions.shape)), 
        x0, 
        method='powell',
        bounds=optimize.Bounds(lb=-np.ones_like(x0), ub=np.ones_like(x0)),
        options={
            'disp': True,
            'maxiter': 1,
        }
    )
    opt_actions = res.x.reshape(actions.shape)
    final_perf = res.fun #evaluate(env, opt_actions)
    final_state = env._physics.get_state().copy()
    return opt_actions, final_state, final_perf

def main():    
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

    spec = env.action_spec()

    steps = 8
    actions, physics_states, J = get_initial_trajectory(env, max_steps=steps)
    print('Initial Trajectory Error', -J)
    # newJ = 0
    # for ts in range(steps):
    #     env.task.set_custom_init(0, ts, physics_states[ts])
    #     newJ += evaluate(env, np.expand_dims(actions[ts], axis=0))
    # print(J, newJ)
    # opt_act, final_state, J = optimize_clip_segment(env, 0, physics_states[0], actions)
    # print('Final error', J)
    seg_size = 4
    for iteration in range(1):
        new_states = []
        total_err = 0
        for ts in range(steps//seg_size):
            #act = np.expand_dims(actions[ts], axis=0)
            act = actions[seg_size*ts: seg_size*(ts+1)]
            opt_act, final_state, J = optimize_clip_segment(env, seg_size*ts, physics_states[ts], act)
            actions[seg_size*ts: seg_size*(ts + 1)] = opt_act
            new_states.append(final_state)
            total_err += J
            print('Step {} optimized to {}'.format(ts, J))
        print('Iteration {} ended with total error {}'.format(iteration, total_err))
        physics_states[1:] = new_states[:-1]


if __name__ == "__main__":
    main()
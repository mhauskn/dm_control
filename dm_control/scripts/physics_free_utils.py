from collections import OrderedDict
import numpy as np

def include_pose(env, time_step):
    obs = time_step.observation
    pos, quat = env.task._walker.get_pose(env.physics)
    if 'walker/position' not in obs:
        obs['walker/position'] = pos.copy()[np.newaxis]
    if 'walker/quaternion' not in obs:
        obs['walker/quaternion'] = quat.copy()[np.newaxis]

def get_observation(walker, physics):
    observation = OrderedDict()
    pos, quat = [arr.copy() for arr in walker.get_pose(physics)]
    joints = physics.bind(walker.mocap_joints).qpos.copy()

    if 'walker/position' not in observation:
        observation['walker/position'] = pos
    if 'walker/quaternion' not in observation:
        observation['walker/quaternion'] = quat

    return observation

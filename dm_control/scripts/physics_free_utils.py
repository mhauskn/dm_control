from collections import OrderedDict
import numpy as np

def include_pose_and_joints(env, time_step):
    obs = time_step.observation
    pos, quat = env.task._walker.get_pose(env.physics)
    obs['walker/position'] = pos.copy()[np.newaxis]
    obs['walker/quaternion'] = quat.copy()[np.newaxis]
    obs['walker/joints'] = env.task._walker.observables.joints_pos(env.physics).copy()[np.newaxis]

def get_observation(walker, physics):
    observation = OrderedDict()
    pos, quat = [arr.copy() for arr in walker.get_pose(physics)]
    joints = physics.bind(walker.mocap_joints).qpos.copy()

    observation['walker/position'] = pos
    observation['walker/quaternion'] = quat
    observation['walker/joints'] = joints

    return observation

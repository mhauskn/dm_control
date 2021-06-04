from collections import OrderedDict
import numpy as np

from dm_env import TimeStep
from dm_control.locomotion.tasks.reference_pose.utils import set_walker


def get_observation(walker, physics):
    observation = OrderedDict()
    pos, quat = [arr.copy() for arr in walker.get_pose(physics)]
    joints = physics.bind(walker.mocap_joints).qpos.copy()

    observation['walker/position'] = pos[np.newaxis]
    observation['walker/quaternion'] = quat[np.newaxis]
    observation['walker/joints'] = joints[np.newaxis]

    return observation

# Acts as substitute for env.step(act).
def physics_free_step(env, pos, quat, joint, change=False):
    """
    pos: Robot position
    quat: Robot quaternion
    joint: Robot's joint angles
    change: If true, pos, quat, and joint are *changes* in those quantities.
    """
    walker = env.task._walker

    # Get current state
    curr_obs = get_observation(walker, env.physics)

    tmp_time_step = env.step(joint) # to advance the time step

    if change:
        pos = curr_obs['walker/position'].squeeze() + pos
        quat = curr_obs['walker/quaternion'].squeeze() + quat
        joint = curr_obs['walker/joints'].squeeze() + joint
    act = np.concatenate([pos, quat, joint])
    set_walker(env.physics, walker, qpos=act, qvel=np.zeros(act.shape[0]-1))

    reward = env.task.get_reward(env.physics)
    observation = get_observation(walker, env.physics)

    return TimeStep(step_type=tmp_time_step.step_type,
                    reward=reward,
                    discount=tmp_time_step.discount,
                    observation=observation)

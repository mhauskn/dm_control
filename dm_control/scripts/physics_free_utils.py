from collections import OrderedDict

def get_observation(walker, physics):
    observation = OrderedDict()
    pos, quat = [arr.copy() for arr in walker.get_pose(physics)]
    joints = physics.bind(walker.mocap_joints).qpos.copy()

    observation['walker/position'] = pos
    observation['walker/quaternion'] = quat
    observation['walker/joints'] = joints

    return observation

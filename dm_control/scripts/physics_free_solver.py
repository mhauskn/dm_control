import os
import os.path as osp
from collections import OrderedDict
import numpy as np
from absl import app, flags, logging
from dm_env import TimeStep
from dm_control.locomotion.walkers import cmu_humanoid
from dm_control.locomotion.arenas import floors
from dm_control.locomotion.tasks.reference_pose import tracking, types
from dm_control.locomotion.mocap import cmu_mocap_data

from solver import log_flags

OUTPUT_DIR = os.environ.get('PT_OUTPUT_DIR', '.')
DATA_DIR = os.environ.get('PT_DATA_DIR', '.')

FLAGS = flags.FLAGS
flags.DEFINE_spaceseplist("clip_names", "CMU_016_22",
                          "Name of reference clips. See cmu_subsets.py")

def physics_free_step(env, pos, quat, joint, change=False):
    tmp_time_step = env.step(joint)
    walker = env.task._walker
    if change:
        old_pos, old_quat = env.task._walker.get_pose(env.physics)
        old_pos = old_pos.copy()
        old_quat = old_quat.copy()
        old_joint = env.task._walker.observables.joints_pos(env.physics).copy()
        pos = pos + old_pos
        quat = quat + old_quat
        joint = joint + old_joint
    walker.set_pose(env.physics, position=pos, quaternion=quat)
    env.physics.bind(walker.mocap_joints).qpos = joint

    reward = env.task.get_reward(env.physics)
    #observation = env.task.get_observation(env.physics)
    observation = OrderedDict()
    pos, quat = env.task._walker.get_pose(env.physics)
    observation['walker/position'] = pos.copy()[np.newaxis]
    observation['walker/quaternion'] = quat.copy()[np.newaxis]
    observation['walker/joints'] = env.task._walker.observables.joints_pos(env.physics).copy()[np.newaxis]

    return TimeStep(step_type=tmp_time_step.step_type,
                    reward=reward,
                    discount=tmp_time_step.discount,
                    observation=observation)


def build_tasks(clip_names):
    walker = cmu_humanoid.CMUHumanoidPositionControlledV2020
    arena = floors.Floor()
    tasks = tracking.MultiClipMocapTracking(
        walker=walker,
        arena=arena,
        ref_path=cmu_mocap_data.get_path_for_cmu_2020(),
        dataset=types.ClipCollection(ids=clip_names),
        ref_steps=(1, 2, 3, 4, 5),
        always_init_at_clip_start=True
    )
    return tasks

def get_actions(clip):
    observables = clip.as_dict()
    trajectory = np.concatenate([observables['walker/position'],
                                 observables['walker/quaternion'],
                                 observables['walker/joints']],
                                axis=1)
    actions = trajectory[1:]
    return actions.astype(np.float32)

def main(_):
    log_flags(FLAGS)
    tasks = build_tasks(clip_names=FLAGS.clip_names)

    for clip, name in zip(tasks._all_clips, FLAGS.clip_names):
        logging.info(name)
        optimized_actions = get_actions(clip)
        fname = "opt_acts_0.npy"
        if not osp.exists(osp.join(OUTPUT_DIR, name)):
            os.makedirs(osp.join(OUTPUT_DIR, name))
        np.save(osp.join(OUTPUT_DIR, name, fname), optimized_actions)

        with open(osp.join(OUTPUT_DIR, name, 'stdout.txt'), 'w+') as f:
            f.write(f"FLAGS.clip_name: {name}\n")

if __name__ == '__main__':
    app.run(main)

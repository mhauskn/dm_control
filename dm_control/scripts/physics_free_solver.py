import os
import os.path as osp
import numpy as np
from absl import app, flags, logging
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

def build_tasks(clip_names):
    walker = cmu_humanoid.CMUHumanoidPositionControlledV2020
    arena = floors.Floor()
    tasks = tracking.MultiClipMocapTracking(
        walker=walker,
        arena=arena,
        ref_path=cmu_mocap_data.get_path_for_cmu(version='2020'),
        dataset=types.ClipCollection(ids=clip_names),
        ref_steps=(1, 2, 3, 4, 5),
        always_init_at_clip_start=True,
        physics_free=True
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

    # Iterate through given clips, making one directory per clip.
    for clip, name in zip(tasks._all_clips, tasks._dataset.ids):
        logging.info(name)
        optimized_actions = get_actions(clip)
        fname = "opt_acts_0.npy"
        if not osp.exists(osp.join(OUTPUT_DIR, name)):
            os.makedirs(osp.join(OUTPUT_DIR, name))
        np.save(osp.join(OUTPUT_DIR, name, fname), optimized_actions)
        np.save(osp.join(DATA_DIR, name + '_start_step_0.npy'), optimized_actions)

        # Needed so that `create_dataset.py` runs
        with open(osp.join(OUTPUT_DIR, name, 'stdout.txt'), 'w+') as f:
            f.write(f"FLAGS.clip_name: {name}\n")

if __name__ == '__main__':
    app.run(main)

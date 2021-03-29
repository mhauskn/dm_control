import sys
import numpy as np
from dm_control import viewer
from solver import build_env
import dm_control.locomotion.tasks.reference_pose.rewards as rewards
from absl import app
from absl import flags
from texttable import Texttable

FLAGS = flags.FLAGS

def analyze_trajectory(env, actions):
    table = Texttable(max_width=160)
    table.set_deco(Texttable.HEADER)
    table.header(['Step', 'r', 'termErr', 'angular_velocity', 'joints_velocity'])
    J = 0
    env.reset()
    avgs = np.zeros(5)
    for idx, act in enumerate(actions):
        time_step = env.step(act)
        row = [
            idx, 
            time_step.reward,
            env._task._termination_error, 
            1.7e-2 * np.linalg.norm(env._task._walker_features['angular_velocity']),
            3.1e-3 * np.linalg.norm(env._task._walker_features['joints_velocity']),
        ]
        table.add_row(row)
        avgs += np.array(row)
        J += time_step.reward
    print(table.draw())
    print('Total Return {:.3f}'.format(J))


def visualize_trajectory(argv):
    env = build_env(
        reward_type=FLAGS.reward_type, 
        ghost_offset=1, 
        clip_name=FLAGS.clip_name,
        start_step=FLAGS.start_step,
    )
    actions = np.load(FLAGS.load_actions_path)
    analyze_trajectory(env, actions)

    def policy(time_step):
        global step
        if time_step.first():
            step = 0
        else:
            step += 1
        if step < len(actions):
            return actions[step]
        else:
            print('{} Out of actions - returning zeros'.format(step))
            return np.zeros_like(actions[0])

    viewer.launch(env, policy=policy)


if __name__ == "__main__":
    app.run(visualize_trajectory)

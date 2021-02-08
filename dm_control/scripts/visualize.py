import sys
import numpy as np
from dm_control import viewer
from solver import build_env, evaluate, CustomInit
import dm_control.locomotion.tasks.reference_pose.rewards as rewards
from absl import app
from absl import flags
from texttable import Texttable

FLAGS = flags.FLAGS
flags.DEFINE_string("traj_path", None, "Path to a numpy trajectory of actions.")


def main(argv):
    del argv
    actions = np.load(FLAGS.traj_path)
    visualize_trajectory(actions)


def visualize_trajectory(actions):
    env = build_env(ghost_offset=1)
    env.reset()

    # table = prettytable.PrettyTable(['Step', 'r', 'jointErr', 'bodiesErr', 'COM', 'JointVel', 'Appendages', 'Quaternions'])
    # table.set_style({'border':False})
    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.header(['Step', 'r', 'jointErr', 'bodiesErr', 'COM', 'JointVel', 'Appndgs', 'Quats'])
    J = 0
    env.reset()
    for idx, act in enumerate(actions):
        time_step = env.step(act)
        differences = rewards.compute_squared_differences(
            env._task._walker_features, env._task._current_reference_features)
        com = .1 * np.exp(-10 * differences['center_of_mass'])
        joints_velocity = 1.0 * np.exp(-0.1 * differences['joints_velocity'])
        appendages = 0.15 * np.exp(-40. * differences['appendages'])
        body_quaternions = 0.65 * np.exp(-2 * differences['body_quaternions'])
        table.add_row([
            idx, 
            time_step.reward,
            env._task._joint_error, 
            env._task._bodies_error,
            differences['center_of_mass'],
            differences['joints_velocity'],
            differences['appendages'],
            differences['body_quaternions']])
        J += time_step.reward
    print(table.draw())
    print('Total Return {:.2f}'.format(J))

    def policy(time_step):
        global step
        if time_step.first():
            step = 0
        else:
            step += 1
        if step < len(actions):
            return actions[step]
        else:
            print('Out of actions - returning zeros')
            return np.zeros_like(actions[0])

    viewer.launch(env, policy=policy)


if __name__ == "__main__":
    app.run(main)

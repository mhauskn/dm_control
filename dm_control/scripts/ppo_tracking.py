import numpy as np

from dm_control.locomotion.mocap import cmu_mocap_data
from dm_control.locomotion.walkers import cmu_humanoid
from dm_control.locomotion.tasks.reference_pose import tracking
from dm_control.locomotion.arenas import floors
from dm_control.locomotion.tasks.reference_pose import types
from dm_control import composer

from vec_env import SubprocVecEnv
from model import ActorCritic
from rl.ppo.ppo import PPOTrainerConfig, PPOTrainer

from absl import flags, logging, app

FLAGS = flags.FLAGS

def build_env(reward_type='termination', ghost_offset=0, clip_name='CMU_016_22', start_step=0,
              force_magnitude=0, disable_observables=False, termination_error_threshold=1e10, seed=42,
              observables=['walker/joints_pos', 'walker/joints_vel']):
    walker = cmu_humanoid.CMUHumanoidPositionControlledV2020
    arena = floors.Floor()
    task = tracking.MultiClipMocapTracking(
        walker=walker,
        arena=arena,
        ref_path=cmu_mocap_data.get_path_for_cmu(version='2020'),
        dataset=types.ClipCollection(ids=[clip_name]),
        ref_steps=(1, 2, 3, 4, 5),
        start_step=start_step,
        max_steps=256,
        reward_type=reward_type,
        always_init_at_clip_start=True,
        termination_error_threshold=termination_error_threshold,
        ghost_offset=ghost_offset,
        force_magnitude=force_magnitude,
        disable_observables=disable_observables,
    )
    env = composer.Environment(
        task=task,
        random_state=np.random.RandomState(seed=seed)
    )
    wrapped_env = EnvWrap(env, observables)
    return wrapped_env


class EnvWrap:
    """ Commits to specific observables which define the task. """
    def __init__(self, env, observables=['walker/joints_pos', 'walker/joints_vel']):
        self.env = env
        self.observables = observables
        self.observation_space = sum(self.env.observation_spec()[o].shape[1] for o in observables)
        self.action_space = self.env.action_spec().shape[0]
    
    def reset(self):
        time_step = self.env.reset()
        obs = np.concatenate([time_step.observation[o].squeeze() for o in self.observables])
        return obs

    def step(self, action):
        time_step = self.env.step(action)
        obs = np.concatenate([time_step.observation[o].squeeze() for o in self.observables])
        return obs, time_step.reward, time_step.last(), {}

    def close(self):
        self.env.close()


def log_flags(flags):
    """ Logs the value of each of the flags. """
    for k in dir(flags):
        if k != '?':
            flag = 'FLAGS.{}'.format(k)
            logging.info('{}: {}'.format(flag, eval(flag)))

def main(argv):
    log_flags(FLAGS)
    vec_env = SubprocVecEnv([build_env for _ in range(8)])
    tconf = PPOTrainerConfig(FLAGS)
    policy = ActorCritic(obs_size=vec_env.observation_space, action_size=vec_env.action_space)
    eval_env = build_env()
    trainer = PPOTrainer(policy, vec_env, eval_env, tconf)
    trainer.train()

if __name__ == "__main__":
    app.run(main)
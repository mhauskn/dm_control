from pathlib import Path
import os.path as osp
import numpy as np
import torch
from absl import app, flags, logging

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from callbacks import SaveVecNormalizeCallback
from wrappers import LocomotionWrapper

FLAGS = flags.FLAGS
flags.DEFINE_string("clip_name", "CMU_016_22", "Name of reference clip. See cmu_subsets.py")
flags.DEFINE_float("learning_rate", 1e-4, "Step size for A2C")
flags.DEFINE_integer("n_workers", 16, "Number of workers")
flags.DEFINE_integer("n_steps", 64, "Number of steps per worker per update")
flags.DEFINE_float("gamma", 0.95, "Discount factor")
flags.DEFINE_integer("seed", 0, "RNG seed")

class LocomotionExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        self._keys = (
            'walker/joints_pos',
            'walker/joints_vel',
            'walker/sensors_velocimeter',
            'walker/sensors_gyro',
            'walker/end_effectors_pos',
            'walker/world_zaxis',
            'walker/body_height',
            'walker/actuator_activation',
            'walker/sensors_touch',
            'walker/sensors_torque',
            'walker/reference_rel_bodies_pos_local',
            'walker/reference_rel_bodies_quats'
        )
        features_dim = sum([observation_space.spaces[k].shape[0] for k in self._keys])
        super().__init__(observation_space, features_dim=features_dim)

    def forward(self, observations):
        return torch.cat([observations[k] for k in self._keys], dim=-1)

def make_env(rank, seed=0, start_step=None):
    def _init():
        task_kwargs = dict(
            reward_type='comic',
            start_step=start_step,
        )
        environment_kwargs = dict(
            random_state=np.random.RandomState(seed=seed+rank)
        )
        env = LocomotionWrapper(task_kwargs=task_kwargs,
                                environment_kwargs=environment_kwargs)
        return env
    return _init

def log_flags(flags):
    """ Logs the value of each of the flags. """
    for k in dir(flags):
        if k != '?':
            flag = 'FLAGS.{}'.format(k)
            logging.info('{}: {}'.format(flag, eval(flag)))

def main(_):
    log_flags(FLAGS)
    log_dir = osp.join('logs', FLAGS.clip_name, str(FLAGS.seed))
    Path(osp.join(log_dir, 'train')).mkdir(parents=True, exist_ok=True)
    Path(osp.join(log_dir, 'eval/model')).mkdir(parents=True, exist_ok=True)

    logger = configure(log_dir, ['stdout', 'csv', 'log'])

    env = VecMonitor(SubprocVecEnv([make_env(i) for i in range(FLAGS.n_workers)]),
                     osp.join(log_dir, 'train'))
    env = VecNormalize(env, gamma=FLAGS.gamma)
    eval_env = VecMonitor(SubprocVecEnv([make_env(i, start_step=0) for i in
                                         range(FLAGS.n_workers)]),
                          osp.join(log_dir, 'eval'))
    eval_env = VecNormalize(eval_env, training=False, gamma=FLAGS.gamma)

    model_path = osp.join(log_dir, 'eval/model')
    save_vec_normalize = SaveVecNormalizeCallback(save_freq=1, save_path=model_path)
    eval_callback = EvalCallback(eval_env, best_model_save_path=model_path,
                                 log_path=osp.join(log_dir, 'eval'), eval_freq=int(1e3),
                                 callback_on_new_best=save_vec_normalize,
                                 n_eval_episodes=1, deterministic=True, render=False)

    policy_kwargs = dict(
        features_extractor_class=LocomotionExtractor,
        net_arch=[dict(pi=[512,512,512], vf=[512,512,512])],
        activation_fn=torch.nn.ReLU,
        log_std_init=np.log(0.3),
    )
    model = A2C("MultiInputPolicy", env, n_steps=64, gamma=FLAGS.gamma,
                learning_rate=FLAGS.learning_rate, policy_kwargs=policy_kwargs,
                verbose=1, device='cpu')
    model.set_logger(logger)
    model.learn(int(1e8), callback=eval_callback)

    mean_reward, _ = evaluate_policy(model, model.get_env(), n_eval_episodes=20)
    print(mean_reward)
    return mean_reward

if __name__ == '__main__':
    app.run(main)

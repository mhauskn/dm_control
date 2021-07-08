from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv, VecNormalize

from wrappers import LocomotionWrapper

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

if __name__ == '__main__':
    Path('./logs/train').mkdir(parents=True, exist_ok=True)
    Path('./logs/eval/model').mkdir(parents=True, exist_ok=True)

    log_path = './logs/'
    logger = configure(log_path, ['stdout', 'csv', 'tensorboard', 'log'])

    n_cpu = 16
    env = VecMonitor(SubprocVecEnv([make_env(i) for i in range(n_cpu)]), './logs/train/')
    env = VecNormalize(env, gamma=0.95)
    eval_env = VecMonitor(DummyVecEnv([make_env(0, start_step=0)]), './logs/eval/')
    eval_env = VecNormalize(eval_env, training=False, gamma=0.95)

    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/eval/model/',
                                 log_path='./logs/eval/', eval_freq=int(1e3),
                                 n_eval_episodes=1, deterministic=True, render=False)

    policy_kwargs = dict(
        features_extractor_class=LocomotionExtractor,
        net_arch=[dict(pi=[512,512,512], vf=[512,512,512])],
        log_std_init=np.log(0.3),
    )
    model = PPO("MultiInputPolicy", env, n_steps=1024, gamma=0.95,
                learning_rate=1e-4, target_kl=0.1, policy_kwargs=policy_kwargs,
                verbose=1, device='cpu')
    model.set_logger(logger)
    model.learn(int(1e8), callback=eval_callback)
import gym
import numpy as np
import torch
from torch.optim import Adam
from torch.nn import functional as F
import multiprocessing

from vec_env import SubprocVecEnv
from model import ActorCritic
from rl.ppo.ppo import PPOTrainerConfig, PPOTrainer

from absl import flags, logging, app

FLAGS = flags.FLAGS

class EnvWrap:
    """ Commits to specific observables which define the task. """
    def __init__(self):
        self.env = gym.make('BipedalWalker-v3')
        self.observation_space = 24
        self.action_space = 4
    
    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

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
    cpus = multiprocessing.cpu_count()
    print(f'Detected {cpus} CPUs')
    vec_env = SubprocVecEnv([EnvWrap for _ in range(cpus)])
    tconf = PPOTrainerConfig(FLAGS)
    policy = ActorCritic(obs_size=vec_env.observation_space, action_size=vec_env.action_space)
    eval_env = EnvWrap()
    trainer = PPOTrainer(policy, vec_env, eval_env, tconf)
    trainer.train()

if __name__ == "__main__":
    app.run(main)


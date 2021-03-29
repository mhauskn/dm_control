import torch
import numpy as np
from trainer import Trainer, TrainerConfig
from model import FFNet
from dataset import TrajectoryDataset
from absl import app
from solver import build_env
from dataset import OBS_KEYS
from dm_control import viewer

CHECKPOINT_PATH = 'saved_model.pt'

def build_observation(time_step):
    obs = time_step.observation
    full_obs = np.concatenate([obs[k][...] for k in OBS_KEYS], axis=1)
    return torch.Tensor(full_obs)


def run_episode(env, model):
    time_step = env.reset()
    J = 0
    episode_steps = 0
    while not time_step.last():
        obs = build_observation(time_step)
        act, _ = model(obs)
        act = act.cpu().numpy()
        time_step = env.step(act)
        J += time_step.reward
        episode_steps += 1
    return J, episode_steps


def evaluate(argv):
    model = FFNet()
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval()
    env = build_env(
        reward_type='termination',
        clip_name='CMU_016_22',
        start_step=0,
        force_magnitude=0,
        disable_observables=False,
    )
    with torch.set_grad_enabled(False):
        J, episode_steps = run_episode(env, model)
        print('Episode Return: {:.2f} Steps: {}'.format(J, episode_steps))


def visualize(argv):
    """ Visualizes the model running on a trajectory. """
    model = FFNet()
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval()
    env = build_env(
        reward_type='termination',
        clip_name='CMU_016_22',
        start_step=0,
        force_magnitude=0,
        disable_observables=False,
    )

    def policy(time_step):
        obs = build_observation(time_step)
        with torch.no_grad():
            act, _ = model(obs)
        return act.cpu().numpy()

    viewer.launch(env, policy)

def train(argv):
    train_dataset = TrajectoryDataset('small_trajectory_dataset.hdf5', block_size=1)
    model = FFNet()
    tconf = TrainerConfig(max_epochs=1, ckpt_path=CHECKPOINT_PATH)
    trainer = Trainer(model, train_dataset, None, tconf)
    trainer.train()

if __name__ == "__main__":
    # app.run(train)
    # app.run(evaluate)
    app.run(visualize)
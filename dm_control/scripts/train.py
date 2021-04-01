import torch
import numpy as np
from trainer import Trainer, TrainerConfig
from model import FFNet
from dataset import TrajectoryDataset
from absl import app
from solver import build_env
from dataset import OBS_KEYS
from dm_control import viewer
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_path", 'saved_model.pt', 'Path to save model checkpoints')
flags.DEFINE_string("dataset_path", 'data/single_episode.hdf5', 'Path to load the dataset.')
# flags.DEFINE_string("clip_name", 'CMU_016_32', 'Name of clip to run.')
flags.DEFINE_string("ref_actions_path", 'pt/overkill4/default/opt_acts_0.npy', 'Path to load reference actions.')

def build_observation(time_step):
    obs = time_step.observation
    full_obs = np.concatenate([obs[k][...] for k in OBS_KEYS], axis=1)
    return torch.Tensor(full_obs)

def run_episode_with_reference_actions(env, model, reference_actions):
    """ Runs the episode using the reference_actions but also compares the model's predictions. """
    time_step = env.reset()
    J = 0
    episode_steps = 0
    norms = []
    while not time_step.last():
        # obs = build_observation(time_step)
        obs = np.zeros((1,152), dtype=np.float32)
        obs[0,episode_steps] = 1.
        obs = torch.Tensor(obs)
        act, _ = model(obs)
        act = act.cpu().numpy()
        ref_act = reference_actions[episode_steps]
        norms.append(np.linalg.norm(ref_act - act))
        time_step = env.step(ref_act)
        J += time_step.reward
        episode_steps += 1
    print("L2 Norm between reference_acts and model's actions: {:.5f}".format(np.mean(norms)))
    return J, episode_steps

def run_episode(env, model):
    time_step = env.reset()
    J = 0
    episode_steps = 0
    while not time_step.last():
        # obs = build_observation(time_step)
        obs = np.zeros((1,152), dtype=np.float32)
        obs[0,episode_steps] = 1.
        obs = torch.Tensor(obs)
        act, _ = model(obs)
        act = act.cpu().numpy()
        time_step = env.step(act)
        J += time_step.reward
        episode_steps += 1
    return J, episode_steps

def load_model(model_path):
    model = FFNet()
    model.load_state_dict(torch.load(model_path))
    return model

def get_env():
    env = build_env(
        reward_type='termination',
        clip_name=FLAGS.clip_name,
        start_step=0,
        force_magnitude=0,
        disable_observables=False,
        termination_error_threshold=0.3,
    )
    return env

def evaluate(model=None, reference_actions=None):
    if not model:
        model = FFNet()
        model.load_state_dict(torch.load(FLAGS.checkpoint_path))
    model.eval()
    env = get_env()
    with torch.no_grad():
        if reference_actions:
            run_episode_with_reference_actions(env, model, reference_actions=np.load(reference_actions))
        else:
            J, episode_steps = run_episode(env, model)
            print('Episode Cost: {:.2f} Steps: {}'.format(J, episode_steps))

def visualize(model=None):
    """ Visualizes the model running on a trajectory. """
    if not model:
        model = FFNet()
        model.load_state_dict(torch.load(FLAGS.checkpoint_path))
    model.eval()
    env = get_env()

    def policy(time_step):
        global episode_steps
        if time_step.first():
            episode_steps = 0
        obs = np.zeros((1,152), dtype=np.float32)
        obs[0,episode_steps] = 1.
        obs = torch.Tensor(obs)
        episode_steps += 1
        # obs = build_observation(time_step)
        with torch.no_grad():
            act, _ = model(obs)
        return act.cpu().numpy()
    viewer.launch(env, policy)

def train():
    train_dataset = TrajectoryDataset(FLAGS.dataset_path, block_size=1)
    model = FFNet()
    tconf = TrainerConfig(
        batch_size=152,
        max_epochs=2000, 
        ckpt_path=FLAGS.checkpoint_path, 
        learning_rate=.0001,
        betas=(0.9, 0.999),
        grad_norm_clip=5.0,
        lr_decay=False,
    )
    trainer = Trainer(model, train_dataset, None, tconf)
    trainer.train()
    return model

def main(argv):
    # model = train()
    evaluate()
    # evaluate(reference_actions=FLAGS.ref_actions_path)
    # visualize()

if __name__ == "__main__":
    app.run(main)
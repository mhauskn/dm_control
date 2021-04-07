import torch
import os
import numpy as np
from solver import build_env
from model import FFNet
from dm_control import viewer
from dataset import OBS_KEYS
from absl import flags, logging, app

FLAGS = flags.FLAGS
flags.DEFINE_string("ref_actions_path", 'pt/overkill4/default/opt_acts_0.npy', 'Path to load reference actions.')
flags.DEFINE_string("load_model_path", 'saved_model.pt', "Path to load the saved model from.")


def build_observation(time_step):
    obs = time_step.observation
    feats = []
    for k in OBS_KEYS:
        feature = np.array(obs[k], dtype=np.float32, copy=True)
        if feature.ndim < 2:
            feature = feature[:, np.newaxis]
        feats.append(feature)
    full_obs = np.concatenate(feats, axis=1)        
    return torch.Tensor(full_obs)

def build_onehot_observation(episode_step):
    obs = np.zeros((1,152), dtype=np.float32)
    obs[0,episode_step] = 1.
    obs = torch.Tensor(obs)
    return obs

def visualize(model):
    """ Visualizes the model running on a trajectory. """
    model.eval()
    env = get_env()

    def policy(time_step):
        global episode_steps
        if time_step.first():
            episode_steps = 0
        episode_steps += 1
        obs = build_observation(time_step)
        with torch.no_grad():
            act, _ = model(obs)
        return act.cpu().numpy()
    viewer.launch(env, policy)

def run_episode_with_reference_actions(env, model, reference_actions):
    """ Runs the episode using the reference_actions but also compares the model's predictions. """
    time_step = env.reset()
    J = 0
    episode_steps = 0
    norms = []
    while not time_step.last():
        obs = build_observation(time_step)
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
        obs = build_observation(time_step)
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

def evaluate(model, reference_actions=None):
    model.eval()
    env = get_env()
    with torch.no_grad():
        if reference_actions:
            run_episode_with_reference_actions(env, model, reference_actions=np.load(reference_actions))
        else:
            J, episode_steps = run_episode(env, model)
            logging.info('Episode Cost: {:.2f} Steps: {}'.format(J, episode_steps))

def main(argv):
    model = load_model(FLAGS.load_model_path)
    evaluate(model)
    # visualize(model)

if __name__ == "__main__":
    app.run(main)
import torch
import os
import numpy as np
from solver import build_env
from model import FFNet, GPT, GPTConfig
from dm_control import viewer
from dataset import OBS_KEYS
from absl import flags, logging, app

FLAGS = flags.FLAGS
flags.DEFINE_string("ref_actions_path", 'pt/overkill4/default/opt_acts_0.npy', 'Path to load reference actions.')
flags.DEFINE_string("exp_dir", '.', "Path to directory containing saved files.")
flags.DEFINE_string("model_fname", 'saved_model.pt', "Filename of model to load (in exp_dir)")
flags.DEFINE_string("config_fname", 'saved_model_config.json', "Filename of config to load (in exp_dir)")
flags.DEFINE_list("observables", "joints_pos, joints_vel", "List of observation features to use.")

def get_observables():
    observables = ['walker/' + o for o in FLAGS.observables]
    sorted(observables)
    return observables

def build_observation(time_step):
    obs = time_step.observation
    feats = []
    for k in get_observables():
        feature = np.array(obs[k], dtype=np.float32, copy=True)
        if feature.ndim < 2:
            feature = feature[:, np.newaxis]
        feats.append(feature)
    full_obs = np.concatenate(feats, axis=1)        
    return full_obs

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

def run_episode(env, model, block_size):
    time_step = env.reset()
    J = 0
    episode_steps = 0
    obs_queue = []
    # Build the context
    for _ in range(block_size):
        obs = build_observation(time_step)
        obs_queue.append(obs)
        time_step = env.step(np.zeros(56))
        # Reset the walker to the current reference frame
        env._task._set_walker(env.physics)
    assert len(obs_queue) == block_size

    while not time_step.last():
        obs = build_observation(time_step)
        obs_queue.append(obs)
        obs_queue.pop(0)
        obs_tt = torch.Tensor(np.stack(obs_queue, axis=1))
        act, _ = model(obs_tt)
        act = act.squeeze()[-1].cpu().numpy() # (1,4,56) ==> (56,)
        time_step = env.step(act)
        J += time_step.reward
        episode_steps += 1
    return J, episode_steps

def load_model():
    # model = FFNet()
    config_path = os.path.join(FLAGS.exp_dir, FLAGS.config_fname)
    model_path = os.path.join(FLAGS.exp_dir, FLAGS.model_fname)
    mconf = GPTConfig.from_json(config_path)
    model = GPT(mconf)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
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
            J, episode_steps = run_episode(env, model, block_size=4)
            logging.info('Episode Cost: {:.2f} Steps: {}'.format(J, episode_steps))
    env.close()

def main(argv):
    model = load_model()
    evaluate(model)
    # visualize(model)

if __name__ == "__main__":
    app.run(main)
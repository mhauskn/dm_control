import torch
import os
import numpy as np
from solver import build_env
from model import FFNet, GPT, GPTConfig
from dm_control import viewer
from dataset import OBS_KEYS
from collections import deque
from absl import flags, logging, app
import glob

FLAGS = flags.FLAGS
flags.DEFINE_string("ref_actions_path", 'pt/overkill4/default/opt_acts_0.npy', 'Path to load reference actions.')
flags.DEFINE_string("exp_dir", '.', "Path to directory containing saved files.")
flags.DEFINE_string("model_fname", 'saved_model.pt', "Filename of model to load (in exp_dir)")
flags.DEFINE_string("config_fname", 'saved_model_config.json', "Filename of config to load (in exp_dir)")
flags.DEFINE_boolean("visualize", False, "Visualize the policy")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def observables_sorted(observables):
    observables = ['walker/' + o for o in observables if not o.startswith('walker/')]
    sorted(observables)
    return observables

def build_observation(time_step, observables):
    obs = time_step.observation
    feats = []
    for k in observables_sorted(observables):
        feature = np.array(obs[k], dtype=np.float32, copy=True)
        if feature.ndim < 2:
            feature = feature[:, np.newaxis]
        feats.append(feature)
    full_obs = np.concatenate(feats, axis=1)        
    return full_obs

@torch.no_grad()
def visualize(env, model, reference_actions):
    """ Visualizes the model running on a trajectory. """
    model.eval()

    def policy(time_step):
        global episode_steps
        global obs_queue

        if time_step.first():
            episode_steps = 0
            obs_queue = deque()

        obs = build_observation(time_step, model.observables)
        obs_queue.append(obs)

        if len(obs_queue) >= model.block_size:
            obs_tt = torch.FloatTensor(np.stack(obs_queue, axis=1)).to(device)
            act, _ = model(obs_tt)
            act = act.squeeze()[-1].cpu().numpy()
            obs_queue.popleft()
        else:
            act = reference_actions[episode_steps]

        episode_steps += 1
        return act

    viewer.launch(env, policy)


def validate_reference_actions(env, reference_actions):
    """ Ensure the reference actions take us through the episode without failure. """
    assert env.task._termination_error_threshold <= 0.3
    time_step = env.reset()
    episode_steps = 0
    norms = []
    for idx, act in enumerate(reference_actions):
        time_step = env.step(act)
        if env.task._should_truncate:
            logging.fatal(f'Ref_action validation failed at step {idx}')
    logging.debug(f'Successfully validated ref_actions on {get_clip_name(env)} for {len(reference_actions)} steps')


@torch.no_grad()
def run_episode(env, model, reference_actions, start_step=0):
    """ Runs an episode, using the reference actions only to build the necessary context. """
    if len(reference_actions) <= max(model.block_size, start_step):
        # Clip is too short to build adequate context for the model
        return 0, 0
    time_step = env.reset()
    J = 0
    episode_steps = 0
    obs_queue = deque()

    # Build the context using reference actions
    for idx in range(max(model.block_size, start_step)):
        obs = build_observation(time_step, model.observables)
        obs_queue.append(obs)
        time_step = env.step(reference_actions[idx])
    while len(obs_queue) > model.block_size:
        obs_queue.popleft()
    assert len(obs_queue) == model.block_size

    while not time_step.last():
        obs = build_observation(time_step, model.observables)
        obs_queue.append(obs)
        obs_queue.popleft()
        obs_tt = torch.FloatTensor(np.stack(obs_queue, axis=1)).to(device)
        act, _ = model(obs_tt)
        act = act.squeeze()[-1].cpu().numpy() # (1,4,56) ==> (56,)
        time_step = env.step(act)
        J += time_step.reward
        episode_steps += 1
    return J, episode_steps


@torch.no_grad()
def run_episode_with_reference_actions(env, model, reference_actions):
    """ Runs the episode using the reference_actions and computes the l2 norm between the
        model and reference actions at each step. Returns the average norm.
    """
    time_step = env.reset()
    J = 0
    episode_steps = 0
    norms = []
    obs_queue = deque()

    while not time_step.last():
        ref_act = reference_actions[episode_steps]
        obs = build_observation(time_step, model.observables)
        obs_queue.append(obs)
        if len(obs_queue) >= model.block_size:
            obs_tt = torch.FloatTensor(np.stack(obs_queue, axis=1)).to(device)
            act, _ = model(obs_tt)
            act = act.cpu().numpy()
            norms.append(np.linalg.norm(ref_act - act))
            obs_queue.popleft()
        time_step = env.step(ref_act)
        J += time_step.reward
        episode_steps += 1
    return np.mean(norms)

def get_clip_name(env):
    return env.task._dataset.ids[0]

def load_model(config_path, model_path):
    mconf = GPTConfig.from_json(config_path)
    model = GPT(mconf)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def evaluate(env, model, reference_actions, preferred_start_step=32):
    model.eval()
    # Evaluate action completion
    steps2term = []
    J, episode_steps = run_episode(
        env,
        model, 
        reference_actions,
        start_step=preferred_start_step,
    )
    # Evaluate similarity to reference actions
    norm_diff = run_episode_with_reference_actions(env, model, reference_actions)
    logging.debug(f'{get_clip_name(env)} ref_act_norm_diff={norm_diff:.5f}')
    return episode_steps, norm_diff


def comprehensive_eval(eval_dir, model, visualize_policy=False):
    """ Loads all clips in the evaluation dir and runs evaluation on each. """

    def parse_path(ref_actions_path):
        basename = os.path.basename(ref_action_path)
        basename = os.path.splitext(basename)[0] # Remove the .npy extension
        res = basename.split('_start_step_')
        clip_name = res[0]
        start_step = int(res[1])
        return clip_name, start_step

    for ref_action_path in glob.glob(os.path.join(eval_dir, '*.npy')):
        ref_actions = np.load(ref_action_path)
        clip_name, start_step = parse_path(ref_action_path)
        env = build_env(
            reward_type='termination',
            ghost_offset=0,
            clip_name=clip_name,
            start_step=start_step,
            force_magnitude=0,
            disable_observables=False,
            termination_error_threshold=0.3,
        )
        validate_reference_actions(env, ref_actions)
        steps2term, norm_diff = evaluate(env, model, ref_actions)
        logging.info(f'Eval {clip_name}: steps2term={steps2term} norm_diff={norm_diff:.1f}')
        if visualize_policy:
            visualize(env, model, ref_actions)
        env.close()


def main(argv):
    config_path = os.path.join(FLAGS.exp_dir, FLAGS.config_fname)
    model_path = os.path.join(FLAGS.exp_dir, FLAGS.model_fname)
    model = load_model(config_path, model_path)
    comprehensive_eval('data/eval', model, visualize_policy=FLAGS.visualize)
    # env = build_env(
    #     reward_type='termination',
    #     ghost_offset=0,
    #     clip_name=FLAGS.clip_name,
    #     start_step=0,
    #     force_magnitude=0,
    #     disable_observables=False,
    #     termination_error_threshold=0.3,
    # )
    # config_path = os.path.join(FLAGS.exp_dir, FLAGS.config_fname)
    # model_path = os.path.join(FLAGS.exp_dir, FLAGS.model_fname)
    # model = load_model(config_path, model_path)
    # ref_actions = np.load(FLAGS.ref_actions_path)
    # validate_reference_actions(env, ref_actions)
    # evaluate(env, model, ref_actions)
    # visualize(env, model, ref_actions)

if __name__ == "__main__":
    app.run(main)
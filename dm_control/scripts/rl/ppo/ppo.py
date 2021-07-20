import os
import time
import numpy as np
import torch
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from absl import flags

flags.DEFINE_integer("max_epochs", 100, "Maximum training epochs.")
flags.DEFINE_integer("steps_per_epoch", 2048, "Number of steps of interaction in each epoch.")
flags.DEFINE_integer("gradient_steps_per_update", 64, "Gradient steps to take each update.")
flags.DEFINE_integer("eval_episodes", 5, "Number of evaluation episodes to run.")
flags.DEFINE_float("lr", .0003, "Learning rate")
flags.DEFINE_float("clip_ratio", 0.2, "Clipping parameter for policy deviation.")
flags.DEFINE_float("ent_coef", 0, "Coefficent of entropy loss.")
flags.DEFINE_float("vf_coef", 0.5, "Coefficient of value loss.")
flags.DEFINE_float("grad_norm_clip", 0.5, "Gradient norm clipping value.")
flags.DEFINE_float("gamma", 0.99, "Discount factor.")
flags.DEFINE_float("lam", 0.97, "Lambda for GAE-Lambda")
flags.DEFINE_float("target_kl", 0.01, "Limits the KL divergence between updates.")


class PPOBuffer:

    def __init__(self, obs_dim, act_dim, n_envs, buffer_size, gamma, lam):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_envs = n_envs
        self.gamma, self.gae_lambda = gamma, lam
        self.ptr = 0

    def reset(self):
        self.observations = np.zeros((self.buffer_size, self.n_envs, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.act_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.ptr = 0

    def store(self, obs, act, rew, done, val, logp):
        """ Append one timestep of agent-environment interaction to the buffer. """
        assert self.ptr < self.buffer_size
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = act
        self.dones[self.ptr] = done
        self.rewards[self.ptr] = rew
        self.values[self.ptr] = val
        self.log_probs[self.ptr] = logp
        self.ptr += 1

    def get(self):
        """ Call this at the end of an epoch to get all of the data from the buffer. """
        assert self.ptr == self.buffer_size
        self.ptr = 0
        data = dict(obs=self.observations, act=self.actions, ret=self.returns,
                    adv=self.advantages, val=self.values, logp=self.log_probs)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}    

    def compute_returns_and_advantage(self, last_values: np.ndarray):
        """ Post-processing step: compute the lambda-return (TD(lambda) estimate) and GAE(lambda) advantage. """
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_non_terminal = 1.0 - self.dones[step]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values


class PPOTrainerConfig:
    def __init__(self, FLAGS):
        self.max_epochs = FLAGS.max_epochs
        self.steps_per_epoch = FLAGS.steps_per_epoch
        self.gradient_steps_per_update = FLAGS.gradient_steps_per_update
        self.lr = FLAGS.lr
        self.clip_ratio = FLAGS.clip_ratio
        self.ent_coef = FLAGS.ent_coef
        self.vf_coef = FLAGS.vf_coef
        self.grad_norm_clip = FLAGS.grad_norm_clip
        self.gamma = FLAGS.gamma
        self.lam = FLAGS.lam
        self.target_kl = FLAGS.target_kl
        self.eval_episodes = FLAGS.eval_episodes


class PPOTrainer:
    def __init__(self, policy, vec_env, eval_env, config):
        self.policy = policy
        self.config = config
        self.env = vec_env
        self.eval_env = eval_env

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.policy = torch.nn.DataParallel(self.policy).to(self.device)

        self.buffer = PPOBuffer(
            obs_dim=vec_env.observation_space, 
            act_dim=vec_env.action_space, 
            n_envs=vec_env.num_envs,
            buffer_size=config.steps_per_epoch,
            gamma=config.gamma,
            lam=config.lam,
        )

        self.policy_optimizer = Adam(policy.parameters(), lr=config.lr)

        self.writer = SummaryWriter(os.environ.get('AMLT_OUTPUT_DIR', '.'), flush_secs=30)
        self.gradient_steps = 0


    def train(self):
        """ Main PPO training loop. """
        start_time = time.time()
        obs = self.env.reset()
        total_steps, total_episodes = 0, 0

        for epoch in range(self.config.max_epochs):
            self.buffer.reset()
            J, ep_steps = np.zeros((self.env.num_envs)), np.zeros((self.env.num_envs))

            # Collect rollouts until we fill the experience buffer
            for _ in range(self.config.steps_per_epoch):
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).to(self.device)
                    _, acts, vals, logps = self.policy(obs_tensor)
                    acts = acts.cpu().numpy()
                    vals = vals.cpu().numpy()
                    logps = logps.cpu().numpy()

                next_obs, rews, dones, _ = self.env.step(acts)
                self.buffer.store(obs, acts, rews, dones, vals, logps)
                J += rews
                ep_steps += 1
                total_steps += self.env.num_envs

                for idx, done in enumerate(dones):
                    if done:
                        self.writer.add_scalar('train/episode_return', J[idx], total_episodes)
                        self.writer.add_scalar('train/episode_steps', ep_steps[idx], total_episodes)
                        J[idx] = 0
                        ep_steps[idx] = 0
                        total_episodes += 1

                obs = next_obs

            self.writer.add_scalar('time/FPS', total_steps / (time.time() - start_time), epoch)
            self.writer.add_scalar('time/time_elapsed', time.time() - start_time, epoch)
            self.writer.add_scalar('time/total_timesteps', total_steps, epoch)

            # Use the final values to compute returns and advantages in preparation for an update
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).to(self.device)
                _, _, val, _ = self.policy(obs_tensor)
                val = val.cpu().numpy()
            self.buffer.compute_returns_and_advantage(last_values=val)
            self.update(epoch)
            self.evaluate(epoch)


    def evaluate(self, epoch):
        def run_episode():
            obs = self.eval_env.reset()
            steps, J = 0, 0
            done = False
            while not done:
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).to(self.device)
                    _, act, _, _ = self.policy(obs_tensor, deterministic=True)
                    act = act.cpu().numpy()
                obs, rew, done, info = self.eval_env.step(act)
                steps += 1
                J += rew
            return J, steps

        returns, ep_lengths = [], []
        for _ in range(self.config.eval_episodes):
            ret, steps = run_episode()
            returns.append(ret)
            ep_lengths.append(steps)

        avg_return = np.mean(returns)
        avg_steps = np.mean(ep_lengths)

        self.writer.add_scalar('eval/episode_return', avg_return, epoch)
        self.writer.add_scalar('eval/episode_steps', avg_steps, epoch)
        print(f'Epoch {epoch} Evaluation: AvgReturn {avg_return:.1f} AvgSteps {avg_steps:.1f}')


    def update(self, epoch):
        """ Update policy using the currently gathered rollout buffer. """
        rollout_data = self.buffer.get()
        early_stopping_step = self.config.gradient_steps_per_update

        for i in range(self.config.gradient_steps_per_update):
            # Forward pass
            pi, _, values, log_prob = self.policy(rollout_data['obs'], rollout_data['act'])

            # Normalize advantage
            advantages = rollout_data['adv']
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # ratio between old and new policy, should be one at the first iteration
            ratio = torch.exp(log_prob - rollout_data['logp'])

            # clipped surrogate loss
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

            # Value loss using the TD(gae_lambda) target
            value_loss = F.mse_loss(rollout_data['ret'], values)

            # Entropy loss favor exploration
            entropy_loss = -torch.mean(pi.entropy())
            loss = policy_loss + self.config.ent_coef * entropy_loss + self.config.vf_coef * value_loss

            # Calculate approximate form of reverse KL Divergence for early stopping
            with torch.no_grad():
                log_ratio = log_prob - rollout_data['logp']
                approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()

            if approx_kl_div > 1.5 * self.config.target_kl:
                early_stopping_step = i
                break

            clip_fraction = torch.mean((torch.abs(ratio - 1) > self.config.clip_ratio).float())

            self.writer.add_scalar('train/entropy_loss', entropy_loss.item(), self.gradient_steps)
            self.writer.add_scalar('train/policy_loss', policy_loss.item(), self.gradient_steps)
            self.writer.add_scalar('train/value_loss', value_loss.item(), self.gradient_steps)
            self.writer.add_scalar('train/clip_fraction', clip_fraction.item(), self.gradient_steps)
            self.writer.add_scalar('train/loss', loss.item(), self.gradient_steps)
            self.writer.add_scalar('train/approx_kl_div', approx_kl_div, self.gradient_steps)
            self.gradient_steps += 1

            self.policy_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_norm_clip)
            self.policy_optimizer.step()

        self.writer.add_scalar('train/early_stopping_step', early_stopping_step, epoch)

"""
This model taken from: https://github.com/karpathy/minGPT/blob/master/mingpt/model.py

GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
from absl import logging
import numpy as np
from torch.distributions.normal import Normal

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, obs_size, action_size, block_size, **kwargs):
        self.obs_size = obs_size
        self.action_size = action_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

    def to_json(self, output_fname):
        with open(output_fname, 'w') as f:
            f.write(json.dumps(self.__dict__))
    
    @staticmethod
    def from_json(fname):
        with open(fname, 'r') as f:
            kwargs = json.loads(f.read())
            return GPTConfig(**kwargs)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GaussianHead(nn.Module):

    def __init__(self, input_dim, act_dim):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std_layer = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_layer = nn.Linear(input_dim, act_dim, bias=False)

    def _distribution(self, x):
        mu = self.mu_layer(x)
        std = torch.exp(self.log_std_layer)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution

    def forward(self, x, act=None, deterministic=True):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(x)
        logp_a = None
        if deterministic:
            pi_action = pi.mean
        else:
            pi_action = pi.rsample()
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        else:
            logp_a = self._log_prob_from_distribution(pi, pi_action)
        return pi, pi_action, logp_a


LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianHead(nn.Module):

    def __init__(self, input_dim, act_dim, act_limit):
        super().__init__()
        self.mu_layer = nn.Linear(input_dim, act_dim, bias=False)
        self.log_std_layer = nn.Linear(input_dim, act_dim, bias=False)
        self.act_limit = act_limit


    def forward(self, obs, act=None, deterministic=False, with_logprob=True):
        mu = self.mu_layer(obs)
        log_std = self.log_std_layer(obs)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)

        if act is None:
            if deterministic:
                # Only used for evaluating policy at test time.
                pi_action = mu
            else:
                pi_action = pi_distribution.rsample()
        else:
            pi_action = act

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=-1)
        else:
            logp_pi = None

        if act is None:
            pi_action = torch.tanh(pi_action)
            pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.tok_emb = nn.Linear(config.obs_size, config.n_embd)
        # self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        # self.head = nn.Linear(config.n_embd, config.action_size, bias=False)
        self.head = SquashedGaussianHead(config.n_embd, config.action_size, act_limit=1)

        self.block_size = config.block_size
        self.observables = config.observables
        self.apply(self._init_weights)
        self.criterion = nn.MSELoss()
        # self.criterion = nn.L1Loss()

        logging.info("%s number of parameters: %e", self.__class__.__name__, sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None):
        b, t, d = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        # logits = self.head(x)
        logits, logp_a = self.head(x, act=targets, deterministic=True)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            # loss = self.criterion(logits, targets)
            # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = -logp_a

        return logits, loss


class FFConfig:
    """ base GPT config, params common to all GPT versions """
    hidden_size = 1024

    def __init__(self, obs_size, action_size, block_size, **kwargs):
        assert block_size == 1, f"FFNet requires block_size=1."
        self.obs_size = obs_size
        self.action_size = action_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

    def to_json(self, output_fname):
        with open(output_fname, 'w') as f:
            f.write(json.dumps(self.__dict__))
    
    @staticmethod
    def from_json(fname):
        with open(fname, 'r') as f:
            kwargs = json.loads(f.read())
            return FFConfig(**kwargs)


class FFNet(nn.Module):
    """ Fully connected baseline. Modeled after the network used in Comic. """

    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.obs_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.action_size)
        )
        self.criterion = nn.MSELoss()
        # self.criterion = nn.L1Loss()
        self.block_size = config.block_size
        self.observables = config.observables

        logging.info("%s number of parameters: %e", self.__class__.__name__, sum(p.numel() for p in self.parameters()))

    def configure_optimizers(self, train_config):
        optimizer = torch.optim.AdamW(self.mlp.parameters(), lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer        


    def forward(self, x, targets=None):
        logits = self.mlp(x)

        loss = None
        if targets is not None:
            loss = self.criterion(logits, targets)

        return logits, loss


class ActorCritic(nn.Module):
    """ The default Actor-Critic network used in Stable Baselines. """
    def __init__(self, obs_size, action_size, hidden_size=64):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.value_net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.policy_head = GaussianHead(hidden_size, action_size)

        logging.info("%s number of parameters: %e", self.__class__.__name__, sum(p.numel() for p in self.parameters()))       

    def forward(self, x, act=None, deterministic=False):
        value = self.value_net(x)
        z = self.policy_net(x)
        pi, act, logp_a = self.policy_head(z, act=act, deterministic=deterministic)
        return pi, act, value.squeeze(), logp_a
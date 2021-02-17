# Copyright 2020 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Define reward function options for reference pose tasks."""

import collections

from dm_control.utils import transformations as tr
import numpy as np

RewardFnOutput = collections.namedtuple('RewardFnOutput',
                                        ['reward', 'debug', 'reward_terms'])


def bounded_quat_dist(source, target):
  """Computes a quaternion distance limiting the difference to a max of pi.

  This is achieved by taking the minimum of d(source, target) and
  d(source, -target).

  Args:
    source: a quaternion
    target: another quaternion

  Returns:
    quaternion distance.
  """
  default_dist = tr.quat_dist(source, target)
  anti_dist = tr.quat_dist(source, -np.asarray(target))
  min_dist = np.minimum(default_dist, anti_dist)
  return min_dist


def sort_dict(d):
  return collections.OrderedDict(sorted(d.items()))


def compute_squared_differences(walker_features, reference_features,
                                exclude_keys=()):
  """Computes squared differences of features."""
  squared_differences = {}
  for k in walker_features:
    if k not in exclude_keys:
      if 'quaternion' not in k:
        squared_differences[k] = np.sum(
            (walker_features[k] - reference_features[k])**2)
  quat_dists = np.array([
      bounded_quat_dist(w, r)
      for w, r in zip(walker_features['body_quaternions'],
                      reference_features['body_quaternions'])
  ])
  squared_differences['body_quaternions'] = np.sum(quat_dists**2)
  squared_differences['quaternion'] = bounded_quat_dist(
      walker_features['quaternion'], reference_features['quaternion'])**2
  return squared_differences


def termination_reward_fn(termination_error, termination_error_threshold,
                          **unused_kwargs):
  """Termination error.

  This reward is intended to be used in conjunction with the termination error
  calculated in the task. Due to terminations if error > error_threshold this
  reward will be in [0, 1].

  Args:
    termination_error: termination error computed in tracking task
    termination_error_threshold: task termination threshold
    unused_kwargs: unused_kwargs

  Returns:
    RewardFnOutput tuple containing reward, debug information and reward terms.
  """
  debug_terms = {
      'termination_error': termination_error,
      'termination_error_threshold': termination_error_threshold
  }
  termination_reward = 1 - termination_error / termination_error_threshold
  return RewardFnOutput(reward=termination_reward, debug=debug_terms,
                        reward_terms=sort_dict(
                            {'termination': termination_reward}))


def debug(reference_features, walker_features, **unused_kwargs):
  debug_terms = compute_squared_differences(walker_features, reference_features)
  return RewardFnOutput(reward=0.0, debug=debug_terms, reward_terms=None)


def multi_term_pose_reward_fn(walker_features, reference_features,
                              **unused_kwargs):
  """A reward based on com, body quaternions, joints velocities & appendages."""
  differences = compute_squared_differences(walker_features, reference_features)
  com = .1 * np.exp(-10 * differences['center_of_mass'])
  joints_velocity = 1.0 * np.exp(-0.1 * differences['joints_velocity'])
  appendages = 0.15 * np.exp(-40. * differences['appendages'])
  body_quaternions = 0.65 * np.exp(-2 * differences['body_quaternions'])
  terms = {
      'center_of_mass': com,
      'joints_velocity': joints_velocity,
      'appendages': appendages,
      'body_quaternions': body_quaternions
  }
  reward = sum(terms.values())
  return RewardFnOutput(reward=reward, debug=terms,
                        reward_terms=sort_dict(terms))


def comic_reward_fn(termination_error, termination_error_threshold,
                    walker_features, reference_features, **unused_kwargs):
  """A reward that mixes the termination_reward and multi_term_pose_reward.

  This reward function was used in
    Hasenclever et al.,
    CoMic: Complementary Task Learning & Mimicry for Reusable Skills,
    International Conference on Machine Learning, 2020.
    [https://proceedings.icml.cc/static/paper_files/icml/2020/5013-Paper.pdf]

  Args:
    termination_error: termination error as described
    termination_error_threshold: threshold to determine whether to terminate
      episodes. The threshold is used to construct a reward between [0, 1]
      based on the termination error.
    walker_features: Current features of the walker
    reference_features: features of the current reference pose
    unused_kwargs: unused addtional keyword arguments.

  Returns:
    RewardFnOutput tuple containing reward, debug terms and reward terms.
  """
  termination_reward, debug_terms, termination_reward_terms = (
      termination_reward_fn(termination_error, termination_error_threshold))
  mt_reward, mt_debug_terms, mt_reward_terms = multi_term_pose_reward_fn(
      walker_features, reference_features)
  debug_terms.update(mt_debug_terms)
  reward_terms = {k: 0.5 * v for k, v in termination_reward_terms.items()}
  reward_terms.update(
      {k: 0.5 * v for k, v in mt_reward_terms.items()})
  return RewardFnOutput(
      reward=0.5 * termination_reward + 0.5 * mt_reward,
      debug=debug_terms,
      reward_terms=sort_dict(reward_terms))


def com_reward_fn(termination_error, termination_error_threshold,
                  walker_features, reference_features, **unused_kwargs):
  """ A cost function that only uses center of mass. """
  differences = compute_squared_differences(walker_features, reference_features)
  return RewardFnOutput(
      reward=differences['center_of_mass'], 
      debug=differences, 
      reward_terms=None)


def quat_reward_fn(termination_error, termination_error_threshold,
                  walker_features, reference_features, **unused_kwargs):
  """ A cost function that only uses quaternions. """
  differences = compute_squared_differences(walker_features, reference_features)
  return RewardFnOutput(
      reward=differences['body_quaternions'], 
      debug=differences, 
      reward_terms=None)


def kumquat_reward_fn(termination_error, termination_error_threshold,
                       walker_features, reference_features, **unused_kwargs):
  """ A cost function that uses center of mass and quaternions. """
  differences = compute_squared_differences(walker_features, reference_features)
  return RewardFnOutput(
      reward=3*differences['center_of_mass'] + differences['body_quaternions'], 
      debug=differences, 
      reward_terms=None)


def supreme_reward_fn(termination_error, termination_error_threshold,
                       walker_features, reference_features, **unused_kwargs):
  """ A cost function that uses center of mass and quaternions. """
  differences = compute_squared_differences(walker_features, reference_features)
  costs = [
    1 * differences['position'],
    0 * differences['joints'],
    1 * differences['center_of_mass'],
    20 * differences['end_effectors'],
    0 * differences['appendages'],
    0 * differences['body_positions'],
    0 * differences['velocity'],
    0 * differences['angular_velocity'],
    0 * differences['joints_velocity'],
    .5 * differences['body_quaternions'],
    0 * differences['quaternion'],
  ]
  return RewardFnOutput(
      reward=sum(costs),
      debug=differences, 
      reward_terms=None)


_REWARD_FN = {
    'termination_reward': termination_reward_fn,
    'multi_term_pose_reward': multi_term_pose_reward_fn,
    'comic': comic_reward_fn,
    'com': com_reward_fn,
    'quat': quat_reward_fn,
    'kumquat': kumquat_reward_fn,
    'supreme': supreme_reward_fn,
}

_REWARD_CHANNELS = {
    'termination_reward': ('termination',),
    'multi_term_pose_reward':
        ('appendages', 'body_quaternions', 'center_of_mass', 'joints_velocity'),
    'comic': ('appendages', 'body_quaternions', 'center_of_mass', 'termination',
              'joints_velocity'),
    'com': ('center_of_mass'),
    'quat': ('body_quaternions'),
    'kumquat': ('center_of_mass', 'body_quaternions'),
    'supreme': ('center_of_mass')
}


def get_reward(reward_key):
  if reward_key not in _REWARD_FN:
    raise ValueError('Requested loss %s, which is not a valid option.' %
                     reward_key)

  return _REWARD_FN[reward_key]


def get_reward_channels(reward_key):
  if reward_key not in _REWARD_CHANNELS:
    raise ValueError('Requested loss %s, which is not a valid option.' %
                     reward_key)

  return _REWARD_CHANNELS[reward_key]

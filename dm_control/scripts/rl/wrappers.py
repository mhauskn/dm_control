# Adapted from
# https://github.com/denisyarats/dmc2gym/blob/master/dmc2gym/wrappers.py

import numpy as np
from gym import core, spaces
from dm_env import specs

from dm_control import composer
from dm_control.locomotion.arenas import floors
from dm_control.locomotion.mocap import cmu_mocap_data
from dm_control.locomotion.tasks.reference_pose import tracking, types
from dm_control.locomotion.walkers import cmu_humanoid

def _spec_to_box(spec):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = np.int(np.prod(s.shape))
        if isinstance(s, specs.BoundedArray):
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros
        if isinstance(s, specs.Array):
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        raise NotImplementedError

    mins, maxs = [], []
    for s in spec:
        if s.dtype == np.int64:
            continue # TODO: implement
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0)
    high = np.concatenate(maxs, axis=0)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=np.float32)


class LocomotionWrapper(core.Env):
    def __init__(
        self,
        ref_steps=(1,), #(1,2,3,4,5),
        dataset=None,
        task_kwargs=None,
        environment_kwargs=None,
        frame_skip=1,

        # for rendering
        height=84,
        width=84,
        camera_id=0,
    ):
        dataset = dataset or types.ClipCollection(ids=['CMU_016_22'])
        task_kwargs = task_kwargs or {}
        environment_kwargs = environment_kwargs or {}

        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._frame_skip = frame_skip

        # create task
        self._env = self._create_env(ref_steps, dataset, task_kwargs, environment_kwargs)

        # true and normalized action spaces
        self._true_action_space = _spec_to_box([self._env.action_spec()])
        self._norm_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self._true_action_space.shape,
            dtype=np.float32
        )

        # create observation space
        obs_spaces = dict()
        for k, v in self._env.observation_spec().items():
            if v.dtype == np.int64: # clip ID
                obs_spaces[k] = spaces.Discrete(len(dataset.ids))
            elif np.prod(v.shape) > 0:
                obs_spaces[k] = spaces.Box(-np.infty, np.infty, shape=(np.prod(v.shape),))
        self._observation_space = spaces.Dict(obs_spaces)

        # set seed
        self.seed(seed=task_kwargs.get('random', 1))

        self._current_joint_pos = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self, time_step):
        obs = time_step.observation
        self._current_joint_pos = obs['walker/joints_pos'].ravel().copy()
        return {k: obs[k].ravel() for k in self.observation_space.spaces}

    def _convert_action(self, action):
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action

    def _create_env(self, ref_steps, dataset, task_kwargs, environment_kwargs):
        walker = cmu_humanoid.CMUHumanoidPositionControlledV2020
        arena = floors.Floor()
        task = tracking.MultiClipMocapTracking(
            walker,
            arena,
            cmu_mocap_data.get_path_for_cmu(version='2020'),
            ref_steps,
            dataset,
            **task_kwargs
        )
        env = composer.Environment(
            task=task,
            **environment_kwargs
        )

        return env

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._norm_action_space

    def seed(self, seed=None):
        self._true_action_space.seed(seed)
        self._norm_action_space.seed(seed)
        self._observation_space.seed(seed)

    def step(self, action):
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)

        # Action corresponds to change in joint angle. We convert to desired joint angles.
        #action = np.clip(action + self._current_joint_pos, -1, 1)

        reward = 0
        for _ in range(self._frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            done = time_step.last()
            if done:
                break
        obs = self._get_obs(time_step)
        extra = dict(internal_state=self._env.physics.get_state().copy(),
                     discount=time_step.discount)
        return obs, reward, done, extra

    def reset(self):
        time_step = self._env.reset()
        return self._get_obs(time_step)

    # pylint: disable=arguments-differ
    def render(self, mode='rgb_array', height=None, width=None, camera_id=0):
        assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(
            height=height, width=width, camera_id=camera_id
        )

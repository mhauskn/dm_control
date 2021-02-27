import math
import resource
import numpy as np
from dm_control import viewer
from solver import build_env, evaluate_and_get_physics_data, set_task_state, evaluate, CustomInit
import dm_control.locomotion.tasks.reference_pose.rewards as rewards
from absl import app
from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from absl import logging

ACTION_PATH = "pt/termination_reward/search_termination_reward_additionalsegs_2_segsize_8_optimizeriters_4/opt_acts_1.npy"

class SolverTest(parameterized.TestCase):

    def test_double_used_physics_state(self):
        env = build_env(reward_type='termination', ghost_offset=0, clip_name="CMU_056_01")
        actions = np.load(ACTION_PATH)
        J, physics_data = evaluate_and_get_physics_data(env, actions)

        cInit = CustomInit(20, physics_data[20])
        J1 = evaluate(env, actions[20:], cInit)
        J2 = evaluate(env, actions[20:], cInit)
        assert math.isclose(J1, J2), "Expected equality, got J1={} J2={}".format(J1, J2)


    def test_set_state(self):
        env = build_env(reward_type='termination', ghost_offset=0, clip_name="CMU_056_01")
        actions = np.load(ACTION_PATH)

        J = 0.
        physics_data, physics_states = [], []
        env.reset()
        for act in actions:
            physics_data.append(env.physics.copy().data)
            physics_states.append(env.physics.get_state().copy())
            time_step = env.step(act)
            J += time_step.reward
            if time_step.last():
                break

        # Check that we get the same return when initializing without reset
        set_task_state(env, start_step=0, physics_data=physics_data[0])
        Jnew = 0.
        for i, act in enumerate(actions):
            assert np.allclose(env.physics.get_state(), 
                               physics_states[i], 
                               rtol=1.e-10, atol=1.e-10)
            time_step = env.step(act)
            Jnew += time_step.reward
            if time_step.last():
                break
        assert math.isclose(J, Jnew), "Got J={} Expected J={}".format(Jnew, J)

        # Test that we can set an arbitrary state and get the correct next state
        n = len(actions) // 2
        set_task_state(env, n, physics_data[n])
        env.step(actions[n])
        assert np.allclose(env.physics.get_state(), 
                           physics_states[n+1],
                           rtol=1.e-10, atol=1.e-10)


    def test_memory_leak(self):
        env = build_env(reward_type='termination', ghost_offset=0, clip_name="CMU_056_01")
        actions = np.load(ACTION_PATH)
        J, physics_data = evaluate_and_get_physics_data(env, actions)
        
        mem_usage_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.
        for idx in range(1000):
            set_task_state(env, 0, physics_data[0])

        new_usage_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.
        
        assert new_usage_mb / mem_usage_mb <= 1.05, \
            'Old Memory Usage: {:.1f} Mb. New Memory Usage {:.1f} Mb'.format(
                mem_usage_mb, new_usage_mb)


if __name__ == "__main__":
  absltest.main()
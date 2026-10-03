"""
Environment construction for training workers.

This module is deliberately light: it imports Gymnasium and the simulator,
and nothing from Stable-Baselines3 or PyTorch. Parallel training runs each
environment in its own worker process, and every worker imports whatever the
pickled factory function lives in. When the factory lived beside the training
code, each of eight workers loaded PyTorch and an OpenBLAS thread pool sized
for all twenty cores, and the machine ran out of memory before the first
update. Keeping the worker side here caps each one at the simulator's own
footprint.

No action-masking wrapper is needed: SB3 fetches masks with
``get_wrapper_attr("action_masks")``, which walks the wrapper stack to the
base environment and finds ``TruckDroneEnv.action_masks`` directly.
"""

import itertools
import os

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics

from .env import TruckDroneEnv

WORKER_THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def limit_worker_threads(n=1):
    """
    Cap BLAS/OpenMP threads for processes spawned after this call. Workers
    inherit the environment, so this must run before the vector env is built.
    """
    for var in WORKER_THREAD_VARS:
        os.environ[var] = str(n)


class CycleOptions(gym.Wrapper):
    """
    Feed ``reset`` a fixed, repeating sequence of options -- instance, world
    seed, preference -- whenever the caller supplies none. SB3's evaluation
    callback resets with no arguments; this makes its episodes a fixed
    benchmark rather than a fresh random draw each time.
    """

    def __init__(self, env, options_list):
        super().__init__(env)
        self._cycle = itertools.cycle(options_list)

    def reset(self, **kwargs):
        if not kwargs.get("options"):
            kwargs["options"] = next(self._cycle)
        return self.env.reset(**kwargs)


def build_env(scenario, instance_ids, seed, env_kwargs, options_list=None):
    """One environment: simulator, optional fixed options, episode statistics."""
    env = TruckDroneEnv(scenario, instance_ids=instance_ids, **env_kwargs)
    if options_list:
        env = CycleOptions(env, options_list)
    # Provides info["episode"] = {"r", "l", "t"} at episode end, the same keys
    # SB3's Monitor writes, so training logs report episode return as usual.
    env = RecordEpisodeStatistics(env)
    env.reset(seed=seed)
    return env

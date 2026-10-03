"""
Training.

Three algorithms, compared on an identical environment so that any difference
is the learner's and not the task's:

``maskable_ppo``  PPO with invalid-action masking (sb3-contrib). Illegal
                  sorties are removed from the policy's distribution before
                  sampling, so no interaction is wasted on an infeasible launch.
``ppo``           Standard PPO, same network and hyperparameters, no mask: it
                  must learn feasibility from penalties.
``dqn``           Value-based control on the same discrete action space.

Every agent is preference-conditioned: each episode draws a time/energy/cost
weighting, shows it to the agent, and scalarises the reward with it. One
trained policy therefore covers the whole trade-off curve.

Experience is collected from several environments in parallel processes, and
each algorithm is trained from several random seeds so the results carry
seed-to-seed uncertainty, not just instance-to-instance.

The best checkpoint is chosen on a fixed set of *selection* worlds that is
disjoint from the worlds the final evaluation uses, so model selection never
sees the test set.

    python -m truckdrone.train --algo maskable_ppo --seeds 0 1 2 3 4
    python -m truckdrone.train --algo all
"""

import argparse
import functools
import json
import os
import time

from . import config
from .env import PREFERENCE_PRESETS
from .envs import build_env, limit_worker_threads
from .scenario import load_scenario, split_instances

# Stable-Baselines3 and PyTorch are imported inside the functions that need
# them, never at module level. Parallel workers import this module when they
# start (it is the __main__ they were spawned from); importing PyTorch there
# would load it once per worker -- see envs.py for what that did.

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")
RESULT_DIR = os.path.join(ROOT, "results")

ALGOS = ("maskable_ppo", "ppo", "dqn")


# ======================================================================
# Environments
# ======================================================================

def make_vec_env(scenario, instance_ids, n_envs, seed, env_kwargs):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    thunks = [functools.partial(build_env, scenario, instance_ids,
                                seed * 1000 + i, env_kwargs) for i in range(n_envs)]
    if n_envs == 1:
        return DummyVecEnv(thunks)
    limit_worker_threads(1)
    return SubprocVecEnv(thunks, start_method="spawn")


def selection_options(n_instances, env_kwargs):
    """The fixed worlds used to pick the best checkpoint."""
    prefs = (list(PREFERENCE_PRESETS.values())
             if env_kwargs.get("preference_conditioned") else [None])
    return [{"instance": i,
             "world_seed": config.SELECTION_WORLD_SEED_BASE + i,
             "preference": prefs[i % len(prefs)]} for i in range(n_instances)]


# ======================================================================
# Logging
# ======================================================================

def metrics_callback(log_dir):
    """Per-episode delivery metrics, mirrored to TensorBoard."""
    from stable_baselines3.common.callbacks import BaseCallback

    class MetricsCallback(BaseCallback):
        """Per-episode delivery metrics, mirrored to TensorBoard."""

        def __init__(self, log_dir, verbose=0):
            super().__init__(verbose)
            self.log_dir = log_dir
            self.episode_metrics = []

        def _on_step(self):
            for done, info in zip(self.locals.get("dones", []), self.locals.get("infos", [])):
                if not done or "served" not in info:
                    continue
                b = info["truck_only"]
                rec = {
                    "timestep": self.num_timesteps,
                    "route_complete": bool(info["route_complete"]),
                    "completion_pct": float(info["completion_pct"]),
                    "drone_deliveries": info["drone_deliveries"],
                    "failed_sorties": info["failed_sorties"],
                    "time_change_pct": (info["total_time_s"] - b["time_s"]) / b["time_s"] * 100,
                    "cost_change_pct": (info["cost_inr"] - b["cost_inr"]) / b["cost_inr"] * 100,
                }
                self.episode_metrics.append(rec)
                if self.logger:
                    for k in ("completion_pct", "drone_deliveries", "failed_sorties",
                              "time_change_pct", "cost_change_pct"):
                        self.logger.record("delivery/" + k, rec[k])
            return True

        def _on_training_end(self):
            if self.episode_metrics:
                with open(os.path.join(self.log_dir, "episode_metrics.json"), "w") as f:
                    json.dump(self.episode_metrics, f)

    return MetricsCallback(log_dir)


# ======================================================================
# Models
# ======================================================================

def _linear(start):
    return lambda progress_remaining: start * progress_remaining


def build_model(algo, env, log_dir, seed):
    """
    The two PPO variants share every hyperparameter so the masking comparison
    isolates masking. The entropy bonus keeps early exploration alive: in this
    action space a policy can collapse onto "always drive", which completes
    every route and never discovers that flying pays.
    """
    from stable_baselines3 import DQN, PPO

    common = dict(verbose=0, seed=seed, tensorboard_log=os.path.join(log_dir, "tb"),
                  policy_kwargs=dict(net_arch=[256, 256]))
    ppo_kw = dict(learning_rate=_linear(3e-4), n_steps=1024, batch_size=256,
                  n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                  ent_coef=0.01)
    if algo == "maskable_ppo":
        from sb3_contrib import MaskablePPO
        return MaskablePPO("MlpPolicy", env, **ppo_kw, **common)
    if algo == "ppo":
        return PPO("MlpPolicy", env, **ppo_kw, **common)
    if algo == "dqn":
        # One gradient step per vectorised step: with four parallel
        # environments that is one update per four transitions, the classic
        # DQN ratio, rather than letting more workers quietly mean fewer
        # updates per sample.
        return DQN("MlpPolicy", env, learning_rate=1e-4, buffer_size=200_000,
                   learning_starts=10_000, batch_size=128, gamma=0.99,
                   train_freq=1, gradient_steps=1, target_update_interval=2_000,
                   exploration_fraction=0.3, exploration_final_eps=0.05, **common)
    raise ValueError("Unknown algorithm: {!r}".format(algo))


def model_dir(tag, seed, root=MODEL_DIR):
    return os.path.join(root, tag, "seed{}".format(seed))


def train(algo, scenario, timesteps, seed=0, env_kwargs=None, tag=None,
          n_envs=config.N_ENVS, root=MODEL_DIR):
    import torch
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    torch.set_num_threads(max(1, int(os.environ.get("TRAIN_TORCH_THREADS", "2"))))
    masked = algo == "maskable_ppo"
    env_kwargs = env_kwargs or config.env_kwargs()
    log_dir = model_dir(tag or algo, seed, root)
    os.makedirs(log_dir, exist_ok=True)

    train_ids, eval_ids = split_instances(scenario)
    env = make_vec_env(scenario, train_ids, n_envs, seed, env_kwargs)
    sel_env = DummyVecEnv([functools.partial(
        build_env, scenario, eval_ids, seed + 777, env_kwargs,
        selection_options(len(eval_ids), env_kwargs))])

    if masked:
        from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback as Eval
    else:
        Eval = EvalCallback
    eval_cb = Eval(sel_env, best_model_save_path=os.path.join(log_dir, "best"),
                   log_path=log_dir, eval_freq=max(1, config.EVAL_FREQ // n_envs),
                   n_eval_episodes=config.N_EVAL_EPISODES, deterministic=True,
                   render=False, verbose=0)
    model = build_model(algo, env, log_dir, seed)

    t0 = time.perf_counter()
    print("[{} seed {}] training {:,} steps on {} envs".format(
        tag or algo, seed, timesteps, n_envs), flush=True)
    model.learn(total_timesteps=timesteps, callback=[eval_cb, metrics_callback(log_dir)])
    model.save(os.path.join(log_dir, "final.zip"))
    minutes = (time.perf_counter() - t0) / 60.0
    env.close()
    sel_env.close()

    summary = {"algo": algo, "tag": tag or algo, "seed": seed,
               "timesteps": timesteps, "train_minutes": minutes,
               "fps": timesteps / max(minutes * 60, 1e-9), "env_kwargs": env_kwargs}
    with open(os.path.join(log_dir, "training.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[{} seed {}] done in {:.1f} min ({:.0f} steps/s)".format(
        tag or algo, seed, minutes, summary["fps"]), flush=True)
    return summary


def load_policy(algo, path):
    from stable_baselines3 import DQN, PPO

    if algo == "maskable_ppo":
        from sb3_contrib import MaskablePPO
        return MaskablePPO.load(path, device="cpu")
    if algo == "ppo":
        return PPO.load(path, device="cpu")
    if algo == "dqn":
        return DQN.load(path, device="cpu")
    raise ValueError("Unknown algorithm: {!r}".format(algo))


def checkpoint(tag, seed, which="best", root=MODEL_DIR):
    d = model_dir(tag, seed, root)
    best = os.path.join(d, "best", "best_model.zip")
    final = os.path.join(d, "final.zip")
    if which == "best" and os.path.exists(best):
        return best
    return final if os.path.exists(final) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--algo", default="maskable_ppo", choices=list(ALGOS) + ["all"])
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--n-envs", type=int, default=config.N_ENVS)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--override", nargs="*", default=[],
                    help="env overrides as key=value, e.g. rendezvous_options=1")
    args = ap.parse_args()

    overrides = {}
    for kv in args.override:
        k, v = kv.split("=", 1)
        overrides[k] = json.loads(v)

    scenario = load_scenario(config.CITY)
    algos = list(ALGOS) if args.algo == "all" else [args.algo]
    for algo in algos:
        for seed in (args.seeds if args.seeds is not None else config.SEEDS[algo]):
            train(algo, scenario, args.timesteps or config.TIMESTEPS[algo], seed=seed,
                  env_kwargs=config.env_kwargs(**overrides), tag=args.tag or algo,
                  n_envs=args.n_envs)


if __name__ == "__main__":
    main()

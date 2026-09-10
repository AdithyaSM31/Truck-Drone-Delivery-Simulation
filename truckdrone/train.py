"""
Training entry point.

Three algorithms are supported, and the comparison between them is one of the
project's actual findings rather than boilerplate:

``maskable_ppo``
    PPO with invalid-action masking (sb3-contrib). The environment reports
    which sorties are physically legal and the policy's logits for illegal
    actions are driven to -inf before sampling. The agent therefore never
    wastes a single interaction on an infeasible dispatch.

``ppo``
    Standard PPO. Identical network and hyperparameters, but it must discover
    the feasibility rules from penalty signal alone.

``dqn``
    Value-based control, included because the action space is small and
    discrete, which is exactly where DQN is meant to be at home.

Training uses only the training half of the instance pool; the evaluation
callback that decides which checkpoint is "best" sees only the held-out half.

    python -m truckdrone.train --algo maskable_ppo --timesteps 300000
    python -m truckdrone.train --algo all --timesteps 300000
"""

import argparse
import json
import os
import time

import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from . import config
from .env import TruckDroneEnv
from .scenario import load_scenario, split_instances

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")
RESULT_DIR = os.path.join(ROOT, "results")

ALGOS = ("maskable_ppo", "ppo", "dqn")


# ======================================================================
# Environment construction
# ======================================================================

def make_env(scenario, instance_ids, masked, seed=0, env_kwargs=None):
    """
    Build one Monitor-wrapped environment.

    For the masked agent the env is additionally wrapped in ``ActionMasker``,
    which is how sb3-contrib finds the mask function. Everything else about
    the environment is identical across algorithms, so any difference in the
    results is attributable to the learning algorithm and not to the task.
    """
    env = TruckDroneEnv(scenario, instance_ids=instance_ids,
                        **(env_kwargs or config.env_kwargs()))
    if masked:
        from sb3_contrib.common.wrappers import ActionMasker
        env = ActionMasker(env, lambda e: e.unwrapped.action_masks())
    env = Monitor(env)
    env.reset(seed=seed)
    return env


# ======================================================================
# Logging callback
# ======================================================================

class MetricsCallback(BaseCallback):
    """Records per-episode delivery metrics and mirrors them to TensorBoard."""

    def __init__(self, log_dir, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.episode_metrics = []

    def _on_step(self):
        # The env returns its info dict on every step, so filtering on the
        # presence of a key would record one row per timestep. Only rows where
        # the episode actually ended are episode metrics.
        dones = self.locals.get("dones")
        if dones is None:
            dones = self.locals.get("done", [])
        for done, info in zip(dones, self.locals.get("infos", [])):
            if not done or "served" not in info:
                continue
            record = {
                "timestep": self.num_timesteps,
                "completion_pct": info["completion_pct"],
                "route_complete": bool(info["route_complete"]),
                "drone_deliveries": info["drone_deliveries"],
                "truck_deliveries": info["truck_deliveries"],
                "failed_sorties": info["failed_sorties"],
                "total_time_min": info["total_time_s"] / 60.0,
                "truck_km": info["truck_distance_m"] / 1000.0,
                "baseline_km": info["truck_only"]["distance_m"] / 1000.0,
            }
            self.episode_metrics.append(record)
            if self.logger:
                self.logger.record("delivery/completion_pct",
                                   info["completion_pct"])
                self.logger.record("delivery/drone_deliveries",
                                   info["drone_deliveries"])
                self.logger.record("delivery/failed_sorties",
                                   info["failed_sorties"])
                self.logger.record("delivery/total_time_min",
                                   info["total_time_s"] / 60.0)
                self.logger.record("delivery/truck_km",
                                   info["truck_distance_m"] / 1000.0)
        return True

    def _on_training_end(self):
        if self.episode_metrics:
            path = os.path.join(self.log_dir, "episode_metrics.json")
            with open(path, "w") as f:
                json.dump(self.episode_metrics, f, indent=2)
            print("[metrics] {} episodes -> {}".format(
                len(self.episode_metrics), path))


# ======================================================================
# Training
# ======================================================================

def build_model(algo, env, log_dir, seed=0):
    """
    Construct the learner.

    The two PPO variants share every hyperparameter so the masking ablation is
    clean. ``ent_coef=0.01`` is deliberate: with a near-degenerate action space
    the policy can collapse onto "always advance the truck" -- which solves the
    task, badly, and never explores a single sortie. The entropy bonus keeps
    that door open long enough for the agent to discover that flying pays.
    """
    tb = os.path.join(log_dir, "tb")
    common = dict(verbose=1, seed=seed, tensorboard_log=tb,
                  policy_kwargs=dict(net_arch=[128, 128]))

    if algo == "maskable_ppo":
        from sb3_contrib import MaskablePPO
        return MaskablePPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048,
                           batch_size=64, n_epochs=10, gamma=0.99,
                           gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                           **common)
    if algo == "ppo":
        return PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048,
                   batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                   clip_range=0.2, ent_coef=0.01, **common)
    if algo == "dqn":
        return DQN("MlpPolicy", env, learning_rate=1e-4, buffer_size=100_000,
                   learning_starts=5_000, batch_size=64, gamma=0.99,
                   train_freq=4, target_update_interval=1_000,
                   exploration_fraction=0.3, exploration_final_eps=0.05,
                   **common)
    raise ValueError("Unknown algorithm: {!r}".format(algo))


def train(algo, scenario, timesteps=config.TIMESTEPS, seed=config.SEED,
          model_dir=MODEL_DIR, env_kwargs=None, tag=None):
    masked = algo == "maskable_ppo"
    env_kwargs = env_kwargs or config.env_kwargs()
    log_dir = os.path.join(model_dir, tag or algo)
    os.makedirs(log_dir, exist_ok=True)

    train_ids, eval_ids = split_instances(scenario)
    env = make_env(scenario, train_ids, masked, seed=seed,
                   env_kwargs=env_kwargs)
    eval_env = make_env(scenario, eval_ids, masked, seed=seed + 1000,
                        env_kwargs=env_kwargs)

    if masked:
        from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
        eval_cls = MaskableEvalCallback
    else:
        eval_cls = EvalCallback

    eval_cb = eval_cls(eval_env, best_model_save_path=os.path.join(log_dir, "best"),
                       log_path=log_dir, eval_freq=config.EVAL_FREQ,
                       n_eval_episodes=config.N_EVAL_EPISODES,
                       deterministic=True, render=False, verbose=1)
    metrics_cb = MetricsCallback(log_dir)

    model = build_model(algo, env, log_dir, seed=seed)

    print("\n" + "=" * 62)
    print("  Training {}  ({:,} timesteps)".format(algo, timesteps))
    print("  {} drones | train instances {} | held-out eval {}".format(
        env_kwargs["n_drones"], len(train_ids), len(eval_ids)))
    print("=" * 62)

    t0 = time.perf_counter()
    model.learn(total_timesteps=timesteps, callback=[eval_cb, metrics_cb],
                progress_bar=False)
    elapsed = time.perf_counter() - t0

    final_path = os.path.join(log_dir, "final.zip")
    model.save(final_path)
    print("\n[{}] trained in {:.1f} min -> {}".format(
        algo, elapsed / 60.0, final_path))

    env.close()
    eval_env.close()
    return {"algo": algo, "timesteps": timesteps,
            "train_minutes": elapsed / 60.0, "n_drones": env_kwargs["n_drones"],
            "final_model": final_path,
            "best_model": os.path.join(log_dir, "best", "best_model.zip")}


def load_policy(algo, path):
    """Load a trained model by algorithm name."""
    if algo == "maskable_ppo":
        from sb3_contrib import MaskablePPO
        return MaskablePPO.load(path)
    if algo == "ppo":
        return PPO.load(path)
    if algo == "dqn":
        return DQN.load(path)
    raise ValueError("Unknown algorithm: {!r}".format(algo))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--algo", default="maskable_ppo",
                    choices=list(ALGOS) + ["all"])
    ap.add_argument("--timesteps", type=int, default=config.TIMESTEPS)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--drones", type=int, default=config.N_DRONES)
    args = ap.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    scenario = load_scenario()

    algos = list(ALGOS) if args.algo == "all" else [args.algo]
    summary = {}
    for algo in algos:
        summary[algo] = train(algo, scenario, timesteps=args.timesteps,
                              seed=args.seed,
                              env_kwargs=config.env_kwargs(n_drones=args.drones))

    path = os.path.join(RESULT_DIR, "training_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print("\nTraining summary -> {}".format(path))


if __name__ == "__main__":
    main()

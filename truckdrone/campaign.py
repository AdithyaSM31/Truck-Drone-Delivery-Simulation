"""
The full training campaign, run as a resumable queue of jobs.

Each job is one (algorithm, seed, configuration) and runs as its own process
with its own log. A job whose ``final.zip`` already exists is skipped, so the
campaign can be stopped and restarted without redoing finished work.

Jobs run ``--parallel`` at a time. Each job already uses several worker
processes for its environments, so the limit is memory as much as CPU: on
Windows every process reserves its memory up front, and two concurrent jobs
of eight workers fit comfortably where three do not.

    python -m truckdrone.campaign                 # everything
    python -m truckdrone.campaign --only main     # just the headline agents
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from . import config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "logs")

# Every new mechanic gets an ablation: the same agent trained without it.
# (tag, overrides). Each is one seed at 1M steps -- enough to show direction
# and size, and stated as such in the write-up.
ABLATIONS = {
    "abl_reference": {},                     # full config, same 1M budget
    "abl_rendezvous1": {"rendezvous_options": 1},
    "abl_no_replan": {"replan_tour": False},
    "abl_specialist": {"preference_conditioned": False},
    "abl_calm_training": {"stochastic": False},
    "abl_drones1": {"n_drones": 1},
    "abl_drones3": {"n_drones": 3},
    "abl_batteries1": {"batteries_per_drone": 1},
}
ABLATION_STEPS = 1_000_000


def jobs(only=None):
    out = []
    if only in (None, "main"):
        for algo in ("maskable_ppo", "ppo", "dqn"):
            n_envs = 4 if algo == "dqn" else 8
            for seed in config.SEEDS[algo]:
                out.append({"tag": algo, "algo": algo, "seed": seed,
                            "steps": config.TIMESTEPS[algo], "n_envs": n_envs,
                            "overrides": {}})
    if only in (None, "ablations"):
        for tag, ov in ABLATIONS.items():
            out.append({"tag": tag, "algo": "maskable_ppo", "seed": 0,
                        "steps": ABLATION_STEPS, "n_envs": 8, "overrides": ov})
    return out


def done(job):
    from .train import model_dir
    return os.path.exists(os.path.join(model_dir(job["tag"], job["seed"]), "final.zip"))


def run(job):
    if done(job):
        return job, "skipped", 0.0
    os.makedirs(LOG_DIR, exist_ok=True)
    log = os.path.join(LOG_DIR, "{}_seed{}.log".format(job["tag"], job["seed"]))
    cmd = [sys.executable, "-W", "ignore", "-m", "truckdrone.train",
           "--algo", job["algo"], "--seeds", str(job["seed"]),
           "--timesteps", str(job["steps"]), "--n-envs", str(job["n_envs"]),
           "--tag", job["tag"]]
    for k, v in job["overrides"].items():
        cmd += ["--override", "{}={}".format(k, json.dumps(v))]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", TRAIN_TORCH_THREADS="3")
    t0 = time.perf_counter()
    with open(log, "w") as f:
        rc = subprocess.call(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
    return job, ("ok" if rc == 0 else "FAILED rc={}".format(rc)), time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=["main", "ablations"], default=None)
    ap.add_argument("--parallel", type=int, default=2)
    args = ap.parse_args()

    queue = jobs(args.only)
    todo = [j for j in queue if not done(j)]
    print("{} jobs, {} already done, {} to run, {} at a time".format(
        len(queue), len(queue) - len(todo), len(todo), args.parallel), flush=True)
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for job, status, secs in pool.map(run, queue):
            print("{:<20} seed {}  {:<8}  {:5.1f} min".format(
                job["tag"], job["seed"], status, secs / 60), flush=True)
    print("campaign finished", flush=True)


if __name__ == "__main__":
    main()

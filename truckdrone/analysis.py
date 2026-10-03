"""
Beyond the headline table: how the trained agent behaves when the question
changes.

Four studies, each a direct answer to something promised in Review 1:

``pareto``      Multi-objective RL. One preference-conditioned agent is asked
                for every weighting on a grid over the (time, energy, cost)
                simplex. Plotting the outcomes traces the trade-off curve a
                single trained network can offer, next to the heuristics
                (which have no knob) and a specialist trained only on time.

``robustness``  Stochastic environments. The same agents in harsher worlds
                than they trained in: wind pinned at 0-12 m/s (training drew
                0-8, so 12 m/s is out of distribution) and three traffic
                regimes from calm to heavy.

``scale``       Generalisation in problem size. Trained on 15-customer days,
                tested on 10- and 20-customer days it has never seen.

``transfer``    Generalisation in place. Trained on Whitefield, Bengaluru,
                tested zero-shot on T. Nagar, Chennai -- a denser, flatter
                network with far more one-way streets.

Every number is a paired % change against the OR-Tools truck-only tour
driven through the same world, over completed routes only (see evaluate.py),
averaged over the trained seeds with a 95% interval.

    python -m truckdrone.analysis                 # all studies
    python -m truckdrone.analysis --only pareto
"""

import argparse
import itertools
import json
import os
import time

import numpy as np

from . import config
from .baselines import POLICIES
from .evaluate import across_seeds, evaluate_policy, learned_checkpoints, summarise
from .scenario import load_scenario, split_instances, with_instances

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(ROOT, "results")
OUT = os.path.join(RESULT_DIR, "analysis.json")

HEURISTICS = ("greedy", "always_nearest")
WIND_LEVELS = (0.0, 4.0, 8.0, 12.0)
TRAFFIC_LEVELS = {
    "calm": {"rush_peak": 1.0, "traffic_sigma": 0.05, "day_sigma": 0.05},
    "nominal": {},
    "heavy": {"rush_peak": 2.5, "traffic_sigma": 0.30, "day_sigma": 0.20},
}
KEEP = ("policy", "n_seeds", "n_runs", "completion_rate_pct", "time_change_pct",
        "time_change_pct_ci95", "energy_change_pct", "energy_change_pct_ci95",
        "cost_change_pct", "cost_change_pct_ci95", "co2_change_pct",
        "truck_km_change_pct", "drone_share_pct", "drone_share_pct_ci95",
        "truck_wait_min_mean", "failed_sorties_mean", "battery_deaths_total")


def simplex_grid(step=0.25):
    n = int(round(1 / step))
    return [(i / n, j / n, (n - i - j) / n)
            for i, j in itertools.product(range(n + 1), repeat=2) if i + j <= n]


def _trim(s):
    return {k: s[k] for k in KEEP if k in s}


class Agents:
    """Loads each trained checkpoint once and scores policies on demand."""

    def __init__(self, algo="maskable_ppo", world_seeds=config.EVAL_WORLD_SEEDS):
        from .train import load_policy
        self.algo = algo
        self.world_seeds = world_seeds
        self.models = [(s, load_policy(algo, p)) for s, p in learned_checkpoints(algo)]
        if not self.models:
            raise RuntimeError("No trained {} checkpoints".format(algo))

    def learned(self, scenario, ids, env_kwargs=None, preference=None, label=None):
        per_seed = []
        for seed, model in self.models:
            runs = evaluate_policy(model, scenario, ids, is_sb3=True,
                                   masked=self.algo == "maskable_ppo",
                                   env_kwargs=env_kwargs, world_seeds=self.world_seeds,
                                   preference=preference or config.HEADLINE_PREFERENCE)
            per_seed.append(summarise(runs, "{}#{}".format(self.algo, seed)))
        return _trim(across_seeds(per_seed, label or self.algo))

    def heuristic(self, name, scenario, ids, env_kwargs=None):
        runs = evaluate_policy(POLICIES[name](), scenario, ids, env_kwargs=env_kwargs,
                               world_seeds=self.world_seeds)
        return _trim(summarise(runs, name))

    def compare(self, scenario, ids, env_kwargs=None):
        out = [self.learned(scenario, ids, env_kwargs)]
        out += [self.heuristic(h, scenario, ids, env_kwargs) for h in HEURISTICS]
        return out


# ======================================================================
# Studies
# ======================================================================

def pareto(agents, scenario, ids, step=0.25):
    points = []
    for w in simplex_grid(step):
        s = agents.learned(scenario, ids, preference=w, label="w={:.2f}/{:.2f}/{:.2f}".format(*w))
        s["weights"] = {"time": w[0], "energy": w[1], "cost": w[2]}
        points.append(s)
        print("  pareto w={} time {:+.1f}%  energy {:+.1f}%  cost {:+.1f}%".format(
            w, s["time_change_pct"] or 0, s["energy_change_pct"] or 0,
            s["cost_change_pct"] or 0), flush=True)
    refs = [agents.heuristic(h, scenario, ids) for h in HEURISTICS]
    from .train import checkpoint
    from .train import load_policy
    spec = checkpoint("abl_specialist", 0)
    if spec:
        runs = evaluate_policy(load_policy("maskable_ppo", spec), scenario, ids,
                               is_sb3=True, masked=True,
                               env_kwargs=config.env_kwargs(preference_conditioned=False),
                               world_seeds=agents.world_seeds)
        refs.append(_trim(summarise(runs, "specialist (time only)")))
    return {"grid_step": step, "points": points, "references": refs}


def robustness(agents, scenario, ids):
    wind = []
    for v in WIND_LEVELS:
        kw = config.env_kwargs(wind_fixed=[v, None])
        wind.append({"wind_ms": v, "results": agents.compare(scenario, ids, kw)})
        print("  wind {:>4} m/s done".format(v), flush=True)
    traffic = []
    for name, ov in TRAFFIC_LEVELS.items():
        kw = config.env_kwargs(**ov)
        traffic.append({"regime": name, "params": ov,
                        "results": agents.compare(scenario, ids, kw)})
        print("  traffic {} done".format(name), flush=True)
    return {"wind": wind, "traffic": traffic,
            "training_wind_max_ms": config.env_kwargs().get("wind_max_ms", 8.0)}


def scale(agents, scenario):
    out = []
    for n, insts in sorted(scenario.get("extra_instances", {}).items(), key=lambda kv: int(kv[0])):
        sc = with_instances(scenario, insts)
        ids = list(range(len(insts)))
        out.append({"customers": int(n), "n_instances": len(ids),
                    "results": agents.compare(sc, ids)})
        print("  {} customers done".format(n), flush=True)
    return out


def transfer(agents, city=config.TRANSFER_CITY):
    sc = load_scenario(city)
    _, ids = split_instances(sc)
    return {"city": city, "n_instances": len(ids),
            "intersections": sc.get("network_intersections"),
            "tortuosity_mean": sc.get("tortuosity_mean"),
            "asymmetric_pair_fraction": sc.get("asymmetric_pair_fraction"),
            "results": agents.compare(sc, ids)}


STUDIES = ("pareto", "robustness", "scale", "transfer")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", choices=STUDIES, default=list(STUDIES))
    ap.add_argument("--step", type=float, default=0.25)
    args = ap.parse_args()

    results = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            results = json.load(f)
    scenario = load_scenario(config.CITY)
    _, ids = split_instances(scenario)
    agents = Agents()
    results.update({"n_seeds": len(agents.models), "world_seeds": list(agents.world_seeds),
                    "n_eval_instances": len(ids)})

    for study in args.only:
        t0 = time.perf_counter()
        print("[{}]".format(study), flush=True)
        if study == "pareto":
            results["pareto"] = pareto(agents, scenario, ids, args.step)
        elif study == "robustness":
            results["robustness"] = robustness(agents, scenario, ids)
        elif study == "scale":
            results["scale"] = scale(agents, scenario)
        elif study == "transfer":
            results["transfer"] = transfer(agents)
        print("[{}] {:.1f} min".format(study, (time.perf_counter() - t0) / 60), flush=True)
        os.makedirs(RESULT_DIR, exist_ok=True)
        with open(OUT, "w") as f:
            json.dump(results, f, indent=1, default=float)
    print("-> {}".format(OUT))


if __name__ == "__main__":
    main()

"""
What the dashboard offers: cities, policies, preferences, worlds, and which
trained seed stands in for each learned algorithm.

The live server and the static exporter both read this, so the frozen site
offers exactly what the server would.

Choosing the seed to show
-------------------------
Each learned algorithm is trained from several seeds. Showing the best one
would flatter it; showing an arbitrary one makes the demo depend on luck. The
dashboard shows the *median* seed by held-out delivery time -- the run a
reader should expect if they trained it again.
"""

import json
import os

from . import config
from .baselines import POLICIES
from .env import PREFERENCE_PRESETS
from .scenario import CITIES, load_scenario, split_instances

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(ROOT, "results")
LEARNED = ("maskable_ppo", "ppo", "dqn")

# Instances offered per city in the frozen site. The transfer city is a
# demonstration, not the main result, so it gets fewer.
STATIC_INSTANCES = {config.CITY: 20, config.TRANSFER_CITY: 10}
STATIC_WORLDS = {config.CITY: config.EVAL_WORLD_SEEDS[:3],
                 config.TRANSFER_CITY: config.EVAL_WORLD_SEEDS[:2]}
REAL_MAP_INSTANCES = {config.CITY: (0, 1, 2, 3, 4), config.TRANSFER_CITY: (0, 1, 2)}


def _evaluation():
    path = os.path.join(RESULT_DIR, "evaluation.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        ev = json.load(f)
    # Files from before the multi-world evaluation are ignored, not misread.
    return ev if "world_seeds" in ev else None


def representative_seed(algo, evaluation=None):
    """The median seed by held-out time change; else the first trained seed."""
    from .train import checkpoint

    trained = [s for s in config.SEEDS[algo] if checkpoint(algo, s)]
    if not trained:
        return None
    ev = evaluation if evaluation is not None else _evaluation()
    if ev:
        for s in ev["summaries"]:
            if s["policy"] == algo and s.get("per_seed"):
                scored = sorted((p["time_change_pct"], int(p["policy"].split("#")[1]))
                                for p in s["per_seed"]
                                if p.get("time_change_pct") is not None)
                scored = [x for x in scored if x[1] in trained]
                if scored:
                    return scored[(len(scored) - 1) // 2][1]
    return trained[0]


def learned_models(evaluation=None):
    """{algo: (seed, checkpoint path)} for every trained algorithm."""
    from .train import checkpoint

    out = {}
    for algo in LEARNED:
        seed = representative_seed(algo, evaluation)
        if seed is not None:
            out[algo] = (seed, checkpoint(algo, seed))
    return out


def preferences_for(policy):
    """Learned agents are preference-conditioned; MaskablePPO shows the full
    range, the comparison agents just the headline setting."""
    if policy == "maskable_ppo":
        return list(PREFERENCE_PRESETS)
    if policy in LEARNED:
        return [config.HEADLINE_PREFERENCE]
    return []


def info(static=False):
    ev = _evaluation()
    models = learned_models(ev)
    policies = list(models) + list(POLICIES)
    cities = {}
    for city in (config.CITY, config.TRANSFER_CITY):
        try:
            sc = load_scenario(city)
        except FileNotFoundError:
            continue
        n_eval = len(split_instances(sc)[1])
        n = min(n_eval, STATIC_INSTANCES[city]) if static else n_eval
        cities[city] = {
            "name": CITIES[city]["name"],
            "intersections": sc.get("network_intersections", 0),
            "segments": sc.get("network_segments", 0),
            "tortuosity": round(sc.get("tortuosity_mean", 0.0), 2),
            "policies": policies,
            "n_instances": n,
            "worlds": list(STATIC_WORLDS[city] if static else config.EVAL_WORLD_SEEDS),
            "preferences": {p: preferences_for(p) for p in policies},
            "real_maps": [i for i in REAL_MAP_INSTANCES[city] if i < n],
            "transfer": city != config.CITY,
        }
    first = load_scenario(config.CITY)
    return {
        "heuristics": list(POLICIES),
        "trained": list(models),
        "seeds": {a: s for a, (s, _) in models.items()},
        "n_customers": len(first["instances"][0]["customers"]),
        "cities": cities,
    }


def results_payload():
    """The evaluation table and the extra analyses, trimmed for the browser."""
    ev = _evaluation()
    if ev is None:
        return None
    out = {"n_eval_instances": ev["n_eval_instances"], "world_seeds": ev["world_seeds"],
           "n_customers": ev["n_customers"], "preference": ev["preference"],
           "summaries": [{k: v for k, v in s.items() if k != "per_seed"}
                         for s in ev["summaries"]],
           "significance": ev.get("significance", {})}
    for name in ("analysis", "ablation"):
        path = os.path.join(RESULT_DIR, name + ".json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            data.pop("runs", None)
            out[name] = data
    return out

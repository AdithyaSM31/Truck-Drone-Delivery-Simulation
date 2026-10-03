"""
Ablations: what does each mechanic actually buy?

Every new piece of the environment gets one: the same MaskablePPO agent,
trained for the same budget (1M steps, seed 0) with that one thing changed.
The comparison point is ``abl_reference`` -- the full configuration trained for
the *same* 1M steps -- not the 2M-step headline agents, so a gap is the
mechanic's and not the budget's.

Two kinds of change are measured differently:

* **Capabilities** (rendezvous choice, re-planning, fleet size, spare packs)
  change the world itself, so the ablated agent is scored in its own world,
  and the greedy heuristic is scored there too. That separates "the task got
  easier/harder" (greedy moves) from "the agent exploits the mechanic"
  (the agent moves more than greedy).
* **Training choices** (preference conditioning, training in a calm world)
  leave the task unchanged, so the ablated agent is scored in the standard
  world, exactly like the reference.

One seed per ablation shows direction and rough size, not a confidence
interval; the write-up says so.

    python -m truckdrone.ablation
"""

import argparse
import json
import os

from . import config
from .baselines import POLICIES
from .campaign import ABLATIONS
from .evaluate import evaluate_policy, paired_tests, summarise
from .scenario import load_scenario, split_instances

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(ROOT, "results")
OUT = os.path.join(RESULT_DIR, "ablation.json")

# tag -> (what it removes or changes, kind)
DESCRIPTIONS = {
    "abl_reference": ("Full configuration, same 1M-step budget", "reference"),
    "abl_rendezvous1": ("Drone must meet the truck at the next stop (no rendezvous choice)",
                        "capability"),
    "abl_no_replan": ("Truck tour never re-optimised after drone deliveries", "capability"),
    "abl_drones1": ("One drone instead of two", "capability"),
    "abl_drones3": ("Three drones instead of two", "capability"),
    "abl_batteries1": ("No spare battery: one pack per drone", "capability"),
    "abl_specialist": ("Trained for time only (no preference conditioning)", "training"),
    "abl_calm_training": ("Trained without traffic or wind, tested with them", "training"),
}
KEEP = ("completion_rate_pct", "time_change_pct", "energy_change_pct", "cost_change_pct",
        "drone_share_pct", "truck_wait_min_mean", "failed_sorties_mean",
        "battery_swaps_mean", "battery_deaths_total")


def _trim(s):
    return {k: s.get(k) for k in KEEP}


def eval_env_kwargs(tag):
    """The world an ablated agent is scored in, and how its policy is asked."""
    ov = dict(ABLATIONS.get(tag, {}))
    kind = DESCRIPTIONS[tag][1]
    if kind == "training":
        # Same task as the reference. A specialist has no preference input,
        # so it is built without one; a calm-trained agent meets the real
        # (stochastic) world.
        ov.pop("stochastic", None)
    return config.env_kwargs(**ov)


def main():
    from .train import checkpoint, load_policy

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", default="best", choices=["best", "final"])
    args = ap.parse_args()

    scenario = load_scenario(config.CITY)
    _, ids = split_instances(scenario)
    rows, runs = [], {}
    greedy_cache = {}

    for tag, (desc, kind) in DESCRIPTIONS.items():
        path = checkpoint(tag, 0, args.which)
        if not path:
            print("  [skip] {} not trained".format(tag))
            continue
        kw = eval_env_kwargs(tag)
        r = evaluate_policy(load_policy("maskable_ppo", path), scenario, ids,
                            is_sb3=True, masked=True, env_kwargs=kw)
        runs[tag] = r
        key = json.dumps(kw, sort_keys=True)
        if key not in greedy_cache:
            greedy_cache[key] = summarise(
                evaluate_policy(POLICIES["greedy"](), scenario, ids, env_kwargs=kw), "greedy")
        row = {"tag": tag, "description": desc, "kind": kind,
               "change": ABLATIONS.get(tag, {}),
               "agent": _trim(summarise(r, tag)),
               "greedy_same_world": _trim(greedy_cache[key])}
        rows.append(row)
        a, g = row["agent"]["time_change_pct"], row["greedy_same_world"]["time_change_pct"]
        print("  {:<18} agent {:>+6.1f}%  greedy {:>+6.1f}%  ({})".format(
            tag, a if a is not None else float("nan"),
            g if g is not None else float("nan"), desc), flush=True)

    ref = next((r for r in rows if r["tag"] == "abl_reference"), None)
    if ref:
        for row in rows:
            for k in ("time_change_pct", "cost_change_pct", "energy_change_pct"):
                a, b = row["agent"][k], ref["agent"][k]
                row.setdefault("vs_reference_pp", {})[k] = (
                    None if a is None or b is None else a - b)
            ga, gb = row["greedy_same_world"]["time_change_pct"], ref["greedy_same_world"]["time_change_pct"]
            row["greedy_vs_reference_pp"] = None if ga is None or gb is None else ga - gb
        # Training-choice ablations share the reference's world, so their
        # episodes pair up exactly and can be tested.
        same_world = [r["tag"] for r in rows if r["kind"] == "training"]
        if same_world:
            tests = paired_tests(runs, "abl_reference", same_world, "time_min")
            for t in tests:
                next(r for r in rows if r["tag"] == t["opponent"])["paired_test_time"] = t

    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"steps": 1_000_000, "seed": 0, "n_eval_instances": len(ids),
                   "world_seeds": list(config.EVAL_WORLD_SEEDS),
                   "preference": config.HEADLINE_PREFERENCE, "rows": rows}, f, indent=1)
    print("-> {}".format(OUT))


if __name__ == "__main__":
    main()

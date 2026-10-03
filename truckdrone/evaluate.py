"""
Evaluation.

Every policy is scored on the 20 held-out instances, each in several fixed
*worlds* (traffic, wind, start hour), and compared with the OR-Tools
truck-only tour driven through the very same world. Comparisons are paired by
(instance, world), so both the delivery problem and the weather cancel out.

Learned agents are trained from several random seeds, and every number for
them is reported as a mean across seeds with a 95% confidence interval -- the
uncertainty that matters for "would this method work again", not just "did
this one run work".

WHY THIS FILE IS CAREFUL ABOUT INCOMPLETE ROUTES
------------------------------------------------
A route that abandons customers finishes fast. An earlier version of this
project reported a 15% saving that was almost entirely two runs which
abandoned 13 of 15 packages. Completion is therefore reported on its own, and
time, energy and cost are averaged over completed routes only; a policy that
completes too few gets no delivery statistic at all.

    python -m truckdrone.evaluate
"""

import argparse
import json
import os

import numpy as np

from . import config
from .baselines import POLICIES
from .env import PREFERENCE_PRESETS, TruckDroneEnv
from .scenario import load_scenario, split_instances

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(ROOT, "results")
MIN_COMPLETED_FOR_STATS = 5
METRICS = ("time", "energy", "cost", "co2", "truck_km")


# ======================================================================
# Rollouts
# ======================================================================

def act(policy, obs, env, is_sb3, masked):
    if not is_sb3:
        return int(policy.predict(obs, env))
    if masked:
        a, _ = policy.predict(obs, action_masks=env.action_masks(), deterministic=True)
    else:
        a, _ = policy.predict(obs, deterministic=True)
    return int(a)


def run_episode(env, policy, options, is_sb3=False, masked=False):
    obs, _ = env.reset(options=options)
    if hasattr(policy, "reset"):
        policy.reset()
    done, steps = False, 0
    while not done:
        obs, _, te, tr, info = env.step(act(policy, obs, env, is_sb3, masked))
        done = te or tr
        steps += 1
    info = dict(info)
    info["steps"] = steps
    return info


def _record(info, instance_id, world_seed):
    b = info["truck_only"]
    return {
        "instance": int(instance_id), "world_seed": int(world_seed),
        "route_complete": bool(info["route_complete"]),
        "completion_pct": float(info["completion_pct"]),
        "customers": int(info["total_customers"]),
        "time_min": info["total_time_s"] / 60.0, "base_time_min": b["time_s"] / 60.0,
        "energy_kwh": info["total_energy_wh"] / 1000.0,
        "base_energy_kwh": b["energy_wh"] / 1000.0,
        "cost_inr": info["cost_inr"], "base_cost_inr": b["cost_inr"],
        "co2_kg": info["co2_kg"], "base_co2_kg": b["co2_kg"],
        "truck_km": info["truck_distance_m"] / 1000.0,
        "base_truck_km": b["distance_m"] / 1000.0,
        "drone_km": info["drone_distance_m"] / 1000.0,
        "drone_deliveries": int(info["drone_deliveries"]),
        "failed_sorties": int(info["failed_sorties"]),
        "battery_deaths": int(info["battery_deaths"]),
        "battery_swaps": int(info["battery_swaps"]),
        "min_battery_pct": float(info["min_battery_pct"]),
        "truck_wait_min": info["truck_wait_time_s"] / 60.0,
        "late_rendezvous": int(info["late_rendezvous"]),
        "replans": int(info["replans"]),
        "wind_ms": float(info["wind_ms"]),
    }


def evaluate_policy(policy, scenario, instance_ids, is_sb3=False, masked=False,
                    env_kwargs=None, world_seeds=config.EVAL_WORLD_SEEDS,
                    preference=config.HEADLINE_PREFERENCE):
    """Score one policy on every held-out instance in every world."""
    pref = PREFERENCE_PRESETS.get(preference, preference)
    env = TruckDroneEnv(scenario, instance_ids=instance_ids,
                        **(env_kwargs or config.env_kwargs()))
    runs = []
    for i, inst in enumerate(instance_ids):
        for ws in world_seeds:
            info = run_episode(env, policy, {"instance": i, "world_seed": ws,
                                             "preference": pref}, is_sb3, masked)
            runs.append(_record(info, inst, ws))
    env.close()
    return runs


# ======================================================================
# Summaries
# ======================================================================

def _pct(rows, key):
    """Mean paired % change against the truck-only baseline in the same world."""
    return float(np.mean([(r[key] - r["base_" + key]) / r["base_" + key] * 100
                          for r in rows]))


def summarise(runs, label):
    n = len(runs)
    done = [r for r in runs if r["route_complete"]]
    s = {
        "policy": label, "n_runs": n, "n_completed": len(done),
        "completion_rate_pct": 100.0 * len(done) / max(1, n),
        "failed_sorties_mean": float(np.mean([r["failed_sorties"] for r in runs])),
        "battery_deaths_total": int(sum(r["battery_deaths"] for r in runs)),
    }
    if len(done) < MIN_COMPLETED_FOR_STATS:
        s.update({"stats_basis": "insufficient",
                  **{m + "_change_pct": None for m in METRICS}})
        return s
    s.update({
        "stats_basis": "completed_routes_only",
        "time_min_mean": float(np.mean([r["time_min"] for r in done])),
        "base_time_min_mean": float(np.mean([r["base_time_min"] for r in done])),
        "truck_km_mean": float(np.mean([r["truck_km"] for r in done])),
        "cost_inr_mean": float(np.mean([r["cost_inr"] for r in done])),
        "co2_kg_mean": float(np.mean([r["co2_kg"] for r in done])),
        "time_change_pct": _pct(done, "time_min"),
        "energy_change_pct": _pct(done, "energy_kwh"),
        "cost_change_pct": _pct(done, "cost_inr"),
        "co2_change_pct": _pct(done, "co2_kg"),
        "truck_km_change_pct": _pct(done, "truck_km"),
        "drone_share_pct": float(np.mean([100.0 * r["drone_deliveries"] / r["customers"]
                                          for r in done])),
        "truck_wait_min_mean": float(np.mean([r["truck_wait_min"] for r in done])),
        "battery_swaps_mean": float(np.mean([r["battery_swaps"] for r in done])),
        "min_battery_pct_mean": float(np.mean([r["min_battery_pct"] for r in done])),
    })
    return s


def across_seeds(per_seed, label):
    """
    Mean and 95% confidence interval across training seeds for every headline
    metric. With few seeds a t-interval is the honest choice: it widens to
    reflect how little five samples can tell you.
    """
    from scipy import stats

    out = {"policy": label, "n_seeds": len(per_seed),
           "completion_rate_pct": float(np.mean([s["completion_rate_pct"] for s in per_seed])),
           "failed_sorties_mean": float(np.mean([s["failed_sorties_mean"] for s in per_seed])),
           "battery_deaths_total": int(sum(s["battery_deaths_total"] for s in per_seed))}
    for key in [m + "_change_pct" for m in METRICS] + [
            "drone_share_pct", "truck_wait_min_mean", "time_min_mean",
            "truck_km_mean", "cost_inr_mean", "co2_kg_mean",
            "battery_swaps_mean", "min_battery_pct_mean", "base_time_min_mean"]:
        vals = [s.get(key) for s in per_seed if s.get(key) is not None]
        if not vals:
            out[key] = None
            continue
        mean = float(np.mean(vals))
        out[key] = mean
        if len(vals) > 1:
            half = float(stats.t.ppf(0.975, len(vals) - 1)
                         * np.std(vals, ddof=1) / np.sqrt(len(vals)))
            out[key + "_ci95"] = half
            out[key + "_seeds"] = [float(v) for v in vals]
    return out


def seed_average_runs(runs_by_seed):
    """One run per (instance, world): the mean over seeds of each metric."""
    keyed = {}
    for runs in runs_by_seed:
        for r in runs:
            keyed.setdefault((r["instance"], r["world_seed"]), []).append(r)
    out = []
    for (inst, ws), rs in sorted(keyed.items()):
        avg = dict(rs[0])
        for k in ("time_min", "energy_kwh", "cost_inr", "co2_kg", "truck_km"):
            avg[k] = float(np.mean([r[k] for r in rs]))
        avg["route_complete"] = all(r["route_complete"] for r in rs)
        out.append(avg)
    return out


def paired_tests(runs_by_policy, subject, opponents, metric="time_min"):
    """
    Paired t-test and Wilcoxon signed-rank on per-(instance, world)
    differences, with the win count -- a policy better on average but losing
    half its episodes is a much weaker claim than one that wins nearly all.
    """
    from scipy import stats

    def done(p):
        return {(r["instance"], r["world_seed"]): r[metric]
                for r in runs_by_policy[p] if r["route_complete"]}

    mine = done(subject)
    out = []
    for opp in opponents:
        theirs = done(opp)
        keys = sorted(set(mine) & set(theirs))
        if len(keys) < MIN_COMPLETED_FOR_STATS:
            continue
        a = np.array([mine[k] for k in keys])
        b = np.array([theirs[k] for k in keys])
        out.append({"subject": subject, "opponent": opp, "metric": metric,
                    "n": len(keys), "mean_diff": float((a - b).mean()),
                    "wins": int((a < b).sum()),
                    "paired_t_p": float(stats.ttest_rel(a, b).pvalue),
                    "wilcoxon_p": float(stats.wilcoxon(a, b).pvalue)})
    return out


def format_table(summaries):
    head = "{:<16} {:>6} {:>8} {:>8} {:>8} {:>8} {:>7} {:>6}".format(
        "policy", "compl", "time", "energy", "cost", "CO2", "drone%", "fails")
    lines = [head, "-" * len(head)]
    for s in summaries:
        if s.get("time_change_pct") is None:
            lines.append("{:<16} {:>5.0f}%  -- insufficient completions --".format(
                s["policy"], s["completion_rate_pct"]))
            continue
        ci = s.get("time_change_pct_ci95")
        lines.append("{:<16} {:>5.0f}% {:>+7.1f}% {:>+7.1f}% {:>+7.1f}% {:>+7.1f}% "
                     "{:>6.0f}% {:>6.1f}{}".format(
                         s["policy"], s["completion_rate_pct"], s["time_change_pct"],
                         s["energy_change_pct"], s["cost_change_pct"],
                         s["co2_change_pct"], s["drone_share_pct"],
                         s["failed_sorties_mean"],
                         "   (time +/-{:.1f}, {} seeds)".format(ci, s["n_seeds"])
                         if ci is not None else ""))
    return "\n".join(lines)


# ======================================================================
# Entry point
# ======================================================================

def learned_checkpoints(algo, seeds=None, which="best"):
    from .train import checkpoint
    seeds = config.SEEDS[algo] if seeds is None else seeds
    return [(s, checkpoint(algo, s, which)) for s in seeds
            if checkpoint(algo, s, which)]


def main():
    from .train import load_policy

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--algos", nargs="*", default=["maskable_ppo", "ppo", "dqn"])
    ap.add_argument("--which", default="best", choices=["best", "final"])
    ap.add_argument("--preference", default=config.HEADLINE_PREFERENCE)
    ap.add_argument("--out", default=os.path.join(RESULT_DIR, "evaluation.json"))
    args = ap.parse_args()

    scenario = load_scenario(config.CITY)
    _, eval_ids = split_instances(scenario)
    print("Evaluating on {} held-out instances x {} worlds, preference '{}'\n".format(
        len(eval_ids), len(config.EVAL_WORLD_SEEDS), args.preference))

    summaries, runs, per_seed_runs = [], {}, {}
    for name, cls in POLICIES.items():
        r = evaluate_policy(cls(), scenario, eval_ids, preference=args.preference)
        runs[name] = r
        summaries.append(summarise(r, name))
        print("  scored {}".format(name), flush=True)

    for algo in args.algos:
        ckpts = learned_checkpoints(algo, which=args.which)
        if not ckpts:
            print("  [skip] no trained {}".format(algo))
            continue
        seed_summaries, seed_runs = [], []
        for seed, path in ckpts:
            r = evaluate_policy(load_policy(algo, path), scenario, eval_ids,
                                is_sb3=True, masked=(algo == "maskable_ppo"),
                                preference=args.preference)
            seed_runs.append(r)
            seed_summaries.append(summarise(r, "{}#{}".format(algo, seed)))
            print("  scored {} seed {}".format(algo, seed), flush=True)
        per_seed_runs[algo] = {str(s): r for (s, _), r in zip(ckpts, seed_runs)}
        runs[algo] = seed_average_runs(seed_runs)
        agg = across_seeds(seed_summaries, algo)
        agg["per_seed"] = seed_summaries
        summaries.append(agg)

    ranked = [s["policy"] for s in sorted(
        (s for s in summaries if s.get("time_change_pct") is not None),
        key=lambda s: s["time_change_pct"])]
    tests = {}
    if len(ranked) > 1:
        for metric in ("time_min", "cost_inr", "energy_kwh"):
            tests[metric] = paired_tests(runs, ranked[0], ranked[1:], metric)

    print("\n" + format_table(summaries) + "\n")
    if tests.get("time_min"):
        print("Paired tests on delivery time for {} (n per test = {}):".format(
            ranked[0], tests["time_min"][0]["n"]))
        for t in tests["time_min"]:
            print("  vs {:<16} {:+6.1f} min  wins {:>3}/{:<3}  t p={:.1e}  W p={:.1e}".format(
                t["opponent"], t["mean_diff"], t["wins"], t["n"],
                t["paired_t_p"], t["wilcoxon_p"]))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"city": config.CITY, "n_eval_instances": len(eval_ids),
                   "world_seeds": list(config.EVAL_WORLD_SEEDS),
                   "n_customers": len(scenario["instances"][0]["customers"]),
                   "preference": args.preference, "summaries": summaries,
                   "significance": tests, "runs": runs,
                   "per_seed_runs": per_seed_runs}, f)
    print("\nResults -> {}".format(args.out))


if __name__ == "__main__":
    main()

"""
Evaluation.

Every policy is scored on the held-out instance half through one shared code
path, and each hybrid run is compared against the OR-Tools truck-only baseline
*for that same instance*, so the comparison is paired and instance difficulty
cancels out.

WHY THIS FILE IS CAREFUL ABOUT INCOMPLETE ROUTES
------------------------------------------------
An episode can end with customers unserved. Such a run finishes fast and burns
little energy, for the worst possible reason: the agent gave up. Averaging
those runs into a delivery-time comparison produces a large fake improvement --
an earlier version of this project reported a 15% time saving that was almost
entirely two runs which abandoned 13 of 15 packages.

So the report has two tiers:

    completion_rate     the fraction of runs that delivered every package.
                        A policy that scores badly here has failed, full stop.

    time / energy       computed over completed runs only, and stated with the
                        number of runs behind them.

If a policy completes too few routes to support a meaningful average, the
comparison is reported as ``None`` rather than filled in with whatever the
surviving runs happened to say.
"""

import argparse
import json
import os

import numpy as np

from . import config
from .baselines import POLICIES
from .env import TruckDroneEnv
from .scenario import load_scenario, split_instances

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")
RESULT_DIR = os.path.join(ROOT, "results")

MIN_COMPLETED_FOR_STATS = 5


# ======================================================================
# Rollout
# ======================================================================

def run_episode(env, policy, instance, is_sb3, masked, deterministic=True):
    """Run one instance under one policy and return its final info dict."""
    obs, _ = env.reset(options={"instance": instance})
    done = False
    steps = 0
    while not done:
        if is_sb3:
            if masked:
                action, _ = policy.predict(
                    obs, action_masks=env.action_masks(),
                    deterministic=deterministic)
            else:
                action, _ = policy.predict(obs, deterministic=deterministic)
            action = int(action)
        else:
            action = int(policy.predict(obs, env))
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    info = dict(info)
    info["steps"] = steps
    return info


def evaluate_policy(policy, scenario, instance_ids, is_sb3=False, masked=False,
                    env_kwargs=None):
    """Score one policy across every evaluation instance."""
    env = TruckDroneEnv(scenario, instance_ids=instance_ids,
                        **(env_kwargs or config.env_kwargs()))
    runs = []
    for i in range(len(instance_ids)):
        info = run_episode(env, policy, i, is_sb3, masked)
        baseline = info["truck_only"]
        runs.append({
            "instance": int(instance_ids[i]),
            "route_complete": bool(info["route_complete"]),
            "completion_pct": float(info["completion_pct"]),
            "served": int(info["served"]),
            "customers": int(info["total_customers"]),
            "hybrid_time_min": info["total_time_s"] / 60.0,
            "baseline_time_min": baseline["time_s"] / 60.0,
            "hybrid_energy_wh": info["total_energy_wh"],
            "baseline_energy_wh": baseline["energy_wh"],
            "truck_km": info["truck_distance_m"] / 1000.0,
            "baseline_km": baseline["distance_m"] / 1000.0,
            "drone_km": info["drone_distance_m"] / 1000.0,
            "drone_deliveries": int(info["drone_deliveries"]),
            "truck_deliveries": int(info["truck_deliveries"]),
            "failed_sorties": int(info["failed_sorties"]),
            "battery_deaths": int(info["battery_deaths"]),
            "battery_swaps": int(info["battery_swaps"]),
            "min_battery_pct": float(info["min_battery_pct"]),
            "truck_wait_min": info["truck_wait_time_s"] / 60.0,
            "steps": info["steps"],
        })
    env.close()
    return runs


# ======================================================================
# Summary
# ======================================================================

def _pct_change(new, old):
    """Signed percentage change; negative means the hybrid system did better."""
    return float(np.mean([(n - o) / o * 100.0 for n, o in zip(new, old)]))


def summarise(runs, label):
    """
    Reduce a set of runs to headline numbers.

    Completion is measured over everything. Time, energy and distance are
    measured over completed runs only, because a route that skipped customers
    is not comparable to one that served them all.
    """
    n = len(runs)
    completed = [r for r in runs if r["route_complete"]]
    summary = {
        "policy": label,
        "n_instances": n,
        "n_completed": len(completed),
        "completion_rate_pct": 100.0 * len(completed) / max(1, n),
        "mean_completion_pct": float(np.mean([r["completion_pct"] for r in runs])),
        "failed_sorties_mean": float(np.mean([r["failed_sorties"] for r in runs])),
        "battery_deaths_total": int(sum(r["battery_deaths"] for r in runs)),
    }

    if len(completed) < MIN_COMPLETED_FOR_STATS:
        summary.update({
            "stats_basis": "insufficient",
            "note": ("only {}/{} routes completed -- too few to quote a "
                     "delivery-time comparison".format(len(completed), n)),
            "time_change_pct": None,
            "energy_change_pct": None,
            "truck_km_change_pct": None,
        })
        return summary

    summary.update({
        "stats_basis": "completed_routes_only",
        "hybrid_time_min_mean": float(np.mean([r["hybrid_time_min"] for r in completed])),
        "hybrid_time_min_std": float(np.std([r["hybrid_time_min"] for r in completed])),
        "baseline_time_min_mean": float(np.mean([r["baseline_time_min"] for r in completed])),
        "time_change_pct": _pct_change(
            [r["hybrid_time_min"] for r in completed],
            [r["baseline_time_min"] for r in completed]),
        "hybrid_energy_wh_mean": float(np.mean([r["hybrid_energy_wh"] for r in completed])),
        "baseline_energy_wh_mean": float(np.mean([r["baseline_energy_wh"] for r in completed])),
        "energy_change_pct": _pct_change(
            [r["hybrid_energy_wh"] for r in completed],
            [r["baseline_energy_wh"] for r in completed]),
        "truck_km_mean": float(np.mean([r["truck_km"] for r in completed])),
        "baseline_km_mean": float(np.mean([r["baseline_km"] for r in completed])),
        "truck_km_change_pct": _pct_change(
            [r["truck_km"] for r in completed],
            [r["baseline_km"] for r in completed]),
        "drone_km_mean": float(np.mean([r["drone_km"] for r in completed])),
        "drone_deliveries_mean": float(np.mean([r["drone_deliveries"] for r in completed])),
        "drone_share_pct": float(np.mean(
            [100.0 * r["drone_deliveries"] / r["customers"] for r in completed])),
        "truck_wait_min_mean": float(np.mean([r["truck_wait_min"] for r in completed])),
        "min_battery_pct_mean": float(np.mean([r["min_battery_pct"] for r in completed])),
        "battery_swaps_mean": float(np.mean([r["battery_swaps"] for r in completed])),
    })
    return summary


def paired_tests(runs_by_policy, subject, opponents):
    """
    Is the difference real, or could 20 instances have produced it by chance?

    Every policy is run on the same instances, so the comparison is paired and
    instance difficulty cancels: for each instance we take the difference in
    delivery time and ask whether those differences are centred on zero. Both a
    paired t-test and a Wilcoxon signed-rank test are reported -- the t-test
    assumes roughly normal differences, Wilcoxon does not, and agreement
    between them means the answer does not rest on that assumption.

    The win count is included because it is the honest headline: a policy that
    is better on average but loses half its instances is a different claim from
    one that wins nearly all of them.
    """
    from scipy import stats

    def completed(policy):
        return {r["instance"]: r["hybrid_time_min"]
                for r in runs_by_policy[policy] if r["route_complete"]}

    mine = completed(subject)
    out = []
    for opponent in opponents:
        theirs = completed(opponent)
        shared = sorted(set(mine) & set(theirs))
        if len(shared) < MIN_COMPLETED_FOR_STATS:
            continue
        a = np.array([mine[i] for i in shared])
        b = np.array([theirs[i] for i in shared])
        out.append({
            "subject": subject,
            "opponent": opponent,
            "n_instances": len(shared),
            "mean_diff_min": float((a - b).mean()),
            "wins": int((a < b).sum()),
            "paired_t_p": float(stats.ttest_rel(a, b).pvalue),
            "wilcoxon_p": float(stats.wilcoxon(a, b).pvalue),
        })
    return out


def format_significance(tests):
    if not tests:
        return ""
    head = "{:<16} {:>10} {:>8} {:>12} {:>12}".format(
        "vs", "mean diff", "wins", "paired t", "wilcoxon")
    lines = ["", "Paired tests for {} (same instances, negative favours it)"
             .format(tests[0]["subject"]), head, "-" * len(head)]
    for t in tests:
        lines.append("{:<16} {:>+7.1f} min {:>5d}/{:<2d} {:>12.1e} {:>12.1e}".format(
            t["opponent"], t["mean_diff_min"], t["wins"], t["n_instances"],
            t["paired_t_p"], t["wilcoxon_p"]))
    return "\n".join(lines)


def format_table(summaries):
    """Render the comparison as a fixed-width table for the terminal."""
    head = ("{:<14} {:>7} {:>9} {:>9} {:>9} {:>8} {:>8}".format(
        "policy", "compl.", "time min", "vs base", "energy", "drone%", "fails"))
    lines = [head, "-" * len(head)]
    for s in summaries:
        if s["time_change_pct"] is None:
            lines.append("{:<14} {:>6.0f}% {:>9} {:>9} {:>9} {:>8} {:>8.1f}".format(
                s["policy"], s["completion_rate_pct"], "--", "--", "--", "--",
                s["failed_sorties_mean"]))
            continue
        lines.append(
            "{:<14} {:>6.0f}% {:>9.1f} {:>+8.1f}% {:>+8.1f}% {:>7.0f}% "
            "{:>8.1f}".format(
                s["policy"], s["completion_rate_pct"],
                s["hybrid_time_min_mean"], s["time_change_pct"],
                s["energy_change_pct"], s["drone_share_pct"],
                s["failed_sorties_mean"]))
    return "\n".join(lines)


# ======================================================================
# Entry point
# ======================================================================

def collect_policies(algos, model_dir=MODEL_DIR, which="best"):
    """
    Gather the heuristic policies plus whichever trained models exist on disk.

    Missing models are skipped with a warning rather than raising, so the
    evaluation is still useful before every algorithm has finished training.
    """
    from .train import load_policy

    entries = []
    for name, cls in POLICIES.items():
        entries.append({"label": name, "policy": cls(), "is_sb3": False,
                        "masked": False})

    for algo in algos:
        fname = "best/best_model.zip" if which == "best" else "final.zip"
        path = os.path.join(model_dir, algo, fname)
        if not os.path.exists(path):
            path = os.path.join(model_dir, algo, "final.zip")
        if not os.path.exists(path):
            print("[skip] no trained model for {} at {}".format(algo, path))
            continue
        entries.append({"label": algo, "policy": load_policy(algo, path),
                        "is_sb3": True, "masked": algo == "maskable_ppo"})
    return entries


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--algos", nargs="*", default=["maskable_ppo", "ppo", "dqn"])
    ap.add_argument("--which", default="best", choices=["best", "final"])
    ap.add_argument("--out", default=os.path.join(RESULT_DIR, "evaluation.json"))
    args = ap.parse_args()

    scenario = load_scenario()
    _, eval_ids = split_instances(scenario)
    print("Evaluating on {} held-out instances "
          "({} customers each)\n".format(len(eval_ids),
                                         len(scenario["instances"][0]["customers"])))

    summaries, detail = [], {}
    for entry in collect_policies(args.algos, which=args.which):
        runs = evaluate_policy(entry["policy"], scenario, eval_ids,
                               is_sb3=entry["is_sb3"], masked=entry["masked"])
        summaries.append(summarise(runs, entry["label"]))
        detail[entry["label"]] = runs
        print("  scored {}".format(entry["label"]))

    print("\n" + format_table(summaries) + "\n")

    # Whether the best policy's lead is real, tested against every other
    # policy on the same instances.
    ranked = [s["policy"] for s in sorted(
        (s for s in summaries if s["time_change_pct"] is not None),
        key=lambda s: s["time_change_pct"])]
    tests = paired_tests(detail, ranked[0], ranked[1:]) if len(ranked) > 1 else []
    if tests:
        print(format_significance(tests) + "\n")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"n_eval_instances": len(eval_ids),
                   "n_customers": len(scenario["instances"][0]["customers"]),
                   "summaries": summaries, "significance": tests,
                   "runs": detail}, f, indent=2)
    print("Results -> {}".format(args.out))


if __name__ == "__main__":
    main()

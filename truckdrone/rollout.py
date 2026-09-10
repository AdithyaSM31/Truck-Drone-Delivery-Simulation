"""
Turn a policy rollout into an animation trace.

The environment reasons in discrete decisions -- "advance", "launch to
customer 3" -- and records what each decision cost. A viewer needs something
else: continuous positions over a clock. This module bridges the two.

Each ``advance`` decision becomes a *leg*: the truck drives from one stop to
the next along the actual node-by-node shortest path through the road graph,
while any drone launched at the previous stop flies out to its customer,
hovers to drop the parcel, then flies ahead to the recovery point. The leg
lasts as long as the slower of the two, which is precisely the concurrency the
whole idea depends on.

The trace carries geometry (where things are), timing (when), and running
metrics (battery, energy, packages delivered), so the dashboard can scrub to
any moment and show the true state. The same instance is also rolled out under
the truck-only baseline so the two can be animated side by side on one clock.
"""

import argparse
import json
import os

import numpy as np

from . import config
from .baselines import POLICIES, TruckOnlyPolicy
from .env import TruckDroneEnv
from .scenario import build_graph, load_scenario, road_path, split_instances

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(ROOT, "results")

VIEW_SIZE = 1000.0


# ======================================================================
# Geometry
# ======================================================================

def view_coords(coords, margin=0.06):
    """
    Map UTM metres onto a square view box, preserving aspect ratio.

    The frontend then treats these as plain pixels, so no projection logic has
    to live in JavaScript. Y is flipped because screen coordinates grow
    downwards while northings grow upwards.
    """
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    span = float(np.max(hi - lo))
    scale = VIEW_SIZE * (1.0 - 2.0 * margin) / span
    centred = (coords - (lo + hi) / 2.0) * scale
    out = centred + VIEW_SIZE / 2.0
    out[:, 1] = VIEW_SIZE - out[:, 1]
    return out


# ======================================================================
# Trace construction
# ======================================================================

def _leg_geometry(graph, trace_legs):
    """Attach the node-by-node road path to every leg."""
    for leg in trace_legs:
        leg["truck_path"] = road_path(graph, leg["truck_from"], leg["truck_to"])
    return trace_legs


def _run(scenario, instance_ids, instance, policy, is_sb3, masked, graph,
         env_kwargs=None):
    """Roll one policy over one instance and return (legs, final info)."""
    env = TruckDroneEnv(scenario, instance_ids=instance_ids,
                        record_trace=True,
                        **(env_kwargs or config.env_kwargs()))
    obs, _ = env.reset(options={"instance": instance})
    done = False
    while not done:
        if is_sb3:
            if masked:
                action, _ = policy.predict(obs, action_masks=env.action_masks(),
                                           deterministic=True)
            else:
                action, _ = policy.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = int(policy.predict(obs, env))
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    legs = _leg_geometry(graph, env.trace)
    route = [int(n) for n in env.truck_route]
    customers = [int(c) for c in env.customer_indices]
    depot = int(env.depot_index)
    env.close()
    return legs, dict(info), route, customers, depot


def build_trace(scenario, graph, instance_ids, instance, policy, label,
                is_sb3=False, masked=False, env_kwargs=None):
    """
    Produce the full animation payload: the hybrid rollout, the truck-only
    baseline on the same instance, and everything needed to draw the map.
    """
    legs, info, route, customers, depot = _run(
        scenario, instance_ids, instance, policy, is_sb3, masked, graph,
        env_kwargs)
    base_legs, base_info, base_route, _, _ = _run(
        scenario, instance_ids, instance, TruckOnlyPolicy(), False, False,
        graph, env_kwargs)

    coords = np.asarray(scenario["coords"], dtype=np.float64)
    xy = view_coords(coords)

    nodes = [{"id": i, "x": round(float(xy[i, 0]), 2),
              "y": round(float(xy[i, 1]), 2)} for i in range(len(coords))]
    edges = [[int(u), int(v)] for u, v in graph.edges()]

    def pack(legs_):
        out = []
        for leg in legs_:
            out.append({
                "t_start": round(leg["t_start_s"], 2),
                "t_end": round(leg["t_end_s"], 2),
                "truck_time_s": round(leg["truck_time_s"], 2),
                "truck_path": leg["truck_path"],
                "truck_km": round(leg["truck_distance_m"] / 1000.0, 3),
                "served_after": leg["served_count"],
                "packs": leg["packs_after"],
                "truck_delivery": leg["truck_served_node"],
                "sorties": [{
                    "drone": s["drone_idx"],
                    "launch": s["launch_node"],
                    "customer": s["customer_node"],
                    "recovery": s["recovery_node"],
                    "time_s": round(s["time_s"], 2),
                    "km": round(s["distance_m"] / 1000.0, 3),
                    "wh": round(s["energy_wh"], 2),
                    "battery_before": round(s["battery_before"], 1),
                    "battery_after": round(s["battery_after"], 1),
                    "swapped": bool(s["swapped"]),
                    "detour_saved_km": round(s["road_detour_saved_m"] / 1000.0, 3),
                } for s in leg["sorties"]],
            })
        return out

    def stats(inf):
        return {
            "time_min": round(inf["total_time_s"] / 60.0, 1),
            "truck_km": round(inf["truck_distance_m"] / 1000.0, 2),
            "drone_km": round(inf["drone_distance_m"] / 1000.0, 2),
            "energy_wh": round(inf["total_energy_wh"], 1),
            "truck_energy_wh": round(inf["truck_energy_wh"], 1),
            "drone_energy_wh": round(inf["drone_energy_wh"], 2),
            "served": inf["served"],
            "customers": inf["total_customers"],
            "route_complete": bool(inf["route_complete"]),
            "drone_deliveries": inf["drone_deliveries"],
            "truck_deliveries": inf["truck_deliveries"],
            "failed_sorties": inf["failed_sorties"],
            "min_battery_pct": round(inf["min_battery_pct"], 1),
            "battery_swaps": inf["battery_swaps"],
            "truck_wait_min": round(inf["truck_wait_time_s"] / 60.0, 1),
        }

    hybrid_stats = stats(info)
    baseline_stats = stats(base_info)
    baseline_stats["time_min"] = round(info["truck_only"]["time_s"] / 60.0, 1)
    baseline_stats["truck_km"] = round(info["truck_only"]["distance_m"] / 1000.0, 2)
    baseline_stats["energy_wh"] = round(info["truck_only"]["energy_wh"], 1)

    return {
        "policy": label,
        "instance": int(instance_ids[instance % len(instance_ids)]),
        "nodes": nodes,
        "edges": edges,
        "depot": depot,
        "customers": customers,
        "truck_route": route,
        "baseline_route": base_route,
        "hybrid": {"legs": pack(legs), "stats": hybrid_stats},
        "baseline": {"legs": pack(base_legs), "stats": baseline_stats},
        "comparison": {
            "time_change_pct": round(
                (hybrid_stats["time_min"] - baseline_stats["time_min"])
                / baseline_stats["time_min"] * 100.0, 1),
            "truck_km_change_pct": round(
                (hybrid_stats["truck_km"] - baseline_stats["truck_km"])
                / baseline_stats["truck_km"] * 100.0, 1),
            "energy_change_pct": round(
                (hybrid_stats["energy_wh"] - baseline_stats["energy_wh"])
                / baseline_stats["energy_wh"] * 100.0, 1),
        },
    }


def make_rollout(scenario, graph, label, instance=0, model_dir=None,
                 which="best"):
    """Build a trace for a heuristic policy name or a trained algorithm name."""
    _, eval_ids = split_instances(scenario)

    if label in POLICIES:
        return build_trace(scenario, graph, eval_ids, instance,
                           POLICIES[label](), label)

    from .train import load_policy

    model_dir = model_dir or os.path.join(ROOT, "models")
    fname = "best/best_model.zip" if which == "best" else "final.zip"
    path = os.path.join(model_dir, label, fname)
    if not os.path.exists(path):
        path = os.path.join(model_dir, label, "final.zip")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "No trained model for {!r}. Train one first:\n"
            "    python -m truckdrone.train --algo {}".format(label, label))
    policy = load_policy(label, path)
    return build_trace(scenario, graph, eval_ids, instance, policy, label,
                       is_sb3=True, masked=(label == "maskable_ppo"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", default="greedy")
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(RESULT_DIR, "rollout.json"))
    args = ap.parse_args()

    scenario = load_scenario()
    graph = build_graph(scenario["coords"])
    trace = make_rollout(scenario, graph, args.policy, instance=args.instance)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(trace, f)

    h, b = trace["hybrid"]["stats"], trace["baseline"]["stats"]
    print("{} on instance {}".format(args.policy, trace["instance"]))
    print("  hybrid   {:.1f} min | {:.1f} truck km | {}/{} served | "
          "{} by drone".format(h["time_min"], h["truck_km"], h["served"],
                               h["customers"], h["drone_deliveries"]))
    print("  baseline {:.1f} min | {:.1f} truck km".format(
        b["time_min"], b["truck_km"]))
    print("  -> {:+.1f}% time, {:+.1f}% truck km".format(
        trace["comparison"]["time_change_pct"],
        trace["comparison"]["truck_km_change_pct"]))
    print("Trace -> {}".format(args.out))


if __name__ == "__main__":
    main()

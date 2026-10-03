"""
Turn a policy rollout into an animation trace.

The environment reasons in decisions; a viewer needs positions on a clock.
Two kinds of record come out of an episode:

  * truck legs -- one per "drive on" decision, each carrying the real street
    polyline between its two stops, its departure, arrival and finish times,
    the doorstep service and any wait for a drone;
  * sorties -- one per launch, with absolute launch, landing and recovery
    times. A sortie can span several truck legs (the agent chose to meet the
    truck further down the route), so sorties are a separate timeline rather
    than something hung off a single leg.

Coordinates are UTM metres mapped into a 1000 x 1000 view box. The mapping is
fixed per city (from the extent of its road network), so every trace for a
city shares one frame and the background map is sent only once.
"""

import argparse
import json
import os

import numpy as np

from . import config
from .baselines import POLICIES, TruckOnlyPolicy
from .env import PREFERENCE_PRESETS, TruckDroneEnv
from .evaluate import act
from .scenario import CITIES, StreetPaths, load_scenario, split_instances

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(ROOT, "results")
VIEW = 1000.0
_FRAMES = {}


# ======================================================================
# View frame
# ======================================================================

class Frame:
    """UTM metres -> view pixels, preserving aspect ratio, y flipped."""

    def __init__(self, city, margin=0.03):
        self.paths = StreetPaths(city)
        bg = self.paths.background_xy()
        pts = np.concatenate(bg) if bg else load_scenario(city)["coords"]
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        self.centre = (lo + hi) / 2.0
        self.scale = VIEW * (1 - 2 * margin) / float(np.max(hi - lo))
        self.metres_per_px = 1.0 / self.scale
        self._background = bg

    def xy(self, pts):
        p = (np.asarray(pts, dtype=np.float64) - self.centre) * self.scale + VIEW / 2
        p[..., 1] = VIEW - p[..., 1]
        return p

    def simplify(self, pts, min_px=1.5):
        """Drop points closer than ``min_px`` to the last kept one."""
        p = self.xy(pts)
        keep = [p[0]]
        for q in p[1:-1]:
            if np.linalg.norm(q - keep[-1]) >= min_px:
                keep.append(q)
        keep.append(p[-1])
        return [[round(float(x), 1), round(float(y), 1)] for x, y in keep]

    def background(self):
        return [self.simplify(line, 2.5) for line in self._background]


def frame(city):
    if city not in _FRAMES:
        _FRAMES[city] = Frame(city)
    return _FRAMES[city]


def map_payload(city):
    """Everything static about a city's map: roads, sites, scale."""
    sc = load_scenario(city)
    fr = frame(city)
    sites = fr.xy(sc["coords"])
    return {"city": city, "name": CITIES[city]["name"],
            "metres_per_px": fr.metres_per_px,
            "roads": fr.background(),
            "sites": [[round(float(x), 1), round(float(y), 1)] for x, y in sites],
            "n_instances": len(split_instances(sc)[1])}


# ======================================================================
# Rollout
# ======================================================================

def _run(env, policy, options, is_sb3, masked):
    obs, _ = env.reset(options=options)
    done = False
    while not done:
        obs, _, te, tr, info = env.step(act(policy, obs, env, is_sb3, masked))
        done = te or tr
    return env.trace, dict(info)


def _pack(trace, info, fr, city):
    paths = fr.paths
    legs = [{
        "from": int(l["from_node"]), "to": int(l["to_node"]),
        "t_start": round(l["t_start"], 1), "t_arrive": round(l["t_arrive"], 1),
        "t_end": round(l["t_end"], 1),
        "path": (fr.simplify(paths.xy(l["from_node"], l["to_node"]))
                 if l["from_node"] != l["to_node"] else []),
        "km": round(l["distance_m"] / 1000.0, 3),
        "service_s": round(l["service_s"], 1), "idle_s": round(l["idle_s"], 1),
        "truck_delivery": l["truck_served_node"],
        "packs": l["packs_after"], "congestion": round(l["congestion"], 3),
    } for l in trace["legs"]]
    sorties = [{
        "drone": s["drone_idx"], "launch": s["launch_node"],
        "customer": s["customer_node"], "recovery": s["recovery_node"],
        "option": s["rendezvous_option"],
        "t_launch": round(s["t_launch"], 1), "t_land": round(s["t_land"], 1),
        "t_recover": round(s["t_recover"], 1),
        "km": round(s["distance_m"] / 1000.0, 3), "wh": round(s["energy_wh"], 2),
        "battery_before": round(s["battery_before"], 1),
        "battery_after": round(s["battery_after"], 1),
        "swapped": bool(s["swapped"]),
        "detour_saved_km": round(s["detour_saved_m"] / 1000.0, 3),
    } for s in trace["sorties"]]
    stats = {
        "time_min": round(info["total_time_s"] / 60.0, 1),
        "truck_km": round(info["truck_distance_m"] / 1000.0, 2),
        "drone_km": round(info["drone_distance_m"] / 1000.0, 2),
        "energy_kwh": round(info["total_energy_wh"] / 1000.0, 2),
        "cost_inr": round(info["cost_inr"], 0), "co2_kg": round(info["co2_kg"], 2),
        "served": info["served"], "customers": info["total_customers"],
        "route_complete": bool(info["route_complete"]),
        "drone_deliveries": info["drone_deliveries"],
        "truck_deliveries": info["truck_deliveries"],
        "battery_swaps": info["battery_swaps"],
        "truck_wait_min": round(info["truck_wait_time_s"] / 60.0, 1),
        "min_battery_pct": round(info["min_battery_pct"], 1),
    }
    return {"legs": legs, "sorties": sorties, "stats": stats}


def build_trace(scenario, instance, world_seed, policy, label, preference,
                is_sb3=False, masked=False, env_kwargs=None, include_baseline=True):
    """Hybrid rollout, plus (optionally) the truck-only run in the same world."""
    city = scenario["city"]
    fr = frame(city)
    _, eval_ids = split_instances(scenario)
    env = TruckDroneEnv(scenario, instance_ids=eval_ids, record_trace=True,
                        **(env_kwargs or config.env_kwargs()))
    pref = PREFERENCE_PRESETS.get(preference, preference)
    opts = {"instance": instance, "world_seed": world_seed, "preference": pref}
    trace, info = _run(env, policy, opts, is_sb3, masked)
    out = {
        "policy": label, "city": city, "instance": instance,
        "instance_global_id": int(eval_ids[instance]), "world_seed": world_seed,
        "preference": preference if isinstance(preference, str) else list(pref),
        "world": {"wind_ms": round(env.wind_speed, 2),
                  "wind_to_deg": round(float(np.degrees(np.arctan2(env.wind[1], env.wind[0]))), 1),
                  "start_hour": round(env.start_hour, 2)},
        "depot": int(env.depot_index),
        "customers": [int(c) for c in env.customer_indices],
        "hybrid": _pack(trace, info, fr, city),
    }
    b = info["truck_only"]
    out["baseline_stats"] = {"time_min": round(b["time_s"] / 60, 1),
                             "truck_km": round(b["distance_m"] / 1000, 2),
                             "energy_kwh": round(b["energy_wh"] / 1000, 2),
                             "cost_inr": round(b["cost_inr"], 0),
                             "co2_kg": round(b["co2_kg"], 2)}
    h = out["hybrid"]["stats"]
    out["comparison"] = {
        k + "_change_pct": round((h[k2] - out["baseline_stats"][k2])
                                 / out["baseline_stats"][k2] * 100, 1)
        for k, k2 in (("time", "time_min"), ("truck_km", "truck_km"),
                      ("energy", "energy_kwh"), ("cost", "cost_inr"), ("co2", "co2_kg"))}
    if include_baseline:
        out["baseline"] = baseline_trace(scenario, instance, world_seed, env_kwargs)
    env.close()
    return out


def baseline_trace(scenario, instance, world_seed, env_kwargs=None):
    city = scenario["city"]
    _, eval_ids = split_instances(scenario)
    env = TruckDroneEnv(scenario, instance_ids=eval_ids, record_trace=True,
                        **(env_kwargs or config.env_kwargs()))
    trace, info = _run(env, TruckOnlyPolicy(),
                       {"instance": instance, "world_seed": world_seed}, False, False)
    env.close()
    return _pack(trace, info, frame(city), city)


def make_rollout(scenario, label, instance=0, world_seed=None, preference=None,
                 model_path=None, include_baseline=True):
    """A trace for a heuristic name or a trained algorithm name."""
    world_seed = config.EVAL_WORLD_SEEDS[0] if world_seed is None else world_seed
    preference = preference or config.HEADLINE_PREFERENCE
    if label in POLICIES:
        return build_trace(scenario, instance, world_seed, POLICIES[label](), label,
                           preference, include_baseline=include_baseline)
    from .train import load_policy
    if model_path is None:
        raise FileNotFoundError("No model path for {!r}".format(label))
    return build_trace(scenario, instance, world_seed, load_policy(label, model_path),
                       label, preference, is_sb3=True,
                       masked=(label == "maskable_ppo"),
                       include_baseline=include_baseline)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", default="greedy")
    ap.add_argument("--city", default=config.CITY)
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--world", type=int, default=config.EVAL_WORLD_SEEDS[0])
    ap.add_argument("--out", default=os.path.join(RESULT_DIR, "rollout.json"))
    args = ap.parse_args()
    trace = make_rollout(load_scenario(args.city), args.policy, args.instance, args.world)
    with open(args.out, "w") as f:
        json.dump(trace, f)
    print(json.dumps({k: trace[k] for k in ("policy", "city", "world", "comparison")}, indent=1))


if __name__ == "__main__":
    main()

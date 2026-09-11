"""
Freeze the dashboard into a static site.

The live dashboard runs policies on demand, which means shipping PyTorch --
544 MB of it, before OR-Tools and SciPy. That is more than twice Vercel's
250 MB serverless budget, and on a container host it buys a cold start long
enough to ruin a demo.

None of it is necessary. Every rollout the dashboard can display is
deterministic: the learned policies are evaluated with ``deterministic=True``,
the heuristics have no randomness, and ``RandomPolicy`` is seeded. So the set
of things a visitor can ask for is finite and known in advance -- every policy
crossed with every held-out instance -- and each one is about 9 KB of JSON.

This script renders all of them once, next to a copy of the dashboard that
reads files instead of calling an API. The result is a few hundred kilobytes
of static assets that load instantly, cost nothing to host, and cannot fall
over during a presentation.

    python -m truckdrone.export_static
    python -m truckdrone.export_static --out site --policies maskable_ppo greedy
"""

import argparse
import json
import os
import shutil
import time

from . import config
from .baselines import POLICIES
from .rollout import make_rollout
from .scenario import build_graph, load_scenario, split_instances
from .train import ALGOS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")
RESULT_DIR = os.path.join(ROOT, "results")
DASHBOARD = os.path.join(ROOT, "dashboard", "static", "index.html")
DEFAULT_OUT = os.path.join(ROOT, "site")

# Injected into the copied dashboard. The page checks for this flag and reads
# from data/ instead of /api/, so one HTML file serves both the Flask dev
# server and the frozen site -- no second copy to keep in sync.
STATIC_FLAG = "<script>window.__STATIC__ = true;</script>\n"

VERCEL_CONFIG = {
    "cleanUrls": True,
    "headers": [
        {
            # Traces are content-addressed by policy and instance and only
            # change when the agents are retrained, so let the browser keep
            # them. The page itself must not be cached, or a redeploy would
            # leave visitors on the old dashboard.
            "source": "/data/(.*)",
            "headers": [{"key": "Cache-Control",
                         "value": "public, max-age=604800, immutable"}],
        },
        {
            "source": "/index.html",
            "headers": [{"key": "Cache-Control", "value": "no-cache"}],
        },
    ],
}


def _trained_policies(model_dir):
    return [a for a in ALGOS
            if os.path.exists(os.path.join(model_dir, a, "best", "best_model.zip"))
            or os.path.exists(os.path.join(model_dir, a, "final.zip"))]


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    return os.path.getsize(path)


def export(out_dir=DEFAULT_OUT, policies=None, model_dir=MODEL_DIR):
    scenario = load_scenario()
    graph = build_graph(scenario["coords"])
    _, eval_ids = split_instances(scenario)

    names = policies or (_trained_policies(model_dir) + list(POLICIES))
    if not names:
        raise RuntimeError("No policies to export -- train an agent first.")

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(os.path.join(out_dir, "data", "traces"), exist_ok=True)

    total = 0
    total += _write_json(os.path.join(out_dir, "data", "info.json"), {
        "heuristics": list(POLICIES),
        "trained": _trained_policies(model_dir),
        "n_instances": len(eval_ids),
        "n_customers": len(scenario["instances"][0]["customers"]),
        "n_nodes": scenario["n_nodes"],
        "n_edges": scenario["n_edges"],
        "region": "Whitefield, Bengaluru",
        "area_km": scenario["area_km"],
    })

    # The evaluation table, if it has been generated. Optional: the dashboard
    # simply omits the panel when it is missing.
    eval_path = os.path.join(RESULT_DIR, "evaluation.json")
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            ev = json.load(f)
        total += _write_json(os.path.join(out_dir, "data", "results.json"), {
            "n_eval_instances": ev["n_eval_instances"],
            "n_customers": ev["n_customers"],
            "summaries": ev["summaries"],
        })

    print("Rendering {} policies x {} instances ...".format(
        len(names), len(eval_ids)))
    t0 = time.perf_counter()
    done = 0
    for policy in names:
        for instance in range(len(eval_ids)):
            trace = make_rollout(scenario, graph, policy, instance=instance,
                                 model_dir=model_dir)
            total += _write_json(
                os.path.join(out_dir, "data", "traces",
                             "{}-{}.json".format(policy, instance)), trace)
            done += 1
        print("  {:<16} {} instances".format(policy, len(eval_ids)))

    # One dashboard, two modes: copy it verbatim and prepend the static flag.
    with open(DASHBOARD, encoding="utf-8") as f:
        html = f.read()
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(STATIC_FLAG + html)

    _write_json(os.path.join(out_dir, "vercel.json"), VERCEL_CONFIG)

    elapsed = time.perf_counter() - t0
    print("\n{} traces in {:.0f}s".format(done, elapsed))
    print("Site: {}".format(out_dir))
    print("Total payload: {:.2f} MB across {} files".format(
        total / 1048576, done + 2))
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--policies", nargs="*", default=None,
                    help="defaults to every trained agent plus every heuristic")
    args = ap.parse_args()
    export(out_dir=args.out, policies=args.policies)


if __name__ == "__main__":
    main()

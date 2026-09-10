"""
Dashboard server.

A small Flask app that runs policies on demand and hands the browser an
animation trace. The heavy objects -- the scenario, the road graph, the loaded
policy networks -- are built once and kept in memory, so switching policy or
instance in the UI costs a rollout and nothing more.

    python dashboard/server.py
    -> http://127.0.0.1:5000
"""

import json
import os
import sys

from flask import Flask, jsonify, request, send_from_directory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from truckdrone.baselines import POLICIES          # noqa: E402
from truckdrone.rollout import make_rollout        # noqa: E402
from truckdrone.scenario import (build_graph, load_scenario,  # noqa: E402
                                 split_instances)
from truckdrone.train import ALGOS                 # noqa: E402

STATIC_DIR = os.path.join(ROOT, "dashboard", "static")
MODEL_DIR = os.path.join(ROOT, "models")
RESULT_DIR = os.path.join(ROOT, "results")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")

_state = {}
_cache = {}


def state():
    """Load the scenario and road graph once, on first request."""
    if not _state:
        print("[server] loading scenario ...")
        scenario = load_scenario()
        _state["scenario"] = scenario
        _state["graph"] = build_graph(scenario["coords"])
        _state["eval_ids"] = split_instances(scenario)[1]
        print("[server] ready: {} nodes, {} held-out instances".format(
            scenario["n_nodes"], len(_state["eval_ids"])))
    return _state


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/info")
def info():
    """Which policies can be run, and how many instances are available."""
    s = state()
    trained = [a for a in ALGOS
               if os.path.exists(os.path.join(MODEL_DIR, a, "best", "best_model.zip"))
               or os.path.exists(os.path.join(MODEL_DIR, a, "final.zip"))]
    return jsonify({
        "heuristics": list(POLICIES.keys()),
        "trained": trained,
        "n_instances": len(s["eval_ids"]),
        "n_customers": len(s["scenario"]["instances"][0]["customers"]),
        "n_nodes": s["scenario"]["n_nodes"],
        "n_edges": s["scenario"]["n_edges"],
        "region": "Whitefield, Bengaluru",
        "area_km": s["scenario"]["area_km"],
    })


@app.route("/api/rollout")
def rollout():
    policy = request.args.get("policy", "greedy")
    instance = int(request.args.get("instance", 0))
    key = (policy, instance)

    if key not in _cache:
        s = state()
        try:
            _cache[key] = make_rollout(s["scenario"], s["graph"], policy,
                                       instance=instance, model_dir=MODEL_DIR)
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404
    return jsonify(_cache[key])


@app.route("/api/results")
def results():
    """The saved evaluation table, if `python -m truckdrone.evaluate` has run."""
    path = os.path.join(RESULT_DIR, "evaluation.json")
    if not os.path.exists(path):
        return jsonify({"error": "No evaluation.json yet. Run: "
                                 "python -m truckdrone.evaluate"}), 404
    with open(path) as f:
        data = json.load(f)
    return jsonify({"n_eval_instances": data["n_eval_instances"],
                    "n_customers": data["n_customers"],
                    "summaries": data["summaries"]})


def port_is_taken(host="127.0.0.1", port=5000):
    """
    True if something is already serving on this port.

    Worth checking explicitly. Windows lets a second process bind a port that
    is already in use rather than refusing, so starting the dashboard twice
    leaves two servers up and the browser talking to whichever answers first --
    which during a live demo looks like the page ignoring your changes.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.4)
        return probe.connect_ex((host, port)) == 0


if __name__ == "__main__":
    HOST, PORT = "127.0.0.1", 5000
    if port_is_taken(HOST, PORT):
        raise SystemExit(
            "\n  A server is already running on http://{}:{}\n"
            "  Open that one, or stop it first (Ctrl-C in its terminal).\n"
            "  Starting a second here would serve stale content.\n"
            .format(HOST, PORT))

    state()
    print("\n  Dashboard -> http://{}:{}\n".format(HOST, PORT))
    app.run(host=HOST, port=PORT, debug=False)

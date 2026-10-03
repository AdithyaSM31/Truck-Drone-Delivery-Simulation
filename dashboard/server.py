"""
Dashboard server.

A small Flask app that runs policies on demand and hands the browser an
animation trace. Scenarios, street geometry and loaded policy networks are
built once and kept in memory, so switching city, policy, preference,
instance or world costs one rollout and nothing more.

    python dashboard/server.py
    -> http://127.0.0.1:5000
"""

import os
import sys

from flask import Flask, abort, jsonify, request, send_from_directory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from truckdrone import catalog, config               # noqa: E402
from truckdrone.baselines import POLICIES            # noqa: E402
from truckdrone.rollout import (baseline_trace, build_trace,  # noqa: E402
                                map_payload)
from truckdrone.scenario import load_scenario        # noqa: E402

STATIC_DIR = os.path.join(ROOT, "dashboard", "static")
MAP_DIR = os.path.join(ROOT, "results", "maps")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")

_scenarios, _policies, _cache = {}, {}, {}
_info = {}


def scenario(city):
    if city not in (config.CITY, config.TRANSFER_CITY):
        abort(404)
    if city not in _scenarios:
        _scenarios[city] = load_scenario(city)
    return _scenarios[city]


def policy(name):
    """(policy object, is_sb3, masked) -- networks loaded once."""
    if name in POLICIES:
        return POLICIES[name](), False, False
    if name not in _policies:
        from truckdrone.train import load_policy
        models = catalog.learned_models()
        if name not in models:
            abort(404)
        _policies[name] = load_policy(name, models[name][1])
    return _policies[name], True, name == "maskable_ppo"


def cached(key, make):
    if key not in _cache:
        _cache[key] = make()
    return _cache[key]


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/info")
def info():
    if not _info:
        _info.update(catalog.info())
    return jsonify(_info)


@app.route("/api/map")
def city_map():
    city = request.args.get("city", config.CITY)
    scenario(city)
    return jsonify(cached(("map", city), lambda: map_payload(city)))


@app.route("/api/rollout")
def rollout():
    city = request.args.get("city", config.CITY)
    name = request.args.get("policy", "greedy")
    pref = request.args.get("preference", config.HEADLINE_PREFERENCE)
    if pref not in catalog.preferences_for(name):
        pref = config.HEADLINE_PREFERENCE
    instance = int(request.args.get("instance", 0))
    world = int(request.args.get("world", config.EVAL_WORLD_SEEDS[0]))

    def make():
        pol, is_sb3, masked = policy(name)
        return build_trace(scenario(city), instance, world, pol, name, pref,
                           is_sb3=is_sb3, masked=masked, include_baseline=False)
    return jsonify(cached(("roll", city, name, pref, instance, world), make))


@app.route("/api/baseline")
def baseline():
    city = request.args.get("city", config.CITY)
    instance = int(request.args.get("instance", 0))
    world = int(request.args.get("world", config.EVAL_WORLD_SEEDS[0]))
    return jsonify(cached(("base", city, instance, world),
                          lambda: baseline_trace(scenario(city), instance, world)))


@app.route("/api/results")
def results():
    data = catalog.results_payload()
    if data is None:
        return jsonify({"error": "No evaluation.json yet. Run: "
                                 "python -m truckdrone.evaluate"}), 404
    return jsonify(data)


@app.route("/maps/<path:name>")
def real_map(name):
    return send_from_directory(MAP_DIR, name)


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

    scenario(config.CITY)
    print("\n  Dashboard -> http://{}:{}\n".format(HOST, PORT))
    app.run(host=HOST, port=PORT, debug=False)

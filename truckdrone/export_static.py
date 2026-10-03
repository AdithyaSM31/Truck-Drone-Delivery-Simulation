"""
Freeze the dashboard into a static site.

The live dashboard runs policies on demand, which means shipping PyTorch --
544 MB of it, before OR-Tools and SciPy. That is more than twice Vercel's
250 MB serverless budget, and on a container host it buys a cold start long
enough to ruin a demo.

None of it is necessary. Every rollout the dashboard can display is
deterministic: the world is fixed by its seed, learned policies act with
``deterministic=True``, and the heuristics are seeded. So the set of things a
visitor can ask for is finite and known in advance (see ``catalog.py``), and
this script renders all of them once, next to a copy of the dashboard that
reads files instead of calling an API.

    data/info.json                         what can be selected
    data/results.json                      evaluation, analysis, ablations
    data/map-<city>.json                   streets and sites, once per city
    data/traces/<city>/<policy>-<pref>-<instance>-<world>.json
    data/baseline/<city>/<instance>-<world>.json
    maps/<city>-<instance>.html            Folium maps on real streets

    python -m truckdrone.export_static
"""

import argparse
import json
import os
import shutil
import time

from . import catalog, config
from .baselines import POLICIES
from .realmap import build_map
from .rollout import baseline_trace, build_trace, map_payload
from .scenario import load_scenario

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASHBOARD = os.path.join(ROOT, "dashboard", "static", "index.html")
DEFAULT_OUT = os.path.join(ROOT, "site")

# Injected into the copied dashboard. The page checks for this flag and reads
# from data/ instead of /api/, so one HTML file serves both the Flask dev
# server and the frozen site -- no second copy to keep in sync.
#
# It goes inside <head>, not above the file. Anything before <!doctype html> --
# even a script tag -- puts the browser into quirks mode, which silently
# changes box sizing and layout out from under the CSS.
STATIC_FLAG = "<script>window.__STATIC__ = true;</script>"
HEAD_MARKER = '<meta charset="utf-8">'


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    return os.path.getsize(path)


def _clear(out_dir):
    # Clear the contents rather than the directory itself. Windows refuses to
    # remove a directory that is any process's working directory -- a local
    # preview server or a shell sitting in site/ is enough.
    #
    # Dot-entries are left alone. `.vercel/project.json` is the link between
    # this directory and the deployed project; deleting it makes the next
    # `vercel deploy` silently create a second project.
    if os.path.isdir(out_dir):
        for entry in os.listdir(out_dir):
            if entry.startswith("."):
                continue
            target = os.path.join(out_dir, entry)
            shutil.rmtree(target) if os.path.isdir(target) else os.remove(target)
    os.makedirs(out_dir, exist_ok=True)


def export(out_dir=DEFAULT_OUT):
    from .train import load_policy

    info = catalog.info(static=True)
    models = catalog.learned_models()
    loaded = {a: load_policy(a, path) for a, (_, path) in models.items()}
    _clear(out_dir)
    data = os.path.join(out_dir, "data")
    total, files = 0, 0
    t0 = time.perf_counter()

    total += _write_json(os.path.join(data, "info.json"), info)
    files += 1
    res = catalog.results_payload()
    if res is not None:
        total += _write_json(os.path.join(data, "results.json"), res)
        files += 1

    for city, spec in info["cities"].items():
        sc = load_scenario(city)
        total += _write_json(os.path.join(data, "map-{}.json".format(city)), map_payload(city))
        files += 1
        n = 0
        for i in range(spec["n_instances"]):
            for w in spec["worlds"]:
                total += _write_json(
                    os.path.join(data, "baseline", city, "{}-{}.json".format(i, w)),
                    baseline_trace(sc, i, w))
                files += 1
                for p in spec["policies"]:
                    if p in POLICIES:
                        pol, is_sb3, masked, prefs = POLICIES[p](), False, False, ["na"]
                    else:
                        pol, is_sb3, masked = loaded[p], True, p == "maskable_ppo"
                        prefs = spec["preferences"][p]
                    for pref in prefs:
                        trace = build_trace(
                            sc, i, w, pol, p,
                            config.HEADLINE_PREFERENCE if pref == "na" else pref,
                            is_sb3=is_sb3, masked=masked, include_baseline=False)
                        total += _write_json(os.path.join(
                            data, "traces", city, "{}-{}-{}-{}.json".format(p, pref, i, w)),
                            trace)
                        files += 1
                        n += 1
        print("  {:<10} {} traces".format(city, n), flush=True)

        # Real-street maps of the headline agent (or the best heuristic if
        # nothing is trained yet).
        lead = "maskable_ppo" if "maskable_ppo" in loaded else "greedy"
        for i in spec["real_maps"]:
            w = spec["worlds"][0]
            if lead in loaded:
                trace = build_trace(sc, i, w, loaded[lead], lead, config.HEADLINE_PREFERENCE,
                                    is_sb3=True, masked=True)
            else:
                trace = build_trace(sc, i, w, POLICIES[lead](), lead,
                                    config.HEADLINE_PREFERENCE)
            path = build_map(sc, trace, os.path.join(out_dir, "maps",
                                                     "{}-{}.html".format(city, i)))
            total += os.path.getsize(path)
            files += 1

    with open(DASHBOARD, encoding="utf-8") as f:
        html = f.read()
    if HEAD_MARKER not in html:
        raise RuntimeError("Cannot find {!r} in the dashboard -- the static flag has "
                           "nowhere safe to go.".format(HEAD_MARKER))
    html = html.replace(HEAD_MARKER, HEAD_MARKER + "\n" + STATIC_FLAG, 1)
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

    print("\n{} files, {:.1f} MB, in {:.0f}s -> {}".format(
        files + 1, total / 1048576, time.perf_counter() - t0, out_dir))
    print("Learned agents shown: " + (", ".join(
        "{} (seed {})".format(a, s) for a, (s, _) in models.items()) or "none"))
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    export(out_dir=args.out)


if __name__ == "__main__":
    main()

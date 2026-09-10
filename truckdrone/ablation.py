"""
Ablations: what does each piece of hardware actually buy?

Two knobs are studied, and neither answer is obvious in advance.

**Fleet size** -- how many drones the truck carries. One drone can remove at
most one stop per leg, which caps the saving whatever the dispatcher does.
Adding drones raises that cap, but each extra drone competes for the same
nearby customers and risks making the truck wait at the rendezvous.

**Battery packs per drone** -- whether each drone carries a spare. With one
pack, a drone's next sortie is gated by how fast that pack refills. With a
spare, the second pack has been charging on the truck the whole time the drone
was away, so turnaround costs a swap instead of a recharge. Which of those two
constraints is actually binding is an empirical question.

The agent is retrained at every setting, so each row measures the achievable
saving with that hardware rather than a policy applied outside the world it
learned in.

    python -m truckdrone.ablation --timesteps 200000
"""

import argparse
import json
import os

from . import config
from .baselines import POLICIES
from .evaluate import evaluate_policy, format_table, summarise
from .scenario import load_scenario, split_instances
from .train import MODEL_DIR, RESULT_DIR, load_policy, train

STUDIES = {
    "drones": {
        "param": "n_drones",
        "values": config.DRONE_COUNT_ABLATION,
        "label": "drone(s)",
        "out": "drone_ablation.json",
    },
    "batteries": {
        "param": "batteries_per_drone",
        "values": config.BATTERY_COUNT_ABLATION,
        "label": "battery pack(s) per drone",
        "out": "battery_ablation.json",
    },
}


def run_study(study, timesteps, skip_training=False):
    scenario = load_scenario()
    _, eval_ids = split_instances(scenario)
    param, results = study["param"], {}

    for value in study["values"]:
        env_kwargs = config.env_kwargs(**{param: value})
        tag = "maskable_ppo_{}{}".format(param[0], value)
        model_path = os.path.join(MODEL_DIR, tag, "best", "best_model.zip")

        if not skip_training and not os.path.exists(model_path):
            train("maskable_ppo", scenario, timesteps=timesteps,
                  env_kwargs=env_kwargs, tag=tag)

        print("\n=== {} {} ===".format(value, study["label"]))
        summaries = [summarise(evaluate_policy(cls(), scenario, eval_ids,
                                               env_kwargs=env_kwargs), name)
                     for name, cls in POLICIES.items()]

        if os.path.exists(model_path):
            policy = load_policy("maskable_ppo", model_path)
            summaries.append(summarise(
                evaluate_policy(policy, scenario, eval_ids, is_sb3=True,
                                masked=True, env_kwargs=env_kwargs),
                "maskable_ppo"))
        else:
            print("[skip] no model at {}".format(model_path))

        print(format_table(summaries))
        for s in summaries:
            results.setdefault(s["policy"], {})[str(value)] = s

    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", nargs="*", default=list(STUDIES),
                    choices=list(STUDIES))
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--skip-training", action="store_true",
                    help="evaluate existing models only")
    args = ap.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    for name in args.study:
        study = STUDIES[name]
        results = run_study(study, args.timesteps, args.skip_training)
        path = os.path.join(RESULT_DIR, study["out"])
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print("\n{} ablation -> {}".format(name, path))


if __name__ == "__main__":
    main()

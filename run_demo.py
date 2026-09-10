"""
One command to get from a fresh clone to a running demo.

Checks what has already been produced, does whatever is still missing, then
opens the dashboard. Safe to re-run: finished stages are skipped.

    python run_demo.py                 # train if needed, evaluate, serve
    python run_demo.py --serve-only    # just open the dashboard
    python run_demo.py --quick         # short training run, for a smoke test
"""

import argparse
import os
import subprocess
import sys
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "models")
RESULT_DIR = os.path.join(ROOT, "results")
ALGOS = ("maskable_ppo", "ppo", "dqn")


def run(args, label):
    print("\n" + "=" * 64)
    print("  " + label)
    print("=" * 64)
    result = subprocess.run([sys.executable, "-W", "ignore"] + args, cwd=ROOT)
    if result.returncode != 0:
        sys.exit("\nFailed: {}".format(label))


def have_model(algo):
    return (os.path.exists(os.path.join(MODEL_DIR, algo, "best", "best_model.zip"))
            or os.path.exists(os.path.join(MODEL_DIR, algo, "final.zip")))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serve-only", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="30k timesteps instead of 300k")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--ablation", action="store_true",
                    help="also run the fleet-size ablation (slow)")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    timesteps = 30_000 if args.quick else 300_000

    if not args.serve_only:
        missing = [a for a in ALGOS if args.retrain or not have_model(a)]
        for algo in missing:
            run(["-m", "truckdrone.train", "--algo", algo,
                 "--timesteps", str(timesteps)],
                "Training {} ({:,} steps)".format(algo, timesteps))
        if not missing:
            print("\nAll agents already trained. Use --retrain to redo them.")

        run(["-m", "truckdrone.evaluate"], "Evaluating on held-out instances")

        if args.ablation:
            run(["-m", "truckdrone.ablation", "--timesteps", str(timesteps)],
                "Fleet-size ablation")

        run(["-m", "truckdrone.report"], "Generating figures and results table")

    print("\n" + "=" * 64)
    print("  Dashboard at http://127.0.0.1:5000   (Ctrl-C to stop)")
    print("=" * 64 + "\n")
    if not args.no_browser:
        webbrowser.open("http://127.0.0.1:5000")
    subprocess.run([sys.executable, "-W", "ignore",
                    os.path.join(ROOT, "dashboard", "server.py")], cwd=ROOT)


if __name__ == "__main__":
    main()

"""
One command to get from a fresh clone to a running demo.

Checks what has already been produced, does whatever is still missing, then
opens the dashboard. Safe to re-run: finished stages are skipped.

    python run_demo.py                 # evaluate + figures if missing, then serve
    python run_demo.py --serve-only    # just open the dashboard
    python run_demo.py --train         # run the full training campaign first (hours)
    python run_demo.py --quick         # 50k-step MaskablePPO, for a smoke test
"""

import argparse
import os
import subprocess
import sys
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(ROOT, "results")
sys.path.insert(0, ROOT)


def run(args, label):
    print("\n" + "=" * 64)
    print("  " + label)
    print("=" * 64, flush=True)
    result = subprocess.run([sys.executable, "-W", "ignore"] + args, cwd=ROOT)
    if result.returncode != 0:
        sys.exit("\nFailed: {}".format(label))


def have(name):
    return os.path.exists(os.path.join(RESULT_DIR, name))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serve-only", action="store_true")
    ap.add_argument("--train", action="store_true",
                    help="run the full resumable training campaign (hours)")
    ap.add_argument("--quick", action="store_true",
                    help="train one MaskablePPO seed for 50k steps if none exists")
    ap.add_argument("--reevaluate", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    if not args.serve_only:
        from truckdrone.train import checkpoint
        if args.train:
            run(["-m", "truckdrone.campaign"], "Training campaign (resumable)")
        elif args.quick and not checkpoint("maskable_ppo", 0):
            run(["-m", "truckdrone.train", "--algo", "maskable_ppo", "--seeds", "0",
                 "--timesteps", "50000"], "Quick MaskablePPO run (50k steps)")
        if args.reevaluate or not have("evaluation.json"):
            run(["-m", "truckdrone.evaluate"], "Held-out evaluation")
        if args.reevaluate or not have("analysis.json"):
            run(["-m", "truckdrone.analysis"],
                "Preference trade-off, robustness, generalisation, transfer")
        if args.reevaluate or not have("ablation.json"):
            run(["-m", "truckdrone.ablation"], "Ablations")
        run(["-m", "truckdrone.report"], "Figures and results tables")

    print("\n" + "=" * 64)
    print("  Dashboard at http://127.0.0.1:5000   (Ctrl-C to stop)")
    print("=" * 64 + "\n", flush=True)
    if not args.no_browser:
        webbrowser.open("http://127.0.0.1:5000")
    subprocess.run([sys.executable, "-W", "ignore",
                    os.path.join(ROOT, "dashboard", "server.py")], cwd=ROOT)


if __name__ == "__main__":
    main()

"""
Figures and the results table for the write-up.

Reads what the other modules wrote -- the SB3 evaluation logs under
``models/`` and ``results/evaluation.json`` -- and produces the plots. Nothing
here recomputes a metric, so a figure can never disagree with the table it
sits next to.

    python -m truckdrone.report
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")
RESULT_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(RESULT_DIR, "figures")

INK = "#1b1f24"
GRID = "#d8dee4"
COLORS = {
    "maskable_ppo": "#2f6fed",
    "ppo": "#8957e5",
    "dqn": "#d29922",
    "greedy": "#3fb950",
    "always_nearest": "#0d9488",   # teal, distinct from greedy's green
    "random": "#8b949e",
    "truck_only": "#6e7681",
}

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": .7, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.spines.top": False,
    "axes.spines.right": False,
})


def _style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


# ======================================================================
# Figure 1 -- learning curves
# ======================================================================

def fig_learning_curves(algos, out):
    """
    Mean held-out episode return against environment steps.

    Each point is SB3's periodic evaluation on the instances the agent never
    trains on, so this shows generalisation rather than how well the agent has
    memorised its training set.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    all_means = []

    for algo in algos:
        path = os.path.join(MODEL_DIR, algo, "evaluations.npz")
        if not os.path.exists(path):
            continue
        with np.load(path) as z:
            steps, results = z["timesteps"], z["results"]
        mean = results.mean(axis=1)
        # Standard error, not standard deviation. The line is a mean over 20
        # evaluation episodes, so the meaningful band is the uncertainty in
        # that mean; plotting the raw spread of individual episodes drowns
        # every curve in overlapping haze.
        err = results.std(axis=1) / np.sqrt(results.shape[1])
        c = COLORS.get(algo, "#333")
        ax.plot(steps, mean, color=c, lw=2, label=algo.replace("_", " "))
        ax.fill_between(steps, mean - err, mean + err, color=c, alpha=.2,
                        linewidth=0)
        all_means.append(mean)

    if not all_means:
        plt.close(fig)
        return None

    # An untrained agent's first evaluation can sit orders of magnitude below
    # everything that follows, which would flatten the entire plot into a line
    # at the top. Clip to the region where the curves actually live and say so.
    stacked = np.concatenate(all_means)
    lo, hi = np.percentile(stacked, 3), stacked.max()
    pad = 0.12 * max(hi - lo, 1.0)
    clipped = stacked.min() < lo - pad
    ax.set_ylim(lo - pad, hi + pad)

    _style(ax, "Learning curves on held-out delivery instances",
           "environment steps",
           "mean episode return  (band: standard error)")
    if clipped:
        ax.text(0.99, 0.02, "early untrained evaluations clipped",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="#6e7681")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


# ======================================================================
# Figure 2 -- policy comparison
# ======================================================================

def fig_policy_comparison(summaries, out):
    """
    Delivery time and truck distance against the OR-Tools truck-only baseline.

    Bars point downwards because negative means better. Only policies that
    completed enough routes to support an average appear.
    """
    usable = [s for s in summaries if s.get("time_change_pct") is not None]
    if not usable:
        return None
    usable.sort(key=lambda s: s["time_change_pct"])

    labels = [s["policy"].replace("_", " ") for s in usable]
    time_pct = [s["time_change_pct"] for s in usable]
    km_pct = [s["truck_km_change_pct"] for s in usable]
    x = np.arange(len(usable))
    w = 0.38

    # Colour encodes the metric, not the policy, so the legend says what the
    # bars mean. Learned policies get a bolder tick label instead.
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.bar(x - w/2, time_pct, w, label="delivery time", color="#2f6fed")
    ax.bar(x + w/2, km_pct, w, label="truck distance", color="#9dc0f5")

    for xi, v in zip(x - w/2, time_pct):
        ax.text(xi, v - 0.6, "{:.1f}".format(v), ha="center", va="top",
                fontsize=8.5)

    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    for tick, s in zip(ax.get_xticklabels(), usable):
        if s["policy"] in ("maskable_ppo", "ppo", "dqn"):
            tick.set_fontweight("bold")
    _style(ax, "Change against the truck-only OR-Tools baseline "
               "(negative is better)", None, "% change")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


# ======================================================================
# Figure 3 -- the effect of action masking
# ======================================================================

def _smooth(values, window=200):
    if len(values) < window:
        return np.asarray(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(np.asarray(values, dtype=float), kernel, mode="valid")


def fig_masking_effect(summaries, out):
    """
    What action masking actually buys, in two panels.

    Left: illegal dispatch attempts per episode across training. The masked
    agent sits at exactly zero from the first step; the unmasked learners start
    high and spend a large part of their interaction budget discovering the
    feasibility rules from penalty signal.

    Right: how much of the delivery work the trained policy hands to the drone.
    This is the part that is easy to miss. The unmasked agents do eventually
    stop attempting illegal sorties -- but they get there by learning to stop
    flying, which is a safe policy and a useless one. Masking removes the
    penalty pressure that makes not-flying attractive, so the agent is free to
    learn *which* sortie is good rather than that sorties are dangerous.
    """
    learned = [s for s in summaries
               if s["policy"] in ("maskable_ppo", "ppo", "dqn")]
    if not learned:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))

    for s in learned:
        algo = s["policy"]
        path = os.path.join(MODEL_DIR, algo, "episode_metrics.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            metrics = json.load(f)
        fails = _smooth([m["failed_sorties"] for m in metrics])
        steps = np.linspace(0, metrics[-1]["timestep"], len(fails))
        axes[0].plot(steps, fails, lw=2, color=COLORS.get(algo, "#333"),
                     label=algo.replace("_", " "))

    _style(axes[0], "Illegal dispatch attempts during training",
           "environment steps", "per episode (moving average)")
    axes[0].legend(frameon=False)

    x = np.arange(len(learned))
    axes[1].bar(x, [s.get("drone_share_pct", 0) for s in learned],
                color=[COLORS.get(s["policy"], "#333") for s in learned])
    for xi, s in zip(x, learned):
        axes[1].text(xi, s.get("drone_share_pct", 0) + 1,
                     "{:.0f}%".format(s.get("drone_share_pct", 0)),
                     ha="center", fontsize=9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([s["policy"].replace("_", " ") for s in learned],
                            fontsize=9)
    _style(axes[1], "Deliveries flown by drone (held-out)", None,
           "% of customers")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


# ======================================================================
# Figure 4 -- fleet size ablation
# ======================================================================

def fig_ablation(path, out, xlabel, title, xticks_int=True):
    """Delivery-time saving against one hardware knob, per policy."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for policy, series in data.items():
        # Skip fleet sizes where the policy completed too few routes to be
        # given a delivery-time figure; plotting a gap is honest, inventing a
        # point is not.
        pairs = [(int(k), v["time_change_pct"]) for k, v in series.items()
                 if v.get("time_change_pct") is not None]
        if not pairs:
            continue
        pairs.sort()
        counts, vals = zip(*pairs)
        ax.plot(counts, vals, "o-", lw=2, ms=6,
                color=COLORS.get(policy, "#333"),
                label=policy.replace("_", " "))

    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(sorted({int(k) for s in data.values() for k in s}))
    _style(ax, title, xlabel, "% change vs truck-only")
    # Curves fan out to the lower right, so park the legend clear of them.
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


# ======================================================================
# Figure 6 -- what a solved instance looks like
# ======================================================================

def fig_route_map(out, policy="maskable_ppo", instance=0):
    """
    One delivery instance, solved two ways, side by side.

    The tables say the hybrid system drives less; this says *why*. On the left
    the truck alone must reach all fifteen doors. On the right the drones take
    the customers that sat furthest off the truck's path, and the truck's route
    visibly contracts. Drone legs are drawn straight because they are: a drone
    is not bound to the road graph, and that freedom is the entire mechanism.
    """
    from .rollout import make_rollout, view_coords
    from .scenario import build_graph, load_scenario

    scenario = load_scenario()
    graph = build_graph(scenario["coords"])
    try:
        trace = make_rollout(scenario, graph, policy, instance=instance)
    except FileNotFoundError:
        return None

    xy = view_coords(np.asarray(scenario["coords"], dtype=np.float64))
    pos = {n["id"]: (n["x"], n["y"]) for n in trace["nodes"]}
    customers, depot = set(trace["customers"]), trace["depot"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6))
    panels = [("baseline", "Truck only (OR-Tools TSP)"),
              (            "hybrid", "{} + 2 drones".format(policy.replace("_", " ")))]

    for ax, (key, title) in zip(axes, panels):
        for u, v in trace["edges"]:                       # road network
            ax.plot(*zip(pos[u], pos[v]), color="#d8dee4", lw=1.0, zorder=1)

        for leg in trace[key]["legs"]:                    # driven route
            pts = [pos[n] for n in leg["truck_path"]]
            ax.plot(*zip(*pts), color="#2f6fed", lw=2.6, zorder=3,
                    solid_capstyle="round")
            for s in leg["sorties"]:                      # straight-line sorties
                ax.plot(*zip(pos[s["launch"]], pos[s["customer"]]),
                        color="#db6d28", lw=1.9, ls=(0, (5, 3)), zorder=4)
                ax.plot(*zip(pos[s["customer"]], pos[s["recovery"]]),
                        color="#db6d28", lw=1.4, ls=(0, (1, 3)), alpha=.75,
                        zorder=4)
                ax.plot(*pos[s["customer"]], "o", ms=8, mfc="#db6d28",
                        mec="white", mew=1.4, zorder=6)

        served_by_truck = {l["truck_delivery"] for l in trace[key]["legs"]}
        for c in customers:
            if c in served_by_truck:
                ax.plot(*pos[c], "o", ms=7, mfc="#2f6fed", mec="white",
                        mew=1.3, zorder=5)
        ax.plot(*pos[depot], "s", ms=12, mfc="#8957e5", mec="white", mew=1.6,
                zorder=7)

        st = trace[key]["stats"]
        ax.set_title("{}\n{:.0f} min  ·  {:.0f} km driven  ·  {} by drone"
                     .format(title, st["time_min"], st["truck_km"],
                             st["drone_deliveries"]),
                     fontsize=10.5, fontweight="bold", pad=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_visible(False)

    handles = [
        plt.Line2D([], [], color="#2f6fed", lw=2.6, label="truck (road network)"),
        plt.Line2D([], [], color="#db6d28", lw=1.9, ls=(0, (5, 3)),
                   label="drone out (straight line)"),
        plt.Line2D([], [], color="#db6d28", lw=1.4, ls=(0, (1, 3)),
                   label="drone back to truck"),
        plt.Line2D([], [], color="#8957e5", marker="s", ls="", ms=9,
                   label="depot"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("One delivery instance, solved two ways "
                 "(Whitefield, Bengaluru — 15 customers)",
                 fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out)
    plt.close(fig)
    return out


# ======================================================================
# Markdown results table
# ======================================================================

def results_table(summaries, n_instances, n_customers):
    rows = ["| policy | routes completed | delivery time | vs baseline | "
            "truck km | vs baseline | delivered by drone | illegal dispatches |",
            "|---|---|---|---|---|---|---|---|"]
    for s in sorted(summaries,
                    key=lambda s: (s["time_change_pct"] is None,
                                   s.get("time_change_pct", 0))):
        if s["time_change_pct"] is None:
            rows.append("| {} | {:.0f}% | — | — | — | — | — | {:.1f} |".format(
                s["policy"].replace("_", " "), s["completion_rate_pct"],
                s["failed_sorties_mean"]))
            continue
        rows.append(
            "| {} | {:.0f}% | {:.1f} min | {:+.1f}% | {:.1f} km | {:+.1f}% | "
            "{:.0f}% | {:.1f} |".format(
                s["policy"].replace("_", " "), s["completion_rate_pct"],
                s["hybrid_time_min_mean"], s["time_change_pct"],
                s["truck_km_mean"], s["truck_km_change_pct"],
                s["drone_share_pct"], s["failed_sorties_mean"]))
    header = ("Held-out evaluation: {} delivery instances, {} customers each. "
              "Time and distance are averaged over completed routes only.\n"
              .format(n_instances, n_customers))
    return header + "\n".join(rows)


def significance_table(tests):
    """The paired tests, as markdown, so the claim travels with the numbers."""
    if not tests:
        return ""
    rows = ["", "",
            "Paired comparison for **{}** on the same {} instances. "
            "Negative favours it.".format(
                tests[0]["subject"].replace("_", " "), tests[0]["n_instances"]),
            "",
            "| vs | mean difference | instances won | paired t | Wilcoxon |",
            "|---|---|---|---|---|"]
    for t in tests:
        rows.append("| {} | {:+.1f} min | {}/{} | {:.1e} | {:.1e} |".format(
            t["opponent"].replace("_", " "), t["mean_diff_min"],
            t["wins"], t["n_instances"], t["paired_t_p"], t["wilcoxon_p"]))
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval", default=os.path.join(RESULT_DIR, "evaluation.json"))
    ap.add_argument("--drone-ablation",
                    default=os.path.join(RESULT_DIR, "drone_ablation.json"))
    ap.add_argument("--battery-ablation",
                    default=os.path.join(RESULT_DIR, "battery_ablation.json"))
    args = ap.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    made = []

    made.append(fig_learning_curves(
        ("maskable_ppo", "ppo", "dqn"),
        os.path.join(FIG_DIR, "fig1_learning_curves.png")))

    if os.path.exists(args.eval):
        with open(args.eval) as f:
            data = json.load(f)
        summaries = data["summaries"]
        made.append(fig_policy_comparison(
            summaries, os.path.join(FIG_DIR, "fig2_policy_comparison.png")))
        made.append(fig_masking_effect(
            summaries, os.path.join(FIG_DIR, "fig3_masking_effect.png")))

        table = results_table(summaries, data["n_eval_instances"],
                              data["n_customers"])
        table += significance_table(data.get("significance", []))
        table_path = os.path.join(RESULT_DIR, "results_table.md")
        with open(table_path, "w") as f:
            f.write(table + "\n")
        print("\n" + table + "\n")
        made.append(table_path)
    else:
        print("[skip] no evaluation.json -- run: python -m truckdrone.evaluate")

    made.append(fig_ablation(
        args.drone_ablation, os.path.join(FIG_DIR, "fig4_drone_ablation.png"),
        "drones carried by the truck",
        "Delivery-time saving against fleet size"))

    made.append(fig_ablation(
        args.battery_ablation,
        os.path.join(FIG_DIR, "fig5_battery_ablation.png"),
        "battery packs per drone (1 = no spare)",
        "What a hot-swappable spare battery buys"))

    made.append(fig_route_map(
        os.path.join(FIG_DIR, "fig6_route_map.png")))

    for path in [m for m in made if m]:
        print("  wrote {}".format(os.path.relpath(path, ROOT)))


if __name__ == "__main__":
    main()

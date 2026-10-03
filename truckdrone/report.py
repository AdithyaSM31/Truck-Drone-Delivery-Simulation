"""
Figures and the results tables for the write-up.

Reads what the other modules wrote -- the training logs under ``models/``,
``results/evaluation.json``, ``results/analysis.json`` and
``results/ablation.json`` -- and draws them. Nothing here recomputes a metric,
so a figure can never disagree with the table it sits next to.

    python -m truckdrone.report
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import matplotlib.ticker                 # noqa: E402,F401
import numpy as np                       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")
RESULT_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(RESULT_DIR, "figures")

INK = "#1b1f24"
MUTED = "#6e7681"
GRID = "#d8dee4"
TRUCK, DRONE = "#2f6fed", "#db6d28"
COLORS = {
    "maskable_ppo": "#2f6fed",
    "ppo": "#8957e5",
    "dqn": "#d29922",
    "greedy": "#3fb950",
    "always_nearest": "#0d9488",
    "random": "#8b949e",
    "truck_only": "#6e7681",
}
METRIC_COLORS = {"time": "#2f6fed", "cost": "#3fb950", "energy": "#d29922"}
LEARNED = ("maskable_ppo", "ppo", "dqn")

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


def _label(p):
    return p.replace("_", " ")


def _load(name):
    path = os.path.join(RESULT_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _millions(ax):
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: "{:g}M".format(v / 1e6) if v else "0"))


def _save(fig, out):
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ======================================================================
# Figure 1 -- learning curves across seeds
# ======================================================================

def fig_learning_curves(out):
    """
    Mean return on the held-out *selection* worlds against environment steps,
    averaged over training seeds; the band is the spread (min-max) across
    seeds, which is the run-to-run variation a reader would see retraining.
    """
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    found = False
    for algo in LEARNED:
        curves = []
        for path in sorted(glob.glob(os.path.join(MODEL_DIR, algo, "seed*", "evaluations.npz"))):
            with np.load(path) as z:
                curves.append((z["timesteps"], z["results"].mean(axis=1)))
        if not curves:
            continue
        n = min(len(c[0]) for c in curves)
        steps = curves[0][0][:n]
        stack = np.stack([c[1][:n] for c in curves])
        c = COLORS[algo]
        ax.plot(steps, stack.mean(0), color=c, lw=2,
                label="{} ({} seed{})".format(_label(algo), len(curves),
                                             "s" if len(curves) > 1 else ""))
        if len(curves) > 1:
            ax.fill_between(steps, stack.min(0), stack.max(0), color=c, alpha=.18, lw=0)
        found = True
    if not found:
        plt.close(fig)
        return None
    lo, hi = ax.get_ylim()
    ys = np.concatenate([l.get_ydata() for l in ax.get_lines()])
    floor = np.percentile(ys, 3)
    if floor - lo > 0.2 * (hi - lo):
        ax.set_ylim(floor - 0.1 * (hi - floor), hi)
        ax.text(0.99, 0.02, "early untrained evaluations clipped", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color=MUTED)
    _style(ax, "Learning curves on held-out selection worlds", "environment steps",
           "mean episode return  (band: range over seeds)")
    ax.legend(frameon=False, loc="lower right")
    _millions(ax)
    return _save(fig, out)


# ======================================================================
# Figure 2 -- policy comparison with seed CIs
# ======================================================================

def fig_policy_comparison(summaries, out):
    usable = [s for s in summaries if s.get("time_change_pct") is not None
              and s["policy"] != "truck_only"]
    if not usable:
        return None
    usable.sort(key=lambda s: s["time_change_pct"])
    x = np.arange(len(usable))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    for k, m in enumerate(("time", "cost", "energy")):
        vals = [s[m + "_change_pct"] for s in usable]
        err = [s.get(m + "_change_pct_ci95", 0) or 0 for s in usable]
        ax.bar(x + (k - 1) * w, vals, w, yerr=err, capsize=2.5, color=METRIC_COLORS[m],
               label={"time": "delivery time", "cost": "operating cost (₹)",
                      "energy": "energy (kWh)"}[m], error_kw={"lw": .9, "ecolor": INK})
    for xi, s in zip(x, usable):
        ax.text(xi - w, s["time_change_pct"] - (s.get("time_change_pct_ci95") or 0) - .8,
                "{:.1f}".format(s["time_change_pct"]), ha="center", va="top", fontsize=8)
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([_label(s["policy"]) for s in usable], fontsize=9)
    for tick, s in zip(ax.get_xticklabels(), usable):
        if s["policy"] in LEARNED:
            tick.set_fontweight("bold")
    _style(ax, "Change against the truck-only OR-Tools tour in the same world "
               "(negative is better)", None, "% change  (error bar: 95% CI over seeds)")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    return _save(fig, out)


# ======================================================================
# Figure 3 -- action masking
# ======================================================================

def _smooth(values, window=200):
    if len(values) < window:
        return np.asarray(values, dtype=float)
    return np.convolve(np.asarray(values, dtype=float), np.ones(window) / window, mode="valid")


def fig_masking_effect(summaries, out):
    """
    Left: illegal dispatch attempts per training episode (seed 0). The masked
    agent is at zero from the first step; the unmasked learners pay to learn
    feasibility from penalties. Right: share of deliveries flown, held out.
    """
    learned = [s for s in summaries if s["policy"] in LEARNED]
    if not learned:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for s in learned:
        path = os.path.join(MODEL_DIR, s["policy"], "seed0", "episode_metrics.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            metrics = json.load(f)
        fails = _smooth([m["failed_sorties"] for m in metrics])
        steps = np.linspace(0, metrics[-1]["timestep"], len(fails))
        axes[0].plot(steps, fails, lw=2, color=COLORS[s["policy"]], label=_label(s["policy"]))
    _style(axes[0], "Illegal dispatch attempts during training", "environment steps",
           "per episode (moving average)")
    axes[0].legend(frameon=False)
    _millions(axes[0])

    x = np.arange(len(learned))
    vals = [s.get("drone_share_pct") or 0 for s in learned]
    axes[1].bar(x, vals, yerr=[s.get("drone_share_pct_ci95", 0) or 0 for s in learned],
                capsize=3, color=[COLORS[s["policy"]] for s in learned])
    for xi, v, s in zip(x, vals, learned):
        axes[1].text(xi + .12, v + 1, "{:.0f}%".format(v), ha="left", fontsize=9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([_label(s["policy"]) for s in learned], fontsize=9)
    _style(axes[1], "Deliveries flown by drone (held-out)", None, "% of customers")
    return _save(fig, out)


# ======================================================================
# Figure 4 -- the preference trade-off (MORL)
# ======================================================================

def fig_pareto(analysis, ablation, out):
    """
    The preference-conditioned agent asked for 15 weightings, against fixed
    policies. Reference agents come from the ablation runs so the comparison
    is at a matched 1M-step budget: the same conditioned agent, and a
    specialist trained for time alone.
    """
    p = (analysis or {}).get("pareto")
    if not p:
        return None
    pts = [q for q in p["points"] if q.get("time_change_pct") is not None]
    refs = [r for r in p.get("references", []) if r["policy"] in ("greedy", "always_nearest")]
    abl = {r["tag"]: r["agent"] for r in (ablation or {}).get("rows", [])}
    for tag, name, colour in (("abl_reference", "conditioned, 1M steps", "#2f6fed"),
                              ("abl_specialist", "time-only specialist, 1M", "#d1242f")):
        if tag in abl:
            refs.append(dict(abl[tag], policy=name, colour=colour))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, other in zip(axes, ("cost", "energy")):
        sc = ax.scatter([q["time_change_pct"] for q in pts],
                        [q[other + "_change_pct"] for q in pts],
                        c=[q["weights"]["time"] for q in pts], cmap="viridis", s=55,
                        edgecolor="white", lw=.8, zorder=3)
        xs = [q["time_change_pct"] for q in pts]
        ys = [q[other + "_change_pct"] for q in pts]
        ax.annotate("← conditioned agent (2M, 5 seeds):\n    all 15 weightings",
                    (max(xs), float(np.mean(ys))), xytext=(10, 0),
                    textcoords="offset points", fontsize=8.5, va="center")
        for r in refs:
            colour = r.get("colour") or COLORS.get(r["policy"], "#333")
            ax.scatter(r["time_change_pct"], r[other + "_change_pct"], marker="D", s=55,
                       color=colour, edgecolor="white", zorder=4)
            above = "colour" in r          # matched-budget agents sit near the cluster
            ax.annotate(_label(r["policy"]), (r["time_change_pct"], r[other + "_change_pct"]),
                        xytext=(-6, 9) if above else (9, -3), textcoords="offset points",
                        fontsize=8.5, color=MUTED, ha="left",
                        va="bottom" if above else "center")
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo - 0.3, hi + 1.6)
        _style(ax, "Time vs {}".format(other), "delivery time, % vs truck-only",
               "{}, % vs truck-only".format(other))
    cb = fig.colorbar(sc, ax=axes, shrink=.85, pad=.02)
    cb.set_label("weight on time")
    fig.text(0.02, 1.0, "Every weighting lands in the same place: the three objectives are "
             "aligned in this world", fontsize=11.5, fontweight="bold", ha="left", va="bottom")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ======================================================================
# Figure 5 -- robustness
# ======================================================================

def _series(blocks, key):
    out = {}
    for b in blocks:
        for r in b["results"]:
            out.setdefault(r["policy"], []).append(
                (b[key], r.get("time_change_pct"), r.get("time_change_pct_ci95") or 0))
    return out


def fig_robustness(analysis, out):
    rb = (analysis or {}).get("robustness")
    if not rb:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), gridspec_kw={"width_ratios": [1.2, 1]})
    for pol, rows in _series(rb["wind"], "wind_ms").items():
        rows = [r for r in rows if r[1] is not None]
        x, y, e = zip(*rows)
        axes[0].errorbar(x, y, yerr=e, marker="o", lw=2, capsize=3, color=COLORS.get(pol),
                         label=_label(pol))
    tmax = rb.get("training_wind_max_ms", 8)
    axes[0].axvspan(tmax, max(b["wind_ms"] for b in rb["wind"]) + .5, color="#f6f8fa", zorder=0)
    axes[0].text(tmax + .2, axes[0].get_ylim()[1], "beyond training range", va="top",
                 fontsize=8, color=MUTED)
    axes[0].axhline(0, color=INK, lw=1)
    _style(axes[0], "Wind speed (direction random)", "wind, m/s", "time, % vs truck-only")
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")

    series = _series(rb["traffic"], "regime")
    regimes = [b["regime"] for b in rb["traffic"]]
    x = np.arange(len(regimes))
    w = 0.8 / max(1, len(series))
    for k, (pol, rows) in enumerate(series.items()):
        vals = [r[1] if r[1] is not None else 0 for r in rows]
        axes[1].bar(x + (k - (len(series) - 1) / 2) * w, vals, w,
                    yerr=[r[2] for r in rows], capsize=2.5, color=COLORS.get(pol),
                    label=_label(pol))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(regimes)
    axes[1].axhline(0, color=INK, lw=1)
    _style(axes[1], "Traffic regime", None, "time, % vs truck-only")
    return _save(fig, out)


# ======================================================================
# Figure 6 -- generalisation: problem size and a new city
# ======================================================================

def fig_generalisation(analysis, evaluation, out):
    a = analysis or {}
    if not a.get("scale") and not a.get("transfer"):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    main = {s["policy"]: s for s in (evaluation or {}).get("summaries", [])}

    sizes = {}
    for block in a.get("scale", []):
        for r in block["results"]:
            sizes.setdefault(r["policy"], {})[block["customers"]] = r
    for pol, by_n in sizes.items():
        if pol in main:
            by_n[15] = main[pol]
        pts = sorted((n, r["time_change_pct"], r.get("time_change_pct_ci95") or 0)
                     for n, r in by_n.items() if r.get("time_change_pct") is not None)
        if pts:
            n, y, e = zip(*pts)
            axes[0].errorbar(n, y, yerr=e, marker="o", lw=2, capsize=3,
                             color=COLORS.get(pol), label=_label(pol))
    axes[0].axvline(15, color=GRID, lw=6, zorder=0)
    axes[0].text(15, axes[0].get_ylim()[1], " trained on 15", va="top", fontsize=8, color=MUTED)
    axes[0].set_xticks([10, 15, 20])
    _style(axes[0], "Unseen problem sizes", "customers per day", "time, % vs truck-only")
    axes[0].legend(frameon=False, fontsize=9)

    t = a.get("transfer")
    if t:
        pols = [r["policy"] for r in t["results"]]
        x = np.arange(len(pols))
        home = [main.get(p, {}).get("time_change_pct") for p in pols]
        away = [r.get("time_change_pct") for r in t["results"]]
        axes[1].bar(x - .2, [h or 0 for h in home], .4, color="#9dc0f5",
                    label="Whitefield (trained)")
        axes[1].bar(x + .2, [v or 0 for v in away], .4,
                    yerr=[r.get("time_change_pct_ci95") or 0 for r in t["results"]],
                    capsize=3, color=TRUCK, label="Chennai (zero-shot)")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([_label(p) for p in pols])
        axes[1].axhline(0, color=INK, lw=1)
        _style(axes[1], "A city it has never seen", None, "time, % vs truck-only")
        axes[1].legend(frameon=False, fontsize=9, loc="lower right")
    return _save(fig, out)


# ======================================================================
# Figure 7 -- ablations
# ======================================================================

def fig_ablation(ablation, out):
    rows = [r for r in (ablation or {}).get("rows", [])
            if r["agent"].get("time_change_pct") is not None]
    if not rows:
        return None
    ref = next((r for r in rows if r["tag"] == "abl_reference"), None)
    rows.sort(key=lambda r: (r["tag"] != "abl_reference", r["agent"]["time_change_pct"]))
    fig, ax = plt.subplots(figsize=(8.8, 0.5 * len(rows) + 1.4))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r["agent"]["time_change_pct"] for r in rows], .55,
            color=[TRUCK if r["tag"] == "abl_reference" else
                   ("#8957e5" if r["kind"] == "training" else "#9dc0f5") for r in rows])
    gx = [r["greedy_same_world"].get("time_change_pct") for r in rows]
    ax.scatter([g if g is not None else np.nan for g in gx], y, marker="|", s=260,
               color=COLORS["greedy"], lw=2.5, zorder=4)
    if ref:
        ax.axvline(ref["agent"]["time_change_pct"], color=TRUCK, lw=1, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([r["description"] for r in rows], fontsize=8.5)
    ax.axvline(0, color=INK, lw=1)
    for yi, r in zip(y, rows):
        ax.text(r["agent"]["time_change_pct"] - .4, yi, "{:+.1f}%".format(
            r["agent"]["time_change_pct"]), ha="right", va="center", fontsize=8.5,
            color=INK, fontweight="bold")
    _style(ax, "Ablations (MaskablePPO, 1M steps, seed 0): delivery time vs truck-only",
           "% change   ·   dark: reference · light: capability removed · purple: training choice"
           "   ·   green tick: greedy in the same world")
    ax.set_xlim(min(r["agent"]["time_change_pct"] for r in rows) - 4.5, 0.5)
    return _save(fig, out)


# ======================================================================
# Figure 8 -- one solved instance on real streets
# ======================================================================

def fig_route_map(out, instance=0):
    from . import catalog, config
    from .baselines import POLICIES
    from .rollout import build_trace
    from .scenario import StreetPaths, load_scenario

    sc = load_scenario(config.CITY)
    models = catalog.learned_models()
    world = config.EVAL_WORLD_SEEDS[0]
    if "maskable_ppo" in models:
        from .train import load_policy
        label = "MaskablePPO (seed {})".format(models["maskable_ppo"][0])
        trace = build_trace(sc, instance, world, load_policy(
            "maskable_ppo", models["maskable_ppo"][1]), "maskable_ppo",
            config.HEADLINE_PREFERENCE, is_sb3=True, masked=True)
    else:
        label = "greedy"
        trace = build_trace(sc, instance, world, POLICIES["greedy"](), "greedy",
                            config.HEADLINE_PREFERENCE)
    paths = StreetPaths(config.CITY)
    xy = sc["coords"] / 1000.0
    customers, depot = trace["customers"], trace["depot"]
    stops = xy[[depot] + customers]
    pad = 0.6
    box = (stops[:, 0].min() - pad, stops[:, 0].max() + pad,
           stops[:, 1].min() - pad, stops[:, 1].max() + pad)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
    panels = [("baseline", "Truck only (OR-Tools tour)", trace["baseline_stats"]),
              ("hybrid", label + " + 2 drones", trace["hybrid"]["stats"])]
    for ax, (key, title, st) in zip(axes, panels):
        for line in paths.background_xy():
            line = np.asarray(line) / 1000.0
            ax.plot(line[:, 0], line[:, 1], color="#e4e8ec", lw=.7, zorder=1)
        part = trace[key]
        for leg in part["legs"]:
            if leg["from"] == leg["to"]:
                continue
            p = np.asarray(paths.xy(leg["from"], leg["to"])) / 1000.0
            ax.plot(p[:, 0], p[:, 1], color=TRUCK, lw=2.4, zorder=3, solid_capstyle="round")
        flown = set()
        for s in part.get("sorties", []):
            flown.add(s["customer"])
            ax.plot(*zip(xy[s["launch"]], xy[s["customer"]]), color=DRONE, lw=1.8,
                    ls=(0, (5, 3)), zorder=4)
            ax.plot(*zip(xy[s["customer"]], xy[s["recovery"]]), color=DRONE, lw=1.2,
                    ls=(0, (1, 3)), alpha=.8, zorder=4)
        for c in customers:
            ax.plot(*xy[c], "o", ms=8 if c in flown else 7,
                    mfc=DRONE if c in flown else TRUCK, mec="white", mew=1.3, zorder=5)
        ax.plot(*xy[depot], "s", ms=12, mfc="#8957e5", mec="white", mew=1.6, zorder=7)
        ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3])
        ax.set_title("{}\n{:.0f} min · {:.1f} km driven · ₹{:.0f}{}".format(
            title, st["time_min"], st["truck_km"], st["cost_inr"],
            " · {} by drone".format(st["drone_deliveries"]) if key == "hybrid" else ""),
            fontsize=10.5, fontweight="bold", pad=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_visible(False)
    handles = [
        plt.Line2D([], [], color=TRUCK, lw=2.4, label="truck on real streets (OSM)"),
        plt.Line2D([], [], color=DRONE, lw=1.8, ls=(0, (5, 3)), label="drone out, one parcel"),
        plt.Line2D([], [], color=DRONE, lw=1.2, ls=(0, (1, 3)), label="drone to rendezvous"),
        plt.Line2D([], [], color="#8957e5", marker="s", ls="", ms=9, label="depot"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Whitefield, Bengaluru — held-out instance #{}, same traffic and wind for both"
                 .format(instance + 1), fontsize=12, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(out)
    plt.close(fig)
    return out


# ======================================================================
# Markdown tables
# ======================================================================

def _ci(s, k):
    v, ci = s.get(k), s.get(k + "_ci95")
    if v is None:
        return "—"
    return "{:+.1f}%{}".format(v, " ± {:.1f}".format(ci) if ci else "")


def results_table(ev):
    rows = ["| policy | seeds | routes completed | time | cost | energy | CO₂ | truck km "
            "| by drone | illegal dispatches |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for s in sorted(ev["summaries"], key=lambda s: (s.get("time_change_pct") is None,
                                                    s.get("time_change_pct") or 0)):
        rows.append("| {} | {} | {:.0f}% | {} | {} | {} | {} | {} | {} | {:.2f} |".format(
            _label(s["policy"]), s.get("n_seeds", "—"), s["completion_rate_pct"],
            _ci(s, "time_change_pct"), _ci(s, "cost_change_pct"),
            _ci(s, "energy_change_pct"), _ci(s, "co2_change_pct"),
            _ci(s, "truck_km_change_pct"),
            "{:.0f}%".format(s["drone_share_pct"]) if s.get("drone_share_pct") is not None
            else "—", s["failed_sorties_mean"]))
    head = ("Held-out evaluation, {}: {} instances × {} worlds, {} customers each, "
            "preference '{}'. Paired % change vs the OR-Tools truck-only tour in the same "
            "world, completed routes only; ± is the 95% CI across training seeds.\n\n").format(
        ev.get("city", ""), ev["n_eval_instances"], len(ev["world_seeds"]),
        ev["n_customers"], ev["preference"])
    return head + "\n".join(rows)


def significance_table(tests):
    out = []
    for metric, ts in (tests or {}).items():
        if not ts:
            continue
        unit = {"time_min": "min", "cost_inr": "₹", "energy_kwh": "kWh"}.get(metric, "")
        out += ["", "", "Paired tests on **{}** for {} (n = {} instance-world pairs; "
                "negative favours it):".format(metric, _label(ts[0]["subject"]), ts[0]["n"]),
                "", "| vs | mean difference | won | paired t p | Wilcoxon p |",
                "|---|---|---|---|---|"]
        for t in ts:
            out.append("| {} | {:+.2f} {} | {}/{} | {:.1e} | {:.1e} |".format(
                _label(t["opponent"]), t["mean_diff"], unit, t["wins"], t["n"],
                t["paired_t_p"], t["wilcoxon_p"]))
    return "\n".join(out)


def analysis_tables(analysis, ablation):
    out = []
    a = analysis or {}
    if a.get("robustness"):
        rb = a["robustness"]
        out += ["", "", "**Robustness** (time % vs truck-only):", "",
                "| setting | " + " | ".join(_label(r["policy"])
                                          for r in rb["wind"][0]["results"]) + " |",
                "|---|" + "---|" * len(rb["wind"][0]["results"])]
        for b in rb["wind"]:
            out.append("| wind {:.0f} m/s | ".format(b["wind_ms"]) + " | ".join(
                _ci(r, "time_change_pct") for r in b["results"]) + " |")
        for b in rb["traffic"]:
            out.append("| {} traffic | ".format(b["regime"]) + " | ".join(
                _ci(r, "time_change_pct") for r in b["results"]) + " |")
    if a.get("scale") or a.get("transfer"):
        out += ["", "", "**Generalisation** (time % vs truck-only):", ""]
        blocks = [("{} customers".format(b["customers"]), b["results"]) for b in a.get("scale", [])]
        if a.get("transfer"):
            blocks.append(("Chennai (zero-shot)", a["transfer"]["results"]))
        pols = [r["policy"] for r in blocks[0][1]]
        out += ["| test set | " + " | ".join(_label(p) for p in pols) + " |",
                "|---|" + "---|" * len(pols)]
        for name, res in blocks:
            out.append("| {} | ".format(name) + " | ".join(
                _ci(r, "time_change_pct") for r in res) + " |")
    if ablation and ablation.get("rows"):
        out += ["", "", "**Ablations** (MaskablePPO, seed 0, 1M steps each):", "",
                "| variant | agent time | greedy, same world | agent vs reference |",
                "|---|---|---|---|"]
        for r in ablation["rows"]:
            d = (r.get("vs_reference_pp") or {}).get("time_change_pct")
            out.append("| {} | {} | {} | {} |".format(
                r["description"], _ci(r["agent"], "time_change_pct"),
                _ci(r["greedy_same_world"], "time_change_pct"),
                "—" if d is None or r["tag"] == "abl_reference" else "{:+.1f} pp".format(d)))
    return "\n".join(out)


def main():
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")       # tables contain CO₂, ₹, ±
    argparse.ArgumentParser(description=__doc__).parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    for old in glob.glob(os.path.join(FIG_DIR, "*.png")):
        os.remove(old)
    ev, an, ab = _load("evaluation.json"), _load("analysis.json"), _load("ablation.json")
    f = lambda name: os.path.join(FIG_DIR, name)       # noqa: E731
    made = [fig_learning_curves(f("fig1_learning_curves.png"))]
    if ev:
        made.append(fig_policy_comparison(ev["summaries"], f("fig2_policy_comparison.png")))
        made.append(fig_masking_effect(ev["summaries"], f("fig3_masking_effect.png")))
    made.append(fig_pareto(an, ab, f("fig4_preference_tradeoff.png")))
    made.append(fig_robustness(an, f("fig5_robustness.png")))
    made.append(fig_generalisation(an, ev, f("fig6_generalisation.png")))
    made.append(fig_ablation(ab, f("fig7_ablations.png")))
    made.append(fig_route_map(f("fig8_route_map.png")))

    if ev:
        table = (results_table(ev) + significance_table(ev.get("significance"))
                 + analysis_tables(an, ab))
        path = os.path.join(RESULT_DIR, "results_table.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(table + "\n")
        made.append(path)
        print("\n" + table + "\n")
    for path in [m for m in made if m]:
        print("  wrote {}".format(os.path.relpath(path, ROOT)))


if __name__ == "__main__":
    main()

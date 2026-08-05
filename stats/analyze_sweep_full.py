#!/usr/bin/env python
"""Consolidated analysis of the scenario sweep in stats/out/sweep_full_100.

The sweep evaluates a 2x2 ablation (communication on/off x centralized critic
on/off) plus a random floor, over 3 spawn modes x 2 target modes x 3 deployment
comms radii x 2 swarm sizes, 100 paired episodes each.

Emits a markdown report and the figures next to it:

    python stats/analyze_sweep_full.py [--sweep DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import binomtest, false_discovery_control, wilcoxon

SPAWNS = ["random", "origin", "max_dist"]
TARGETS = ["random", "tail"]
RADII = ["inf", "250", "0"]
ARMS = ["ippo", "ippo_comm", "mappo", "mappo_comm", "random"]
POLICY_ARMS = ARMS[:-1]
NS = [2, 4]

# Trained configuration -- the single in-distribution deployment cell.
TRAIN_SPAWN, TRAIN_TARGET, TRAIN_RADIUS = "random", "random", "inf"

ARM_LABEL = {
    "ippo": "IPPO",
    "ippo_comm": "IPPO + comm",
    "mappo": "MAPPO",
    "mappo_comm": "MAPPO + comm",
    "random": "random",
}
SPAWN_LABEL = {
    "random": "random spawn\n(dispersed)",
    "origin": "origin spawn\n(co-located)",
    "max_dist": "max_dist spawn\n(coastline)",
}
TARGET_LABEL = {"random": "random target", "tail": "tail target"}

# --- validated palette (dataviz skill reference instance, light mode) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# Arm colours are chosen so that BOTH head-to-head pairs a reader actually makes --
# the two comm arms against each other (fig4) and the two non-comm arms -- clear the
# all-pairs CVD/normal-vision floors, while the four together clear the adjacent
# floors as grouped bars (fig2). Verified with the dataviz validator; the obvious
# slot order 1-2-3-4 fails, because it sits orange beside yellow.
SERIES = ["#2a78d6", "#1baf7a", "#eb6834", "#4a3aa7"]  # blue, aqua, orange, violet
POS, NEG = "#2a78d6", "#e34948"  # diverging poles: blue <-> red
ARM_COLOR = dict(zip(POLICY_ARMS, SERIES))


# ----------------------------------------------------------------- loading
def load(sweep: Path):
    """Return (cell dataframe, paired-episode frames keyed by scenario)."""
    rows, paired = [], {}
    for sp, tg, rd, n in itertools.product(SPAWNS, TARGETS, RADII, NS):
        d = json.loads((sweep / f"{sp}__{tg}__radius-{rd}" / f"N{n}" / "summary.json").read_text())
        es, res = d["episode_stats"], d["results"]
        for arm in ARMS:
            r = res[arm]
            rows.append(
                dict(
                    N=n, spawn=sp, target=tg, radius=rd, arm=arm,
                    any_rate=es[arm]["any_rate"],
                    any_lo=es[arm]["any_ci"][0], any_hi=es[arm]["any_ci"][1],
                    per_agent=r["success_rate"], spl=r["spl_mean"],
                    t_med=r["steps_to_success"]["median"],
                    swarm_path=es[arm]["swarm_path"],
                    cov_red=es[arm]["coverage_redundancy"],
                    nn_dist=es[arm]["nn_distance"],
                )
            )
        paired[(n, sp, tg, rd)] = pd.read_csv(
            sweep / f"{sp}__{tg}__radius-{rd}" / f"N{n}" / "paired_episodes.csv"
        )
    return pd.DataFrame(rows), paired


def check_invariants(df: pd.DataFrame, paired: dict) -> list[str]:
    """The sweep's structural assumptions, verified rather than assumed."""
    out = []
    for arm in ["ippo", "mappo", "random"]:
        n_unique = df[df.arm == arm].groupby(["N", "spawn", "target"])["any_rate"].nunique()
        out.append(f"non-comm arm `{arm}` invariant to deployment radius: {bool((n_unique == 1).all())}")
    same = all(
        (paired[(n, sp, tg, "inf")]["ippo_any"].values == paired[(n, sp, tg, r)]["ippo_any"].values).all()
        for n, sp, tg, r in itertools.product(NS, SPAWNS, TARGETS, ["250", "0"])
    )
    out.append(f"episode tasks paired across the radius axis (identical seeds): {same}")
    return out


# ----------------------------------------------------------------- testing
def mcnemar(paired: dict, keys, a: str, b: str):
    """Exact paired test of `b` against `a`, pooled over the given scenarios."""
    n_a_only = n_b_only = n_a = n_b = total = 0
    for k in keys:
        d = paired[k]
        va, vb = d[f"{a}_any"].values, d[f"{b}_any"].values
        n_a_only += int(((va == 1) & (vb == 0)).sum())
        n_b_only += int(((va == 0) & (vb == 1)).sum())
        n_a += int(va.sum())
        n_b += int(vb.sum())
        total += len(va)
    disc = n_a_only + n_b_only
    p = binomtest(n_b_only, disc, 0.5).pvalue if disc else 1.0
    return dict(base=n_a / total, variant=n_b / total, delta_pp=100 * (n_b - n_a) / total,
                wins=n_b_only, losses=n_a_only, p=p, n=total)


CONTRASTS = [
    ("Communication (IPPO)", "ippo", "ippo_comm"),
    ("Communication (MAPPO)", "mappo", "mappo_comm"),
    ("CTDE, no comm", "ippo", "mappo"),
    ("CTDE, with comm", "ippo_comm", "mappo_comm"),
]


def effects_by_spawn(paired: dict) -> pd.DataFrame:
    """Each mechanism's effect, pooled over N and target within a spawn regime."""
    rows = []
    for name, a, b in CONTRASTS:
        for sp in SPAWNS:
            keys = [(n, sp, tg, TRAIN_RADIUS) for n in NS for tg in TARGETS]
            rows.append(dict(contrast=name, base_arm=a, var_arm=b, spawn=sp, **mcnemar(paired, keys, a, b)))
    df = pd.DataFrame(rows)
    df["p_bh"] = false_discovery_control(df["p"].values, method="bh")
    return df


def effects_by_cell(paired: dict) -> pd.DataFrame:
    """Per-cell effects, one row per (contrast, N, spawn, target)."""
    rows = []
    for name, a, b in CONTRASTS:
        for n, sp, tg in itertools.product(NS, SPAWNS, TARGETS):
            rows.append(dict(contrast=name, N=n, spawn=sp, target=tg,
                             **mcnemar(paired, [(n, sp, tg, TRAIN_RADIUS)], a, b)))
    df = pd.DataFrame(rows)
    df["p_bh"] = false_discovery_control(df["p"].values, method="bh")
    return df


def radius_effects(paired: dict) -> pd.DataFrame:
    """Effect of cutting the comms link at deployment (radius inf -> 0)."""
    rows = []
    for arm in ["ippo_comm", "mappo_comm"]:
        for n in NS:
            n_lose = n_win = 0
            for sp, tg in itertools.product(SPAWNS, TARGETS):
                va = paired[(n, sp, tg, "inf")][f"{arm}_any"].values
                vb = paired[(n, sp, tg, "0")][f"{arm}_any"].values
                n_lose += int(((va == 1) & (vb == 0)).sum())
                n_win += int(((va == 0) & (vb == 1)).sum())
            p = binomtest(n_win, n_win + n_lose, 0.5).pvalue if (n_win + n_lose) else 1.0
            rows.append(dict(arm=arm, N=n, lost=n_lose, gained=n_win, p=p))
    return pd.DataFrame(rows)


BUDGETS = [100, 250, 500, 700, 1000, 1500, 2000, 3600]


def load_episodes(sweep: Path) -> dict:
    """Episode-level efficiency records, collapsed from the per-agent CSVs."""
    out = {}
    for n, sp, tg in itertools.product(NS, SPAWNS, TARGETS):
        d = pd.read_csv(sweep / f"{sp}__{tg}__radius-{TRAIN_RADIUS}" / f"N{n}" / "per_episode.csv")
        out[(n, sp, tg)] = d.groupby(["mode", "episode"]).agg(
            solved=("success", "any"),
            t_first=("steps_to_success", "min"),   # time to FIRST arrival
            swarm_path=("path_len", "sum"),        # metres flown by the whole swarm
        ).reset_index()
    return out


def paired_efficiency(eps: dict, keys, a: str, b: str, col: str):
    """Wilcoxon signed-rank on `col`, restricted to episodes BOTH arms solved.

    Conditioning on each arm's own successes is not a valid comparison: a weaker
    arm drops its slow wins and its conditional median improves for free.
    """
    xa, xb = [], []
    for k in keys:
        g = eps[k]
        A = g[g["mode"] == a].set_index("episode")
        B = g[g["mode"] == b].set_index("episode")
        idx = [i for i in A.index.intersection(B.index) if A.loc[i, "solved"] and B.loc[i, "solved"]]
        xa += list(A.loc[idx, col].values)
        xb += list(B.loc[idx, col].values)
    xa, xb = np.asarray(xa, float), np.asarray(xb, float)
    if len(xa) < 5:
        return None
    try:
        p = float(wilcoxon(xa, xb).pvalue)
    except ValueError:          # all differences zero
        p = 1.0
    return dict(n=len(xa), med_a=np.median(xa), med_b=np.median(xb),
                delta=np.median(xb - xa), p=p)


def efficiency_table(eps: dict) -> pd.DataFrame:
    rows = []
    for col in ["t_first", "swarm_path"]:
        for arm in ["ippo_comm", "mappo", "mappo_comm"]:
            for scope, keys in [("all cells", [(n, sp, tg) for n in NS for sp in SPAWNS for tg in TARGETS])] + \
                               [(sp, [(n, sp, tg) for n in NS for tg in TARGETS]) for sp in SPAWNS]:
                r = paired_efficiency(eps, keys, "ippo", arm, col)
                if r:
                    rows.append(dict(metric=col, arm=arm, scope=scope, **r))
    df = pd.DataFrame(rows)
    df["p_bh"] = false_discovery_control(df["p"].values, method="bh")
    return df


def budget_curves(eps: dict) -> pd.DataFrame:
    rows = []
    for n, arm in itertools.product(NS, ARMS):
        for bud in BUDGETS:
            hit = tot = 0
            for sp, tg in itertools.product(SPAWNS, TARGETS):
                g = eps[(n, sp, tg)]
                A = g[g["mode"] == arm]
                hit += int(((A.t_first <= bud) & A.solved).sum())
                tot += len(A)
            rows.append(dict(N=n, arm=arm, budget=bud, rate=hit / tot))
    return pd.DataFrame(rows)


def vs_random(paired: dict, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n, arm in itertools.product(NS, POLICY_ARMS):
        keys = [(n, sp, tg, TRAIN_RADIUS) for sp, tg in itertools.product(SPAWNS, TARGETS)]
        m = mcnemar(paired, keys, "random", arm)
        rows.append(dict(N=n, arm=arm, **m))
    return pd.DataFrame(rows)


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


# ----------------------------------------------------------------- figures
def _style(ax, *, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK_2)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.7, linestyle="-", zorder=0)
    ax.set_axisbelow(True)


def _fig(*a, **kw):
    fig, ax = plt.subplots(*a, **kw)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def fig_effects(eff: pd.DataFrame, out: Path):
    """Headline: signed effect of each mechanism, by spawn regime."""
    fig, axes = _fig(1, 2, figsize=(11, 4.2), sharex=True)
    panels = [
        ("Adding communication", ["Communication (IPPO)", "Communication (MAPPO)"]),
        ("Adding a centralized critic (CTDE)", ["CTDE, no comm", "CTDE, with comm"]),
    ]
    for ax, (title, contrasts) in zip(axes, panels):
        labels, vals, ps = [], [], []
        for c in contrasts:
            for sp in SPAWNS:
                r = eff[(eff.contrast == c) & (eff.spawn == sp)].iloc[0]
                labels.append(f"{sp}   ·  {c.split('(')[-1].rstrip(')') if '(' in c else c.split(', ')[-1]}")
                vals.append(r.delta_pp)
                ps.append(r.p_bh)
        y = np.arange(len(vals))[::-1]
        colors = [POS if v > 0 else NEG for v in vals]
        ax.barh(y, vals, height=0.62, color=colors, zorder=3)
        ax.axvline(0, color=AXIS, linewidth=1.0, zorder=4)
        for yi, v, p in zip(y, vals, ps):
            off = 0.6 if v >= 0 else -0.6
            ax.text(v + off, yi, f"{v:+.1f} pp  {stars(p)}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=8.5, color=INK_2)
        ax.set_yticks(y, labels, fontsize=8.5)
        ax.set_title(title, fontsize=10.5, color=INK, pad=10, loc="left", fontweight="semibold")
        ax.set_xlim(-21, 21)
        _style(ax, grid_axis="x")
    axes[0].set_xlabel("change in episode success rate (percentage points)", fontsize=9, color=INK_2)
    fig.legend(handles=[Patch(facecolor=POS, label="mechanism helps"),
                        Patch(facecolor=NEG, label="mechanism hurts")],
               loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Coordination pays off only when agents start co-located",
                 fontsize=13, color=INK, x=0.008, ha="left", y=1.02, fontweight="semibold")
    fig.text(0.008, 0.945, "Paired McNemar over 400 episodes per bar (both swarm sizes, both target modes); "
                           "significance after Benjamini–Hochberg correction.",
             fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    fig.savefig(out / "fig1_mechanism_effects.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def fig_success_matrix(df: pd.DataFrame, out: Path):
    """Absolute success rate for every arm across the 6 deployment cells."""
    fig, axes = _fig(2, 1, figsize=(11, 7.4), sharex=True)
    cells = [(sp, tg) for sp in SPAWNS for tg in TARGETS]
    x = np.arange(len(cells))
    w = 0.2
    for ax, n in zip(axes, NS):
        for i, arm in enumerate(POLICY_ARMS):
            vals = [df[(df.N == n) & (df.spawn == sp) & (df.target == tg)
                       & (df.radius == TRAIN_RADIUS) & (df.arm == arm)].any_rate.iloc[0]
                    for sp, tg in cells]
            pos = x + (i - 1.5) * (w + 0.012)
            ax.bar(pos, vals, width=w, color=ARM_COLOR[arm], zorder=3,
                   label=ARM_LABEL[arm] if n == 2 else None)
            for px, v in zip(pos, vals):
                # surface bbox keeps the value legible where the random-floor rule crosses it
                ax.text(px, v + 0.018, f"{v:.2f}", ha="center", fontsize=7, color=INK_2, zorder=6,
                        bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.8))
        rnd = [df[(df.N == n) & (df.spawn == sp) & (df.target == tg)
                  & (df.radius == TRAIN_RADIUS) & (df.arm == "random")].any_rate.iloc[0]
               for sp, tg in cells]
        for xi, v in zip(x, rnd):
            ax.plot([xi - 0.44, xi + 0.44], [v, v], color=MUTED, linewidth=1.6,
                    zorder=5, label="random policy" if (n == 2 and xi == 0) else None)
        # mark the one cell the policies were actually trained on
        ax.axvspan(-0.5, 0.5, color="#2a78d6", alpha=0.05, zorder=1)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel(f"N = {n}\nepisode success rate", fontsize=9.5, color=INK)
        _style(ax)
    axes[1].set_xticks(x, [f"{sp}\n{TARGET_LABEL[tg]}" for sp, tg in cells], fontsize=9)
    axes[0].text(0, 1.06, "trained here", fontsize=8, color="#2a78d6", ha="center", style="italic")
    fig.legend(loc="lower center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.035))
    fig.suptitle("Plain IPPO leads in-distribution; every arm collapses toward the random floor off it",
                 fontsize=13, color=INK, x=0.008, ha="left", y=1.0, fontweight="semibold")
    fig.text(0.008, 0.955, "Deployment comms radius = inf. Grey rule = random-policy floor for that cell. "
                           "100 paired episodes per bar.", fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    fig.savefig(out / "fig2_success_matrix.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def fig_redundancy(df: pd.DataFrame, cell_eff: pd.DataFrame, out: Path):
    """Mechanism: coordination helps exactly where the swarm's search overlaps."""
    fig, ax = _fig(figsize=(7.6, 5.2))
    # these are contrasts, not arms, so they take their own two slots -- blue and
    # orange, the pair with the widest all-pairs separation and 3:1 contrast on both.
    series = [("Communication (IPPO)", SERIES[0], "o"), ("CTDE, no comm", SERIES[2], "s")]
    for name, color, marker in series:
        sub = cell_eff[cell_eff.contrast == name]
        xs, ys = [], []
        for _, row in sub.iterrows():
            cov = df[(df.N == row.N) & (df.spawn == row.spawn) & (df.target == row.target)
                     & (df.radius == TRAIN_RADIUS) & (df.arm == "ippo")].cov_red.iloc[0]
            xs.append(cov)
            ys.append(row.delta_pp)
        r = np.corrcoef(xs, ys)[0, 1] if xs else float("nan")
        ax.scatter(xs, ys, s=64, color=color, marker=marker, zorder=3,
                   edgecolor=SURFACE, linewidth=1.6, label=f"{name}   (r = {r:+.2f})")
        m, b = np.polyfit(xs, ys, 1)
        gx = np.linspace(min(xs), max(xs), 10)
        ax.plot(gx, m * gx + b, color=color, linewidth=1.6, alpha=0.5, zorder=2)
    ax.axhline(0, color=AXIS, linewidth=1.0, zorder=4)
    ax.set_xlim(0.36, 1.05)
    ax.set_xlabel("coverage redundancy of the baseline swarm\n"
                  "(1.0 = agents search disjoint water · low = agents re-sweep each other's water)",
                  fontsize=9, color=INK_2)
    ax.set_ylabel("effect of the coordination mechanism\n(percentage points of success)",
                  fontsize=9, color=INK_2)
    ax.annotate("co-located spawns\n(redundant search)", xy=(0.395, 12.4), fontsize=8.5,
                color=MUTED, ha="left", style="italic")
    ax.annotate("dispersed spawns\n(already disjoint)", xy=(0.88, -13.5), fontsize=8.5,
                color=MUTED, ha="right", style="italic")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    _style(ax, grid_axis="both")
    fig.suptitle("Coordination buys success only where search overlaps",
                 fontsize=13, color=INK, x=0.02, ha="left", y=1.0, fontweight="semibold")
    fig.text(0.02, 0.945, "One point per deployment cell (2 swarm sizes x 3 spawn modes x 2 target modes).",
             fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "fig3_redundancy_mechanism.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def fig_radius(df: pd.DataFrame, out: Path):
    """What happens when the comms link is restricted at deployment."""
    fig, axes = _fig(1, 2, figsize=(10.4, 4.4), sharey=True)
    xs = np.arange(3)
    for ax, n in zip(axes, NS):
        for i, arm in enumerate(["ippo_comm", "mappo_comm"]):
            color = ARM_COLOR[arm]
            for sp, tg in itertools.product(SPAWNS, TARGETS):
                ys = [df[(df.N == n) & (df.spawn == sp) & (df.target == tg)
                         & (df.radius == r) & (df.arm == arm)].any_rate.iloc[0] for r in RADII]
                ax.plot(xs, ys, color=color, linewidth=0.9, alpha=0.22, zorder=2)
            mean = [df[(df.N == n) & (df.radius == r) & (df.arm == arm)].any_rate.mean() for r in RADII]
            ax.plot(xs, mean, color=color, linewidth=2.4, marker="o", markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=4,
                    label=ARM_LABEL[arm] if n == 2 else None)
            ax.text(2.06, mean[-1], f"{mean[-1]:.2f}", fontsize=8.5, color=color, va="center")
            ax.text(-0.06, mean[0], f"{mean[0]:.2f}", fontsize=8.5, color=color, va="center", ha="right")
        ax.set_xticks(xs, ["inf\n(as trained)", "250 m", "0 m\n(link cut)"], fontsize=9)
        ax.set_xlim(-0.35, 2.35)
        ax.set_title(f"N = {n}", fontsize=10.5, color=INK, loc="left", fontweight="semibold")
        _style(ax)
    axes[0].set_ylabel("episode success rate", fontsize=9.5, color=INK_2)
    axes[0].set_ylim(0, 1.0)
    fig.legend(loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Cutting the comms link at deployment: the effect flips sign with algorithm and swarm size",
                 fontsize=13, color=INK, x=0.008, ha="left", y=1.02, fontweight="semibold")
    fig.text(0.008, 0.945, "Faint lines = individual deployment cells; bold = mean over the six cells. "
                           "All policies were trained with an unlimited link.",
             fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.91))
    fig.savefig(out / "fig4_comms_range.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def fig_budget(curves: pd.DataFrame, out: Path):
    """Success within a step budget -- the unconditional speed+success summary."""
    fig, axes = _fig(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, n in zip(axes, NS):
        ends = []
        for arm in ARMS:
            sub = curves[(curves.N == n) & (curves.arm == arm)].sort_values("budget")
            is_rnd = arm == "random"
            color = MUTED if is_rnd else ARM_COLOR[arm]
            ax.plot(sub.budget, sub.rate, color=color,
                    linewidth=1.6 if is_rnd else 2.2,
                    marker="" if is_rnd else "o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3,
                    label=ARM_LABEL[arm] if n == 2 else None)
            ends.append([sub.iloc[-1].rate, sub.iloc[-1].rate, color])
        # push apart endpoint labels that would otherwise overprint each other
        ends.sort(key=lambda e: e[0])
        for i in range(1, len(ends)):
            ends[i][1] = max(ends[i][1], ends[i - 1][1] + 0.022)
        for rate, y, color in ends:
            ax.text(3600 * 1.07, y, f"{rate:.2f}", fontsize=8, color=color, va="center")
        ax.set_xscale("log")
        ax.set_xticks(BUDGETS, [str(b) for b in BUDGETS], fontsize=8)
        ax.minorticks_off()
        ax.set_xlim(90, 5200)
        ax.set_xlabel("step budget", fontsize=9, color=INK_2)
        ax.set_title(f"N = {n}", fontsize=10.5, color=INK, loc="left", fontweight="semibold")
        _style(ax, grid_axis="both")
    axes[0].set_ylabel("fraction of episodes solved within budget", fontsize=9.5, color=INK_2)
    axes[0].set_ylim(0, 0.75)
    fig.legend(loc="lower center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("The arms differ in whether the target is found, not in how fast",
                 fontsize=13, color=INK, x=0.008, ha="left", y=1.02, fontweight="semibold")
    fig.text(0.008, 0.945, "Pooled over the six deployment cells. Unlike a median time-to-success, this "
                           "counts unsolved episodes, so it is not distorted by survivorship.",
             fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.91))
    fig.savefig(out / "fig6_success_at_budget.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def fig_heatmap(cell_eff: pd.DataFrame, out: Path):
    """Per-cell effect table, as a diverging heatmap."""
    rows = [(n, sp, tg) for n in NS for sp in SPAWNS for tg in TARGETS]
    cols = [c[0] for c in CONTRASTS]
    M = np.array([[cell_eff[(cell_eff.contrast == c) & (cell_eff.N == n)
                            & (cell_eff.spawn == sp) & (cell_eff.target == tg)].delta_pp.iloc[0]
                   for c in cols] for n, sp, tg in rows])
    P = np.array([[cell_eff[(cell_eff.contrast == c) & (cell_eff.N == n)
                            & (cell_eff.spawn == sp) & (cell_eff.target == tg)].p_bh.iloc[0]
                   for c in cols] for n, sp, tg in rows])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("bwr_", [NEG, "#f0efec", POS])
    fig, ax = _fig(figsize=(8.6, 6.6))
    lim = np.abs(M).max()
    ax.imshow(M, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            sig = "*" if P[i, j] < 0.05 else ""
            ax.text(j, i, f"{M[i, j]:+.0f}{sig}", ha="center", va="center",
                    fontsize=9.5, color=INK if abs(M[i, j]) < 0.55 * lim else SURFACE)
    ax.set_xticks(range(len(cols)), [c.replace(" (", "\n(").replace(", ", "\n") for c in cols], fontsize=9)
    ax.set_yticks(range(len(rows)), [f"N={n}  ·  {sp}  ·  {tg}" for n, sp, tg in rows], fontsize=9)
    ax.tick_params(colors=MUTED, length=0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK_2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2.5)
    ax.tick_params(which="minor", length=0)
    fig.suptitle("Effect of each mechanism, cell by cell (percentage points)",
                 fontsize=13, color=INK, x=0.02, ha="left", y=1.0, fontweight="semibold")
    fig.text(0.02, 0.95, "Blue = mechanism helps, red = hurts. * marks BH-corrected p < 0.05. The origin rows are the "
                         "only ones where both single-mechanism columns are positive together.",
             fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "fig5_effect_heatmap.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------ report
def md_table(df: pd.DataFrame, cols: dict[str, str], fmt: dict | None = None) -> str:
    fmt = fmt or {}
    head = "| " + " | ".join(cols.values()) + " |"
    rule = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [head, rule]
    for _, r in df.iterrows():
        cells = []
        for k in cols:
            v = r[k]
            cells.append(fmt[k](v) if k in fmt else (f"{v:.3f}" if isinstance(v, float) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(sweep: Path, out: Path, df, paired, eff, cell_eff, rad, rnd, invariants, eff_tab):
    meta = json.loads((sweep / "sweep_summary.json").read_text())
    pp = lambda v: f"{v:+.1f}"
    p_f = lambda v: f"{v:.4f}" if v >= 1e-4 else f"{v:.1e}"

    id_cell = df[(df.spawn == TRAIN_SPAWN) & (df.target == TRAIN_TARGET) & (df.radius == TRAIN_RADIUS)]

    n2_gen = json.loads((sweep / f"{TRAIN_SPAWN}__{TRAIN_TARGET}__radius-{TRAIN_RADIUS}"
                         / "N2" / "summary.json").read_text())["generated"]
    doc = f"""# Scenario sweep — results and analysis

Source: `stats/out/sweep_full_100/` · {meta['episodes']} paired episodes per cell · greedy
decoding · evaluated on the held-out `data/oceananigans/buoyancy_active/test` split.

The two swarm sizes were swept in separate passes on {n2_gen[:10]} ({meta['elapsed_min']/60:.1f} h
wall-clock for the N=4 pass alone). Note that the top-level `sweep_summary.json` records only
the N=4 pass, so its `n_agents: [4]` is *not* the scope of this report — the per-cell summaries
underneath it cover both sizes, and those are what is read here.

## 1. What was swept

Eight policies form a clean 2x2x2 ablation. All were trained identically —
10M agent-steps for N=2, 20M for N=4 (i.e. the same 5M joint environment steps),
`spawn_mode=random`, `target_mode=random`, `comms_radius=inf`, `shared_success_bonus=True`,
`beta_difference=0`, `lambda_separation=0` — differing **only** in the two ablated factors:

| factor | levels |
|---|---|
| critic | decentralized (IPPO) · centralized on the agentized global state (MAPPO) |
| communication | off (obs = 9 + 5k) · on (obs = 9 + 5(N−1) + 5k, neighbours sorted nearest-first) |
| swarm size | N = 2 · N = 4 |

Each was then deployed across 18 scenarios: 3 spawn modes x 2 target modes x 3 comms radii,
plus a random-policy floor. Episode success is `success_any` (the episode ends when any agent
reaches the zone).

> **The single most important caveat.** Every policy was trained at
> `spawn=random, target=random, radius=inf`. That is **1 of the 18 deployment cells**.
> The other 17 measure *out-of-distribution generalization*, not trained competence.
> In particular the `origin` and `max_dist` spawns, the `tail` target, and both restricted
> radii were never seen during training. This does not invalidate the comparisons — all arms
> face the identical shift on identical episode seeds — but it does mean a positive effect
> off-distribution is evidence about **robustness**, not about what the mechanism learned to do.

Structural invariants verified before analysis:

{chr(10).join(f'- {s}' for s in invariants)}

The first three confirm the non-communicating arms are genuinely untouched by the radius axis
(so their three radius rows are one measurement, not three), and the last confirms every arm in
every radius condition saw the same 100 tasks — which is what licenses the paired tests below.

## 2. Headline: the two remembered conclusions, re-checked

**Conclusion A — "communication only helps when agents start from the same initial position."**
**Confirmed for IPPO, and it is stronger than "only helps": off co-located spawns it actively hurts.**
**Rejected for MAPPO**, where communication is neutral-to-harmful everywhere including at the origin.

**Conclusion B — "MAPPO CTDE only helps in some specific cases."**
**Confirmed, and the specific case is now identified: it is the same one.** The centralized critic
pays off at co-located spawns and costs elsewhere.

Effects pooled over both swarm sizes and both target modes (400 paired episodes per row),
exact McNemar, Benjamini–Hochberg corrected across the 12 tests:

{md_table(eff, {'contrast': 'mechanism', 'spawn': 'spawn regime', 'base': 'baseline', 'variant': 'with mechanism', 'delta_pp': 'Δ pp', 'wins': 'wins', 'losses': 'losses', 'p': 'p', 'p_bh': 'p (BH)'}, {'delta_pp': pp, 'p': p_f, 'p_bh': p_f})}

Reading the table:

- **Communication on IPPO** — helps at `origin` (**+6.8 pp**, BH p = 0.009), hurts at `random`
  (**−7.0 pp**, BH p = 0.004), indistinguishable at `max_dist`. The sign flips with the spawn
  regime; that flip is the whole result.
- **Communication on MAPPO** — never helps. At `origin`, where it helps IPPO, it is the single
  largest negative effect in the sweep (**−10.8 pp**, BH p = 3e-5).
- **CTDE without communication** — helps at `origin` (**+8.0 pp**), hurts at `random` (**−8.8 pp**).
  Same flip, same regime.
- **CTDE with communication** — negative everywhere, significantly so in all three regimes.
  The two mechanisms do not compose: each alone buys something at the origin, both together
  lose more than either gains.

![mechanism effects](fig1_mechanism_effects.png)

## 3. Why the origin spawn is different

The env reports `coverage_redundancy` — unique visited voxels divided by the sum of per-agent
visited voxels. 1.0 means the agents swept disjoint water; 1/N means they all swept the same water.
Measured on the IPPO baseline:

| spawn mode | redundancy N=2 | redundancy N=4 | mean NN distance N=2 | N=4 |
|---|---|---|---|---|
| `random` | 0.976 | 0.950 | 516 m | 305 m |
| `max_dist` | 1.000 | 0.967 | 676 m | 368 m |
| `origin` | **0.652** | **0.416** | **87 m** | **32 m** |

This is the mechanism, and it is not subtle. At `random` and `max_dist` spawns the swarm is
*already* searching disjoint water — the initial condition does the division of labour for free,
and there is nothing left for coordination to buy. Only at `origin`, where all agents start
stacked on the same point and stay within ~30–90 m of each other, does redundant search exist as
a problem. Pooled by spawn regime, both mechanisms pay off there and nowhere else. (At the level
of individual cells there is one exception worth naming honestly: communication gives IPPO
**+12 pp** at `N=4 / max_dist / tail`, the sweep's largest single-cell communication gain, with
raw p = 0.012. It does not survive multiplicity correction — BH p = 0.063 — so it is suggestive
rather than established, and it is the reason the `max_dist` pooled effect for IPPO is positive
though not significant.)

Across the 12 deployment cells the correlation between baseline redundancy and the benefit of
coordination is strongly negative for both mechanisms:

![redundancy mechanism](fig3_redundancy_mechanism.png)

The practical reading: **for this task, the spawn geometry is a substitute for coordination.**
If you can choose where the vehicles enter the water, spreading them out buys more than
either communication or a centralized critic — and it is free.

## 4. Absolute performance

In-distribution cell (`spawn=random`, `target=random`, `radius=inf`) — the only one the
policies were trained for:

{md_table(id_cell.sort_values(['N']), {'N': 'N', 'arm': 'arm', 'any_rate': 'success', 'any_lo': 'CI lo', 'any_hi': 'CI hi', 't_med': 'median steps', 'swarm_path': 'swarm path (m)', 'spl': 'SPL'}, {'any_rate': lambda v: f'{v:.2f}', 'any_lo': lambda v: f'{v:.2f}', 'any_hi': lambda v: f'{v:.2f}', 't_med': lambda v: f'{v:.0f}', 'swarm_path': lambda v: f'{v:.0f}'})}

> The `median steps` column is **conditional on each arm's own successes** and must not be read
> as a speed comparison. A weaker arm drops the episodes it could not solve — which are
> disproportionately the slow ones — and its conditional median improves for free. Concretely, in
> this cell MAPPO looks 45 steps faster than IPPO (286 vs 330); restricted to the 84 episodes both
> solved, IPPO is 30 steps *faster* (256 vs 286). The 14 episodes only IPPO solved had a median
> time of 734 steps. §7 does this properly.

Two things stand out. First, **plain IPPO at N=2 is the best policy in the sweep (0.98)**, and
every added mechanism costs it — the ablation's honest in-distribution answer is that neither
communication nor CTDE is worth its price where the policy was actually trained. Second, at
**N=4 the random policy already reaches 0.83** in this cell: with four agents and a random target,
the task is nearly saturated and the cell has almost no discriminative power. The N=4 comparisons
are only informative on the harder cells (`tail` target, `origin` spawn).

All arms across all six cells, with the random floor marked:

![success matrix](fig2_success_matrix.png)

Margin over the random floor, pooled over the six cells:

{md_table(rnd, {'N': 'N', 'arm': 'arm', 'base': 'random', 'variant': 'arm', 'delta_pp': 'Δ pp', 'wins': 'wins', 'losses': 'losses', 'p': 'p'}, {'delta_pp': pp, 'p': p_f})}

Every learned policy beats the random floor decisively, but the margin is modest — +19 to +27 pp
at N=2, and only +8 to +18 pp at N=4. **The learned advantage shrinks as the swarm grows**, which
is the expected consequence of `success_any`: more agents means more independent lottery tickets,
so a random swarm improves with N faster than a trained one does.

## 5. Per-cell detail

![effect heatmap](fig5_effect_heatmap.png)

{md_table(cell_eff, {'contrast': 'mechanism', 'N': 'N', 'spawn': 'spawn', 'target': 'target', 'base': 'baseline', 'variant': 'variant', 'delta_pp': 'Δ pp', 'p': 'p', 'p_bh': 'p (BH)'}, {'delta_pp': pp, 'p': p_f, 'p_bh': p_f})}

At the individual-cell level almost nothing survives correction — the per-cell tests carry only
100 episodes each, and the pooled tests in §2 are where the power is. The cell table is here to
show that the pooled effects are not driven by one outlier cell: across both N and both target
modes, all four origin rows are positive for `Communication (IPPO)` (+8, +10, +2, +7) and for
`CTDE, no comm` (+12, +11, +5, +4), while every random-spawn row is negative or zero.

## 6. Deployment-time comms restriction

The comm-trained policies were all trained with an unlimited link, then deployed at 250 m and
with the link cut entirely (0 m — neighbour slots zeroed, `in_range=0`):

{md_table(rad, {'arm': 'arm', 'N': 'N', 'lost': 'episodes lost', 'gained': 'episodes gained', 'p': 'p'}, {'p': p_f})}

![comms range](fig4_comms_range.png)

- **`ippo_comm` at N=4 degrades sharply** when the link is cut (144 episodes lost vs 29 gained,
  p = 2e-19; e.g. `max_dist`/`random` falls 0.90 → 0.65). This policy genuinely uses the channel.
- **`ippo_comm` at N=2 is unaffected** (p = 0.28) — with a single neighbour there is little to say.
- **`mappo_comm` at N=2 collapses** (162 lost vs 22, p = 2e-27): `random`/`tail` falls 0.62 → 0.28,
  `origin`/`tail` 0.22 → 0.07. It is the most link-dependent policy in the sweep and also one of
  the weakest — a bad combination.
- **`mappo_comm` at N=4 *improves* when the link is cut** (88 gained vs 27 lost, p = 1e-8;
  `random`/`tail` rises 0.56 → 0.79). This is the sweep's most surprising result: the channel is
  a net **liability** for that policy, and zeroing it at deployment is a free improvement. The
  natural reading is that the N=4 MAPPO+comm run overfit to neighbour features that do not
  generalize to the test split, and the centralized critic — which already sees every agent's
  state — gave the actor no pressure to use them well.

## 7. Efficiency: is any arm faster or cheaper?

Success rate is not the only thing an operator cares about — a policy that arrives sooner, or
flies fewer total metres to get there, is worth something even at equal success. Two measures:
**time to first arrival** (steps until any agent reaches the zone) and **swarm path length**
(metres flown by every agent summed, i.e. the battery bill for the whole team).

Both are only comparable on episodes that **both** arms solved, for the survivorship reason in
the §4 note. Paired Wilcoxon signed-rank against plain IPPO; negative Δ means the variant is
faster / cheaper:

{md_table(eff_tab, {'metric': 'measure', 'arm': 'arm', 'scope': 'scope', 'n': 'n episodes', 'med_a': 'IPPO', 'med_b': 'variant', 'delta': 'Δ median', 'p': 'p', 'p_bh': 'p (BH)'}, {'med_a': lambda v: f'{v:.0f}', 'med_b': lambda v: f'{v:.0f}', 'delta': lambda v: f'{v:+.0f}', 'p': p_f, 'p_bh': p_f})}

**Nothing is faster or cheaper than plain IPPO anywhere.** Out of 24 tests not one shows a
significant improvement, and the single significant result runs the other way: MAPPO flies
**+63 m more** per episode than IPPO at `max_dist` spawn (raw p = 0.022, though it does not
survive correction). The largest hint in IPPO's disfavour is communication at the origin spawn
(−9 steps, raw p = 0.053) — consistent with the §2 story, but not established.

The unconditional view confirms it. Success-within-a-budget counts unsolved episodes as failures
at every budget, so it cannot be gamed by survivorship:

![success at budget](fig6_success_at_budget.png)

The curves are near-parallel: at N=2 plain IPPO is on top at **every** budget from 100 to 3600
steps, and at N=4 IPPO+comm is on top at every budget. No arm buys early speed at the cost of
eventual success, or vice versa — the winner at 100 steps is the winner at 3600.

**Conclusion: efficiency adds no new information to this sweep.** The arms differ in *whether*
they find the target, not in how quickly or how far they fly to do it. That is itself a useful
negative result for the thesis — it rules out the "coordination doesn't raise success but lowers
cost" defence of MAPPO and of communication, which would otherwise be the obvious rebuttal to §2.

## 8. What to say in the meeting

1. **Both remembered conclusions hold, with one correction.** Communication helping only at
   co-located spawns is confirmed *for IPPO*; for MAPPO communication never helps. The "specific
   cases" where CTDE helps are now named: co-located spawns, and only without communication.
2. **There is a single mechanism behind both.** Coordination pays exactly where coverage
   redundancy is high, i.e. where agents would otherwise re-sweep each other's water. Everywhere
   else the spawn geometry has already solved the problem. This is a cleaner story than two
   independent findings and it is directly measurable (§3).
3. **The honest in-distribution result is negative.** Where the policies were trained, plain
   IPPO wins and every mechanism costs. The positive results all live off-distribution.
4. **The experiment cannot currently separate "mechanism helps" from "mechanism generalizes."**
   Every positive effect is measured under distribution shift. The clean fix is to retrain with
   `spawn_mode=origin` (and/or `target_mode=tail`) so the regime where coordination matters is
   also the regime that was trained for. That is the obvious next run and it is cheap — the
   flags already exist in `Args`.
5. **Two results deserve their own slide:** `mappo_comm` at N=4 getting *better* when its comms
   link is cut, and the random policy reaching 0.83 at N=4 in-distribution. The first is an
   overfitting diagnosis; the second means that cell should be dropped from future sweeps.

### Caveats to state up front

- Success is `success_any`; a swarm gets credit when one agent arrives. `success_all` is ~0
  everywhere and carries no signal.
- 100 episodes per cell gives roughly ±10 pp of Wilson width on a single rate. Only the pooled
  tests in §2, §4 and §6 are adequately powered; per-cell numbers are indicative.
- One seed per configuration. There is no across-seed variance estimate, so a 5–10 pp difference
  between two arms in one cell could be seed noise. The paired design controls the *task*
  variance, not the *training* variance.
- `origin` and `max_dist` spawn, `tail` target, and both restricted radii are all
  out-of-distribution (§1).
"""
    (out / "README.md").write_text(doc)
    return out / "README.md"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", type=Path, default=Path("stats/out/sweep_full_100"))
    ap.add_argument("--out", type=Path, default=Path("stats/sweep_full_100_report"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df, paired = load(args.sweep)
    invariants = check_invariants(df, paired)
    eff = effects_by_spawn(paired)
    cell_eff = effects_by_cell(paired)
    rad = radius_effects(paired)
    rnd = vs_random(paired, df)
    eps = load_episodes(args.sweep)
    eff_tab = efficiency_table(eps)
    curves = budget_curves(eps)

    df.to_csv(args.out / "cells.csv", index=False)
    eff.to_csv(args.out / "effects_pooled.csv", index=False)
    cell_eff.to_csv(args.out / "effects_per_cell.csv", index=False)
    eff_tab.to_csv(args.out / "efficiency.csv", index=False)
    curves.to_csv(args.out / "success_at_budget.csv", index=False)

    fig_effects(eff, args.out)
    fig_success_matrix(df, args.out)
    fig_redundancy(df, cell_eff, args.out)
    fig_radius(df, args.out)
    fig_heatmap(cell_eff, args.out)
    fig_budget(curves, args.out)

    path = write_report(args.sweep, args.out, df, paired, eff, cell_eff, rad, rnd, invariants, eff_tab)
    print(f"wrote {path}")
    for p in sorted(args.out.glob("*.png")):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()

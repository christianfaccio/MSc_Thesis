#!/usr/bin/env python
"""Thesis-ready analysis of the deployment sweeps.

Reads the sweep output directories and emits, for every result, a LaTeX table
(\\input-able, booktabs, matching the idiom already used in thesis/ch3.tex), a
vector PDF figure, and the tidy CSV the two were computed from. Nothing has to
be transcribed by hand.

It supersedes analyze_sweep_full.py, which assumed ONE directory holding both
swarm sizes and hard-coded a narrative written against the July data. Here the
swarm sizes live in separate directories, every number in the output is
computed, and no conclusion is baked into the text.

Inputs
------
success_any sweeps (stats/scenario_sweep.py), one directory per swarm size:
    <dir>/<spawn>__<target>__radius-<r>/N<n>/{summary.json,paired_episodes.csv,per_episode.csv}
    3 spawn x 2 target x 3 radius = 18 scenarios.

success_all sweeps (stats/success_all_sweep.py), one directory per swarm size:
    <dir>/{arrival_times.csv,episodes.csv,arrival_by_rank.csv,scenario_summary.csv}
    2 spawn x 1 target = 2 scenarios, radius as trained.
Only read when --success-all is passed, so the success_any half can be analysed
while those sweeps are still running.

Sections
--------
1  effect of communication and of CTDE, per swarm size   -> tab1*, fig1*
2  success matrix, every arm x every deployment cell      -> tab2,  fig2
3  effect of the deployment communication range           -> tab3,  fig3
4  success within a step budget                           -> tab4,  fig4
5  time to first success (conditional AND penalized)      -> tab5*, fig5
6  arrival times of the non-first agents  [--success-all] -> tab6,  fig6
8  effect of the spawn geometry (agent initialization)    -> tab8,  fig8
9  cost of a rare target, success AND time                -> tab9,  fig9
10 effect of swarm size (scaling N)                        -> tab10, fig10

These numbers are artefact IDs, not thesis section numbers -- the chapter orders
its sections differently. 7 is deliberately unused here: fig7_training_curves is
emitted by stats/plot_training_curves.py and also lands in thesis/assets.

Usage
-----
    # success_any only, while the success_all sweeps still run
    python stats/analyze_sweeps.py \\
        --sweep 2=stats/out/sweep_full_100_2N_new \\
        --sweep 4=stats/out/sweep_full_100_4N_new \\
        --out stats/out/thesis_report

    # once the success_all sweeps are done
    python stats/analyze_sweeps.py \\
        --sweep 2=stats/out/sweep_full_100_2N_new \\
        --sweep 4=stats/out/sweep_full_100_4N_new \\
        --success-all \\
        --success-all-sweep 2=stats/out/success_all_2N \\
        --success-all-sweep 4=stats/out/success_all_4N \\
        --out stats/out/thesis_report --assets-dir thesis/assets

Two statistical conventions, applied throughout
-----------------------------------------------
CONDITIONAL TIMES ARE NOT A SPEED COMPARISON. A median taken over an arm's own
successes is survivorship-biased: a weaker arm drops the episodes it could not
solve, which are disproportionately the slow ones, and its median improves for
free. Every time-like quantity is therefore reported twice -- conditionally,
with the number of episodes it was computed from, and as the penalized

    E[min(t, T)]  =  mean over ALL episodes, unsolved charged the full budget T

which can only improve if an arm both solves more AND arrives sooner. Its
unit-free form 1 - E[min(t,T)]/T lies in [0,1] (0 = never succeeds, 1 = always
succeeds instantly) and is exactly the normalized area under the section-4
success-at-budget curve, so sections 4 and 5 are two views of one quantity.

CENSORING HERE IS ADMINISTRATIVE, NOT RANDOM. An episode ends early only when
every agent has arrived, so an agent is either observed or censored at exactly
max_steps -- never in between. Under a single common censoring time the
empirical arrival CDF is unbiased and Kaplan-Meier reduces to it exactly, so
section 6 plots the empirical CDF per arrival rank (plateau height = fraction
that ever arrived) rather than pulling in a survival-analysis dependency.
"""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import binomtest, false_discovery_control, fisher_exact

matplotlib.rcParams["pdf.fonttype"] = 42       # embed TrueType, not Type-3
matplotlib.rcParams["ps.fonttype"] = 42

SPAWNS = ["random", "origin", "max_dist"]
TARGETS = ["random", "tail"]
RADII = ["inf", "250", "0"]
ARMS = ["ippo", "ippo_comm", "mappo", "mappo_comm", "random"]
POLICY_ARMS = ARMS[:-1]

# The trained configuration -- the single in-distribution deployment cell.
TRAIN_SPAWN, TRAIN_TARGET, TRAIN_RADIUS = "random", "random", "inf"

BUDGETS = [100, 250, 500, 700, 1000, 1500, 2000, 3600]

ARM_LABEL = {"ippo": "IPPO", "ippo_comm": "IPPO + comm", "mappo": "MAPPO",
             "mappo_comm": "MAPPO + comm", "random": "random"}
SPAWN_LABEL = {"random": "random", "origin": "origin", "max_dist": "max\\_dist"}
TARGET_LABEL = {"random": "random target", "tail": "tail target"}

# Section 8 plots the spawn regimes along an axis that MEANS something rather
# than in declaration order: how dispersed the swarm is at t=0. `origin` puts
# every agent on one point (the corner of the L-shaped coast), `random` is the
# uniform draw the policies were trained on, `max_dist` spaces them evenly along
# the coastline -- 707 m apart at N=2, and the widest of the three at every N.
SPAWN_ORDER = ["origin", "random", "max_dist"]
SPAWN_PLOT = {"origin": "origin\n(all on one point)",
              "random": "random\n(uniform, as trained)",
              "max_dist": "max_dist\n(spread along coast)"}

CONTRASTS = [
    ("Communication (IPPO)", "ippo", "ippo_comm"),
    ("Communication (MAPPO)", "mappo", "mappo_comm"),
    ("CTDE, no comm", "ippo", "mappo"),
    ("CTDE, with comm", "ippo_comm", "mappo_comm"),
]

# --- validated palette (dataviz reference instance, light mode) ---
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#1baf7a", "#eb6834", "#4a3aa7"]   # blue, aqua, orange, violet
POS, NEG = "#2a78d6", "#e34948"
ARM_COLOR = dict(zip(POLICY_ARMS, SERIES))


# ------------------------------------------------------------------ loading
def load_any(sweeps: dict[int, Path]):
    """Cell-level dataframe + paired-episode frames, over every swarm size.

    `sweeps` maps swarm size -> the sweep directory for that size. Scenarios
    missing from a directory are skipped with a warning rather than fatal, so a
    partially finished sweep still analyses.
    """
    rows, paired, missing = [], {}, []
    for n, root in sweeps.items():
        for sp, tg, rd in itertools.product(SPAWNS, TARGETS, RADII):
            base = root / f"{sp}__{tg}__radius-{rd}" / f"N{n}"
            if not (base / "summary.json").exists():
                missing.append(str(base))
                continue
            d = json.loads((base / "summary.json").read_text())
            es, res = d["episode_stats"], d["results"]
            for arm in ARMS:
                if arm not in res:
                    continue
                r, e = res[arm], es[arm]
                rows.append(dict(
                    N=n, spawn=sp, target=tg, radius=rd, arm=arm,
                    any_rate=e["any_rate"], any_lo=e["any_ci"][0], any_hi=e["any_ci"][1],
                    per_agent=r["success_rate"], spl=r["spl_mean"],
                    t_med_cond=r["steps_to_success"]["median"],
                    swarm_path=e.get("swarm_path"), cov_red=e.get("coverage_redundancy"),
                    nn_dist=e.get("nn_distance"), max_steps=d.get("max_steps"),
                    episodes=d.get("episodes")))
            paired[(n, sp, tg, rd)] = pd.read_csv(base / "paired_episodes.csv")
    if missing:
        print(f"[warn] {len(missing)} scenario(s) absent; skipped:")
        for m in missing[:6]:
            print(f"         {m}")
        if len(missing) > 6:
            print(f"         ... and {len(missing) - 6} more")
    if not rows:
        raise SystemExit("no scenarios found -- check --sweep paths.")
    df = pd.DataFrame(rows)
    # A swarm size that contributed NOTHING must not be dropped silently: the run
    # would still emit a full-looking report that is quietly missing an entire
    # experimental condition.
    empty = [n for n in sweeps if n not in set(df.N)]
    if empty:
        raise SystemExit(
            "no scenarios found for swarm size(s) " + ", ".join(f"N={n}" for n in empty)
            + ".\n  Directories given:\n"
            + "\n".join(f"    N={n}: {p}{'' if p.exists() else '   <- does not exist'}"
                        for n, p in sweeps.items())
            + "\n  Drop the --sweep entry to build a report without it, or re-run that sweep.")
    return df, paired


def load_episodes(sweeps: dict[int, Path]):
    """Per-episode outcome at the trained radius: solved + time to first arrival.

    Collapsed from the per-agent CSV, so `t_first` is the minimum arrival step
    over the swarm -- the success_any clock.
    """
    out = {}
    for n, root in sweeps.items():
        for sp, tg in itertools.product(SPAWNS, TARGETS):
            f = root / f"{sp}__{tg}__radius-{TRAIN_RADIUS}" / f"N{n}" / "per_episode.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f)
            g = (d.groupby(["mode", "episode"])
                   .agg(solved=("success", "any"),
                        t_first=("steps_to_success", "min"),
                        swarm_path=("path_len", "sum"))
                   .reset_index())
            out[(n, sp, tg)] = g
    return out


def load_success_all(sweeps: dict[int, Path]) -> pd.DataFrame:
    """Per-agent arrival records from the success_all sweeps, concatenated."""
    frames = []
    for n, root in sweeps.items():
        f = root / "arrival_times.csv"
        if not f.exists():
            print(f"[warn] {f} absent -- success_all sweep for N={n} not finished; skipped.")
            continue
        d = pd.read_csv(f)
        d["N"] = n
        frames.append(d)
    if not frames:
        raise SystemExit("--success-all was passed but no arrival_times.csv was found.")
    return pd.concat(frames, ignore_index=True)


def check_invariants(df: pd.DataFrame, paired: dict) -> list[str]:
    """Structural assumptions of the sweep, verified rather than assumed."""
    out = []
    for arm in ["ippo", "mappo", "random"]:
        sub = df[df.arm == arm]
        if sub.empty:
            continue
        uniq = sub.groupby(["N", "spawn", "target"])["any_rate"].nunique()
        out.append(f"non-comm arm `{arm}` invariant to deployment radius: {bool((uniq == 1).all())}")
    keys = [(n, sp, tg) for (n, sp, tg, rd) in paired if rd == TRAIN_RADIUS]
    same = all(
        (paired[(n, sp, tg, TRAIN_RADIUS)]["ippo_any"].values
         == paired[(n, sp, tg, r)]["ippo_any"].values).all()
        for (n, sp, tg) in keys for r in ["250", "0"] if (n, sp, tg, r) in paired)
    out.append(f"episode tasks paired across the radius axis (identical seeds): {same}")
    return out


# ------------------------------------------------------------------ metrics
def mcnemar(paired: dict, keys, a: str, b: str):
    """Exact paired test of arm `b` against arm `a`, pooled over `keys`."""
    n_a_only = n_b_only = n_a = n_b = total = 0
    for k in keys:
        d = paired.get(k)
        if d is None or f"{a}_any" not in d or f"{b}_any" not in d:
            continue
        va, vb = d[f"{a}_any"].values, d[f"{b}_any"].values
        n_a_only += int(((va == 1) & (vb == 0)).sum())
        n_b_only += int(((va == 0) & (vb == 1)).sum())
        n_a += int(va.sum())
        n_b += int(vb.sum())
        total += len(va)
    if not total:
        return None
    disc = n_a_only + n_b_only
    p = binomtest(n_b_only, disc, 0.5).pvalue if disc else 1.0
    return dict(base=n_a / total, variant=n_b / total, delta_pp=100 * (n_b - n_a) / total,
                wins=n_b_only, losses=n_a_only, p=p, n=total)


def penalized_time(t: np.ndarray, solved: np.ndarray, T: float) -> float:
    """E[min(t, T)] over ALL episodes -- unsolved charged the full budget.

    Unlike a conditional median this cannot be improved by failing more: an
    episode that is never solved contributes the largest possible value.
    """
    t = np.asarray(t, float)
    solved = np.asarray(solved, bool)
    v = np.where(solved & np.isfinite(t), np.minimum(t, T), T)
    return float(v.mean())


def effects_by_spawn(paired: dict, ns: list[int]) -> pd.DataFrame:
    """Each mechanism's effect within a swarm size and spawn regime.

    Pooled over target modes only. N is NOT pooled: the point of the table is
    how the effect scales with swarm size, and N=2 and N=4 episodes cannot be
    paired anyway (reset() draws spawns before the target, so a seed gives a
    different task at a different N).
    """
    rows = []
    for name, a, b in CONTRASTS:
        for n in ns:
            for sp in SPAWNS:
                m = mcnemar(paired, [(n, sp, tg, TRAIN_RADIUS) for tg in TARGETS], a, b)
                if m:
                    rows.append(dict(contrast=name, base_arm=a, var_arm=b, N=n, spawn=sp, **m))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["p_bh"] = false_discovery_control(df["p"].values, method="bh")
    return df


def effects_by_cell(paired: dict, ns: list[int]) -> pd.DataFrame:
    """Per-cell effects, one row per (contrast, N, spawn, target)."""
    rows = []
    for name, a, b in CONTRASTS:
        for n, sp, tg in itertools.product(ns, SPAWNS, TARGETS):
            m = mcnemar(paired, [(n, sp, tg, TRAIN_RADIUS)], a, b)
            if m:
                rows.append(dict(contrast=name, N=n, spawn=sp, target=tg, **m))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["p_bh"] = false_discovery_control(df["p"].values, method="bh")
    return df


def radius_effects(paired: dict, df: pd.DataFrame, ns: list[int]) -> pd.DataFrame:
    """Effect of restricting the comms link at deployment (inf -> 250 -> 0).

    Paired within an arm: the same policy on the same episodes, with only the
    observation-side neighbour block changed.
    """
    rows = []
    for arm in ["ippo_comm", "mappo_comm"]:
        for n in ns:
            rates = {}
            for rd in RADII:
                sub = df[(df.N == n) & (df.radius == rd) & (df.arm == arm)]
                rates[rd] = sub.any_rate.mean() if not sub.empty else np.nan
            lost = won = 0
            for sp, tg in itertools.product(SPAWNS, TARGETS):
                ka, kb = (n, sp, tg, "inf"), (n, sp, tg, "0")
                if ka not in paired or kb not in paired:
                    continue
                va, vb = paired[ka][f"{arm}_any"].values, paired[kb][f"{arm}_any"].values
                lost += int(((va == 1) & (vb == 0)).sum())
                won += int(((va == 0) & (vb == 1)).sum())
            p = binomtest(won, won + lost, 0.5).pvalue if (won + lost) else 1.0
            rows.append(dict(arm=arm, N=n, rate_inf=rates["inf"], rate_250=rates["250"],
                             rate_0=rates["0"], delta_pp=100 * (rates["0"] - rates["inf"]),
                             lost=lost, gained=won, p=p))
    return pd.DataFrame(rows)


def budget_curves(eps: dict, ns: list[int]) -> pd.DataFrame:
    """Fraction of episodes solved within a step budget, pooled over cells.

    Counts unsolved episodes as failures at every budget, so unlike a
    conditional median it cannot be distorted by survivorship.
    """
    rows = []
    for n, arm in itertools.product(ns, ARMS):
        for bud in BUDGETS:
            hit = tot = 0
            for sp, tg in itertools.product(SPAWNS, TARGETS):
                g = eps.get((n, sp, tg))
                if g is None:
                    continue
                A = g[g["mode"] == arm]
                hit += int(((A.t_first <= bud) & A.solved).sum())
                tot += len(A)
            if tot:
                rows.append(dict(N=n, arm=arm, budget=bud, rate=hit / tot))
    return pd.DataFrame(rows)


def time_matrix(eps: dict, df: pd.DataFrame, ns: list[int]) -> pd.DataFrame:
    """Time to FIRST success per (N, cell, arm), conditional and penalized.

    `t_med_cond` / `t_mean_cond` are over that arm's own successes and carry
    `n_solved` so the conditioning is visible; `t_penalized` = E[min(t,T)] is
    over every episode. `time_weighted_success` = 1 - E[min(t,T)]/T is the
    unit-free form, and equals the normalized area under the budget curve.
    """
    rows = []
    for n, sp, tg in itertools.product(ns, SPAWNS, TARGETS):
        g = eps.get((n, sp, tg))
        if g is None:
            continue
        sub = df[(df.N == n) & (df.spawn == sp) & (df.target == tg) & (df.radius == TRAIN_RADIUS)]
        T = float(sub.max_steps.iloc[0]) if not sub.empty and pd.notna(sub.max_steps.iloc[0]) else 1800.0
        for arm in ARMS:
            A = g[g["mode"] == arm]
            if A.empty:
                continue
            solved = A.solved.values.astype(bool)
            t = A.t_first.values.astype(float)
            ts = t[solved & np.isfinite(t)]
            pen = penalized_time(t, solved, T)
            rows.append(dict(
                N=n, spawn=sp, target=tg, arm=arm, episodes=len(A),
                n_solved=int(solved.sum()), success_rate=float(solved.mean()),
                t_med_cond=(float(np.median(ts)) if len(ts) else np.nan),
                t_mean_cond=(float(ts.mean()) if len(ts) else np.nan),
                t_penalized=pen, time_weighted_success=1.0 - pen / T, max_steps=T))
    return pd.DataFrame(rows)


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval for a proportion -- behaves at k=0 and k=n."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def newcombe_diff(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Newcombe score interval for p1 - p2 between INDEPENDENT samples.

    The paired McNemar intervals used elsewhere in this report are unavailable
    across the spawn axis (see spawn_shift), and a naive Wald interval is poor
    near 0 or 1 -- which is exactly where the harder spawn regimes sit.
    """
    if not n1 or not n2:
        return (np.nan, np.nan)
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    return (p1 - p2 - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
            p1 - p2 + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def pool_eps(eps: dict, n: int, cells) -> pd.DataFrame | None:
    """Episode frames for one swarm size over a set of (spawn, target) cells."""
    frames = [eps[(n, sp, tg)] for sp, tg in cells if (n, sp, tg) in eps]
    return pd.concat(frames, ignore_index=True) if frames else None


def budget_of(df: pd.DataFrame, n: int, **kw) -> float:
    """The episode budget T for a slice, defaulting to the trainers' max_steps."""
    m = (df.N == n) & (df.radius == TRAIN_RADIUS)
    for k, v in kw.items():
        m &= (df[k] == v)
    sub = df[m]
    return (float(sub.max_steps.iloc[0])
            if not sub.empty and pd.notna(sub.max_steps.iloc[0]) else 1800.0)


def outcome_stats(A: pd.DataFrame, T: float) -> dict:
    """First-arrival outcome of one arm on one pooled set of episodes.

    `solved` is success_any and `t_first` the earliest arrival in the swarm, so
    an episode counts once regardless of which agent got there.
    """
    solved = A.solved.values.astype(bool)
    t = A.t_first.values.astype(float)
    ts = t[solved & np.isfinite(t)]
    pen = penalized_time(t, solved, T)
    lo, hi = wilson(int(solved.sum()), len(A))
    return dict(episodes=len(A), n_solved=int(solved.sum()),
                success_rate=float(solved.mean()), success_lo=lo, success_hi=hi,
                t_med_cond=(float(np.median(ts)) if len(ts) else np.nan),
                t_mean_cond=(float(ts.mean()) if len(ts) else np.nan),
                t_penalized=pen, time_weighted_success=1.0 - pen / T, max_steps=T)


def spawn_matrix(eps: dict, df: pd.DataFrame, ns: list[int]) -> pd.DataFrame:
    """Section 8: outcome by spawn regime, pooled over target mode.

    Pooling happens at the EPISODE level, not by averaging the two per-target
    summaries: the mean of two medians is not a median, and the penalized mean
    has to stay weighted by episode count. The target axis is not the subject of
    this section -- it has its own -- and both modes contribute the same number
    of episodes to every spawn regime, so the pooling is balanced and cannot
    manufacture a Simpson reversal.

    Everything here is the FIRST arrival: `solved` is success_any and `t_first`
    is the minimum arrival step over the swarm, so an episode counts once no
    matter which agent got there.
    """
    rows = []
    for n, sp in itertools.product(ns, SPAWNS):
        g = pool_eps(eps, n, [(sp, tg) for tg in TARGETS])
        if g is None:
            continue
        T = budget_of(df, n, spawn=sp)
        for arm in ARMS:
            A = g[g["mode"] == arm]
            if not A.empty:
                rows.append(dict(N=n, spawn=sp, arm=arm, **outcome_stats(A, T)))
    return pd.DataFrame(rows)


def target_matrix(eps: dict, df: pd.DataFrame, ns: list[int]) -> pd.DataFrame:
    """Section 9: outcome by target regime, pooled over spawn geometry.

    The mirror of spawn_matrix: there the target axis was collapsed, here the
    spawn axis is, so each section reports the factor it is about with the other
    averaged out. Pooling is again at the episode level and again balanced --
    every target regime is evaluated under all three spawn geometries, 100
    episodes each.
    """
    rows = []
    for n, tg in itertools.product(ns, TARGETS):
        g = pool_eps(eps, n, [(sp, tg) for sp in SPAWNS])
        if g is None:
            continue
        T = budget_of(df, n, target=tg)
        for arm in ARMS:
            A = g[g["mode"] == arm]
            if not A.empty:
                rows.append(dict(N=n, target=tg, arm=arm, **outcome_stats(A, T)))
    return pd.DataFrame(rows)


def spawn_shift(sm: pd.DataFrame, ns: list[int]) -> pd.DataFrame:
    """Each out-of-distribution spawn against the trained one, per arm.

    UNPAIRED, unlike every other contrast in this report, and deliberately so.
    Changing `spawn_mode` changes how reset() consumes the RNG -- `origin` and
    `max_dist` place agents deterministically while `random` draws them -- so the
    same seed yields a DIFFERENT target under a different spawn regime and the
    episodes cannot be matched. McNemar would be invalid here; Fisher's exact
    test and a Newcombe interval assume only independent samples, which is what
    we actually have.
    """
    rows = []
    for n, arm in itertools.product(ns, ARMS):
        base = sm[(sm.N == n) & (sm.spawn == TRAIN_SPAWN) & (sm.arm == arm)]
        if base.empty:
            continue
        b = base.iloc[0]
        for sp in SPAWNS:
            if sp == TRAIN_SPAWN:
                continue
            cur = sm[(sm.N == n) & (sm.spawn == sp) & (sm.arm == arm)]
            if cur.empty:
                continue
            c = cur.iloc[0]
            p = fisher_exact([[int(c.n_solved), int(c.episodes - c.n_solved)],
                              [int(b.n_solved), int(b.episodes - b.n_solved)]])[1]
            lo, hi = newcombe_diff(int(c.n_solved), int(c.episodes),
                                   int(b.n_solved), int(b.episodes))
            rows.append(dict(
                N=n, arm=arm, spawn=sp, ref_spawn=TRAIN_SPAWN,
                rate=c.success_rate, ref_rate=b.success_rate,
                delta_pp=100 * (c.success_rate - b.success_rate),
                delta_lo_pp=100 * lo, delta_hi_pp=100 * hi, p=p,
                t_penalized=c.t_penalized, ref_t_penalized=b.t_penalized,
                delta_t_penalized=c.t_penalized - b.t_penalized,
                t_med_cond=c.t_med_cond, ref_t_med_cond=b.t_med_cond,
                n_solved=int(c.n_solved), episodes=int(c.episodes)))
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_bh"] = false_discovery_control(out["p"].values, method="bh")
    return out


def scaling_matrix(eps: dict, df: pd.DataFrame, ns: list[int]) -> pd.DataFrame:
    """Section 10: outcome and search behaviour by swarm size, pooled over ALL cells.

    Neither deployment axis is the subject here, so both are averaged out and
    every entry is 600 episodes. `cov_red` and `nn_dist` are only recorded per
    scenario, not per episode, so they come from the cell summaries.
    """
    rows = []
    for n in ns:
        g = pool_eps(eps, n, CELLS)
        if g is None:
            continue
        T = budget_of(df, n)
        for arm in ARMS:
            A = g[g["mode"] == arm]
            if A.empty:
                continue
            sub = df[(df.N == n) & (df.radius == TRAIN_RADIUS) & (df.arm == arm)]
            rows.append(dict(N=n, arm=arm, **outcome_stats(A, T),
                             swarm_path=float(A.swarm_path.mean()),
                             cov_red=float(sub.cov_red.mean()) if not sub.empty else np.nan,
                             nn_dist=float(sub.nn_dist.mean()) if not sub.empty else np.nan))
    return pd.DataFrame(rows)


def scaling_independence(sc: pd.DataFrame, ns: list[int],
                         n_boot: int = 20000, seed: int = 0) -> pd.DataFrame:
    """The larger swarm against what INDEPENDENT agents would already give you.

    "More agents solve more episodes" is true by construction and says nothing
    about coordination: r non-interacting copies of an r-times-smaller swarm
    already reach

        p_pred  =  1 - (1 - p_base)^r

    simply by each trying separately. That is the null this section needs. A
    swarm above it searches better together than apart; one below it is getting
    in its own way. Because p_pred is itself estimated, the interval comes from
    a parametric bootstrap over BOTH binomials rather than treating the
    prediction as a known constant.

    The comparison is unpaired -- reset() places the agents before it draws the
    target, so a seed gives a different task at a different N (verified against
    the env) -- but both swarm sizes sample the same task distribution, so the
    rates remain comparable.
    """
    rng = np.random.default_rng(seed)
    base = min(ns)
    rows = []
    for n in ns:
        if n == base:
            continue
        r = n / base
        if abs(r - round(r)) > 1e-9:      # only exact multiples have this null
            continue
        r = int(round(r))
        for arm in ARMS:
            a = sc[(sc.N == base) & (sc.arm == arm)]
            b = sc[(sc.N == n) & (sc.arm == arm)]
            if a.empty or b.empty:
                continue
            a, b = a.iloc[0], b.iloc[0]
            na, nb = int(a.episodes), int(b.episodes)
            pa, pb = a.n_solved / na, b.n_solved / nb
            pred = 1.0 - (1.0 - pa) ** r
            d = (rng.binomial(nb, pb, n_boot) / nb
                 - (1.0 - (1.0 - rng.binomial(na, pa, n_boot) / na) ** r))
            lo, hi = np.percentile(d, [2.5, 97.5])
            rows.append(dict(
                arm=arm, n_base=base, N=n, ratio=r, rate_base=pa, rate_obs=pb,
                rate_pred=pred, gap_pp=100 * (pb - pred),
                gap_lo_pp=100 * lo, gap_hi_pp=100 * hi,
                p=min(1.0, 2 * min(float((d <= 0).mean()), float((d >= 0).mean())))))
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_bh"] = false_discovery_control(out["p"].values, method="bh")
    return out


def scaling_margin(paired: dict, ns: list[int]) -> pd.DataFrame:
    """How much the learned policy is worth over an untrained swarm, at each N.

    Paired within a swarm size (same episodes, same tasks), so exact McNemar
    applies -- this is the one contrast in the section that keeps its pairing.
    """
    rows = []
    for n in ns:
        keys = [(n, sp, tg, TRAIN_RADIUS) for sp, tg in CELLS]
        for arm in POLICY_ARMS:
            m = mcnemar(paired, keys, "random", arm)
            if m:
                rows.append(dict(N=n, arm=arm, **m))
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_bh"] = false_discovery_control(out["p"].values, method="bh")
    return out


def paired_flip(paired: dict, col: str, keys_a, keys_b) -> dict | None:
    """Pooled discordant counts for ONE arm between two matched scenario families.

    Unlike mcnemar(), which contrasts two arms inside one scenario, this
    contrasts one arm across two scenarios whose episodes line up row for row.
    Only valid where that alignment actually holds -- see the callers.
    """
    a_only = b_only = n_a = n_b = total = 0
    for ka, kb in zip(keys_a, keys_b):
        da, db = paired.get(ka), paired.get(kb)
        if da is None or db is None or col not in da or col not in db:
            continue
        if len(da) != len(db):
            continue
        va, vb = da[col].values, db[col].values
        a_only += int(((va == 1) & (vb == 0)).sum())
        b_only += int(((va == 0) & (vb == 1)).sum())
        n_a += int(va.sum())
        n_b += int(vb.sum())
        total += len(va)
    if not total:
        return None
    disc = a_only + b_only
    return dict(rate_a=n_a / total, rate_b=n_b / total,
                delta_pp=100 * (n_b - n_a) / total, gained=b_only, lost=a_only,
                p=binomtest(b_only, disc, 0.5).pvalue if disc else 1.0, n=total)


def spawn_paired(paired: dict, ns: list[int]) -> pd.DataFrame:
    """`origin` against `max_dist` with the TASK HELD FIXED -- the one paired
    contrast available on the spawn axis, and the strongest statement in the
    section.

    Verified against the env rather than assumed: reset() draws the NetCDF file
    BEFORE placing the agents but the snapshot index and the target AFTER, and
    only spawn_mode="random" consumes randomness for the positions. So at a given
    seed `origin` and `max_dist` get the same field, the same frozen snapshot and
    the same target, and differ ONLY in where the swarm starts -- while `random`
    diverges from both and can only be compared unpaired (see spawn_shift).
    """
    rows = []
    for n, arm in itertools.product(ns, ARMS):
        m = paired_flip(paired, f"{arm}_any",
                        [(n, "origin", tg, TRAIN_RADIUS) for tg in TARGETS],
                        [(n, "max_dist", tg, TRAIN_RADIUS) for tg in TARGETS])
        if m:
            rows.append(dict(N=n, arm=arm, rate_origin=m.pop("rate_a"),
                             rate_max_dist=m.pop("rate_b"), **m))
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_bh"] = false_discovery_control(out["p"].values, method="bh")
    return out


def target_paired(paired: dict, ns: list[int]) -> pd.DataFrame:
    """Rare (`tail`) target against the ordinary one, per arm, with everything
    else held fixed.

    Fully paired, and the cleanest manipulation in the sweep. reset() draws the
    field, the frozen snapshot, the spawn positions and the headings BEFORE it
    picks the target, and `target_mode` changes only that last draw -- verified
    against the env, which returns byte-identical agent placements under both
    modes at a given seed. So an episode pair differs in the rarity of the
    target and in nothing else, which is exactly what this section claims to
    measure, and exact McNemar applies.
    """
    rows = []
    for n, arm in itertools.product(ns, ARMS):
        m = paired_flip(paired, f"{arm}_any",
                        [(n, sp, "random", TRAIN_RADIUS) for sp in SPAWNS],
                        [(n, sp, "tail", TRAIN_RADIUS) for sp in SPAWNS])
        if m:
            rows.append(dict(N=n, arm=arm, rate_random=m.pop("rate_a"),
                             rate_tail=m.pop("rate_b"), **m))
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_bh"] = false_discovery_control(out["p"].values, method="bh")
    return out


def rank_arrival(arrivals: pd.DataFrame) -> pd.DataFrame:
    """Per-episode arrival time of the r-th agent, with censoring flag.

    The event for rank r is the r-th order statistic of the swarm's arrival
    times. In an episode where fewer than r agents ever arrived, that statistic
    is censored at max_steps -- which is the only censoring time possible here,
    since the episode ends early only when everyone has arrived.
    """
    rows = []
    keys = ["scenario", "N", "arm", "episode"]
    for (scen, n, arm, ep), g in arrivals.groupby(keys, sort=False):
        T = float(g.max_steps.iloc[0])
        got = np.sort(g.loc[g.reached == 1, "arrival_step"].dropna().values.astype(float))
        for r in range(1, int(n) + 1):
            observed = len(got) >= r
            rows.append(dict(scenario=scen, N=int(n), arm=arm, episode=ep, rank=r,
                             observed=bool(observed),
                             t=(float(got[r - 1]) if observed else np.nan),
                             max_steps=T))
    return pd.DataFrame(rows)


def rank_table(rk: pd.DataFrame) -> pd.DataFrame:
    """Per (scenario, N, arm, rank): reach probability, conditional and penalized time."""
    rows = []
    for (scen, n, arm, r), g in rk.groupby(["scenario", "N", "arm", "rank"], sort=False):
        T = float(g.max_steps.iloc[0])
        obs = g.observed.values.astype(bool)
        t = g.t.values.astype(float)
        ts = t[obs]
        pen = penalized_time(t, obs, T)
        rows.append(dict(scenario=scen, N=int(n), arm=arm, rank=int(r), episodes=len(g),
                         reach_rate=float(obs.mean()), n_reached=int(obs.sum()),
                         t_med_cond=(float(np.median(ts)) if len(ts) else np.nan),
                         t_mean_cond=(float(ts.mean()) if len(ts) else np.nan),
                         t_penalized=pen, time_weighted=1.0 - pen / T, max_steps=T))
    return pd.DataFrame(rows).sort_values(["N", "scenario", "rank", "arm"])


def ecdf(t: np.ndarray, observed: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Empirical arrival CDF over ALL episodes (censored ones never fire).

    With one common administrative censoring time this is unbiased and
    identical to the Kaplan-Meier estimate, so no survival library is needed.
    """
    t = np.asarray(t, float)
    observed = np.asarray(observed, bool)
    ok = observed & np.isfinite(t)
    return np.array([(ok & (t <= b)).sum() / len(t) for b in grid])


def stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


# ------------------------------------------------------------------- LaTeX
def _tex_num(v, nd=2, dash="--"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return dash
    return f"{v:.{nd}f}"


MAX_FLOAT_ROWS = 28      # beyond this a `table` float overflows the page


def latex_table(path: Path, caption: str, label: str, header: list[str],
                rows: list[list[str]], align: str | None = None, note: str | None = None):
    """One \\input-able table in the idiom already used by thesis/ch3.tex.

    Falls back to `longtable` past MAX_FLOAT_ROWS: a `table` float cannot break
    across pages, so a long one silently runs off the bottom. Both packages are
    already in the thesis preamble.
    """
    ncol = len(header)
    align = align or ("l" + "c" * (ncol - 1))
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    head = " &\n".join(header) + r" \\ \midrule"
    spec = (r"@{\hspace*{1.5em}}" + align[0] + r"@{\extracolsep{\fill}}"
            + align[1:] + r"@{\hspace*{1.5em}}")
    notes = f"\n\\vspace{{0.25em}}\n{{\\footnotesize {note}}}" if note else ""
    stamp = "% generated by stats/analyze_sweeps.py -- do not edit by hand"

    if len(rows) > MAX_FLOAT_ROWS:
        path.write_text(
            f"""{stamp}
{{\\footnotesize
\\begin{{longtable}}{{{spec}}}
\\caption{{{caption}\\label{{{label}}}}}\\\\
\\toprule
{head}
\\endfirsthead
\\toprule
{head}
\\endhead
\\bottomrule
\\endfoot
{body}
\\end{{longtable}}
}}{notes}
""")
        return path

    path.write_text(
        f"""{stamp}
\\begin{{table}}[t]
\\caption{{{caption}\\label{{{label}}}}}\\vspace{{0.25em}}
\\centering{{%
\\tagpdfsetup{{table/header-rows={{1}}}}
\\begin{{tabular*}}{{\\textwidth}}{{{spec}}}\\toprule
{head}
{body}
\\bottomrule
\\end{{tabular*}}
}}%{notes}
\\end{{table}}
""")
    return path


# ------------------------------------------------------------------ figures
def _style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK_2)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def _fig(*a, **kw):
    fig, ax = plt.subplots(*a, **kw)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _save(fig, out: Path, name: str):
    p = out / f"{name}.pdf"
    fig.savefig(p, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return p


def _spread(items, min_gap):
    """Push apart end-of-line labels that would otherwise overprint.

    `items` is a list of [value, y, ...]; `y` is nudged upward in place so
    consecutive labels sit at least `min_gap` apart, preserving their order.
    """
    items.sort(key=lambda e: e[0])
    for i in range(1, len(items)):
        items[i][1] = max(items[i][1], items[i - 1][1] + min_gap)
    return items


# Figures carry no title or subtitle: in the thesis that text belongs in the
# LaTeX caption, and duplicating it inside the artwork looks wrong at print size.
# The wording lives here instead and is emitted as ready-made float blocks by
# write_figure_captions(), so nothing has to be retyped.
CAPTIONS = {
    "fig1_mechanism_effects": (
        "Effect of each coordination mechanism on episode success, by deployment geometry",
        "Paired exact McNemar within each cell; significance after Benjamini--Hochberg "
        "correction over all cells. Blue = the mechanism helps, red = it hurts. Rows of "
        "panels separate the two target modes: the \\texttt{tail} row is the rare-target "
        "regime coordination is meant to pay for. "
        "The $N=2$ and $N=4$ bars are separate paired experiments, so their difference is "
        "unpaired and is shown rather than tested."),
    "fig2_success_matrix": (
        "Episode success across the deployment cells",
        "Deployment communication radius as trained. The grey rule is the random-policy "
        "floor for that cell; the shaded column is the training condition, and every other "
        "cell is out of distribution."),
    "fig3_comms_range": (
        "Restricting the communication range at deployment",
        "Faint lines are individual deployment cells, bold is the mean over them. Every "
        "policy was trained with an unlimited link, so any change along this axis is a "
        "train/deploy mismatch rather than a learned response."),
    "fig4_success_at_budget": (
        "Success within a step budget",
        "Pooled over the deployment cells. Episodes never solved count as failures at every "
        "budget, so the curves are free of survivorship bias. The normalized area under a "
        "curve is the time-weighted success of Table~\\ref{tab:time_penalized}."),
    "fig5_time_to_first_success": (
        "Time to first success: the conditional view disagrees with the penalized one",
        "Left: median over each arm's own successes, which is survivorship-biased -- an arm "
        "that solves fewer episodes drops its slow ones and can look faster ($n$ = episodes "
        "solved, printed on each bar). Right: every episode counted, with unsolved episodes "
        "charged the full budget."),
    "fig6_arrival_by_rank": (
        "Arrival of the $r$-th agent, not just the first",
        "Empirical arrival CDF of the $r$-th agent over all episodes; the plateau is the "
        "fraction that ever arrived within the budget. Censoring is administrative at "
        "\\texttt{max\\_steps}, so this equals the Kaplan--Meier estimate. Plateau heights "
        "are bounded by the previous rank -- an $r$-th agent cannot arrive unless $r-1$ "
        "already did -- so the informative comparison is between arms within a panel."),
    "fig8_spawn_effect": (
        "Where the swarm starts: effect of the spawn geometry on whether the target is "
        "found, and on how long the first agent takes to find it",
        "Spawn regimes are ordered by how dispersed the swarm is at $t=0$: \\texttt{origin} "
        "places every agent on a single point at the corner of the coast, \\texttt{random} is "
        "the uniform draw the policies were trained on (shaded), \\texttt{max\\_dist} spaces "
        "them evenly along the coastline. Faint lines are the two target modes separately, "
        "bold is the pooled result. Both rows describe the FIRST arrival only: an episode ends "
        "when any one agent reaches the zone. Top: episode success rate. Bottom: penalized "
        "time $E[\\min(t,T)]$, which charges an unsolved episode the full budget and therefore "
        "cannot be improved by failing more often -- the conditional median, which can, is in "
        "Table~\\ref{tab:spawn_effect}. The grey random-policy line is the control: where it "
        "moves in step with the others, the spawn regime changed the difficulty of the task "
        "itself rather than the value of what the policies learned."),
    "fig9_hard_targets": (
        "Cost of a rare target: how much later, and how much less often, the swarm arrives",
        "Empirical distribution of the time to first arrival, pooled over the three spawn "
        "geometries. The value each curve reaches at the right-hand edge is that arm's success "
        "rate and the shape of its rise is its speed, so both quantities are read off one plot; "
        "episodes that are never solved simply never fire, which is why no curve reaches~1. "
        "Note that the curves are still climbing when the budget cuts them off at "
        "\\texttt{max\\_steps}: none of these arms has finished finding what it would eventually "
        "find, so every success rate quoted here is a rate \\emph{within the budget} rather than "
        "an asymptote. The tick on each "
        "curve marks the median over that arm's own successes -- the conditional number of "
        "Table~\\ref{tab:hard_targets}, shown against the distribution it conditions away. "
        "Left: an ordinary target anywhere in the domain. Right: a target drawn from the "
        "rarest few per cent of the salinity distribution in its depth plane, which shrinks the "
        "success zone. Axes are shared, so the cost of rarity is the rightward shift plus the "
        "drop in plateau."),
    "fig10_scaling_n": (
        "Effect of swarm size on finding the target and on how long it takes",
        "Pooled over all six deployment cells, 600 episodes per point. Left: episode success "
        "rate, counting an episode once whichever agent arrives first. Right: penalized time "
        "to first arrival $E[\\min(t,T)]$, which charges an unsolved episode the full budget "
        "and so cannot be improved by failing more often. The $N=2$ and $N=4$ episodes are "
        "separate draws from the same task distribution rather than matched pairs -- "
        "\\texttt{reset()} places the agents before it draws the target, so a seed gives a "
        "different task at a different swarm size -- and these are therefore unpaired "
        "comparisons of rates. Table~\\ref{tab:scaling_n} adds the search-overlap and travel "
        "cost of the extra agents, the margin over an untrained swarm of the same size, and "
        "the comparison against what non-interacting agents would already achieve."),
}


def write_figure_captions(paths: list[Path], out: Path) -> Path:
    """Ready-to-use float blocks for every generated figure."""
    blocks = []
    for p in paths:
        if p.suffix != ".pdf":
            continue
        key = p.stem
        entry = CAPTIONS.get(key)
        suffix = ""
        if entry is None:                      # per-N variants share one caption
            base = key.rsplit("_N", 1)[0]
            entry = CAPTIONS.get(base)
            if entry is None:
                continue
            suffix = f" ($N = {key.rsplit('_N', 1)[1]}$)"
        title, note = entry
        blocks.append(
            f"\\begin{{figure}}[t]\n"
            f"\\centering\n"
            f"\\includegraphics[width=\\textwidth]{{assets/{p.name}}}\n"
            f"\\caption[{title}]{{{title}{suffix}. {note}\\label{{fig:{key}}}}}\n"
            f"\\end{{figure}}\n")
    path = out / "figures.tex"
    path.write_text("% generated by stats/analyze_sweeps.py -- do not edit by hand\n"
                    "% \\input this, or copy the blocks you want into the chapter.\n\n"
                    + "\n".join(blocks))
    return path


CELLS = [(sp, tg) for sp in SPAWNS for tg in TARGETS]
CELL_TEX = {(sp, tg): f"{SPAWN_LABEL[sp]} / {tg}" for sp, tg in CELLS}


def fig_effects(cell_eff: pd.DataFrame, ns: list[int], out: Path):
    """Section 1: each mechanism's effect, per spawn regime, target mode and swarm size.

    Target mode gets its own ROW of panels rather than being pooled away: a rare
    tail target is the regime coordination is supposed to pay for, and averaging
    it with the easy random target hides that.
    """
    panels = [("Adding communication", ["Communication (IPPO)", "Communication (MAPPO)"]),
              ("Adding a centralized critic (CTDE)", ["CTDE, no comm", "CTDE, with comm"])]
    hatch = {n: h for n, h in zip(ns, ["", "//", "xx"])}
    fig, axes = _fig(len(TARGETS), len(panels), figsize=(11, 4.4 * len(TARGETS)),
                     sharex=True, squeeze=False)
    lim = 2 + 1.45 * float(np.abs(cell_eff.delta_pp).max())
    for r_i, tg in enumerate(TARGETS):
        for c_i, (title, contrasts) in enumerate(panels):
            ax = axes[r_i, c_i]
            labels, vals, ps, hs = [], [], [], []
            for c in contrasts:
                for sp in SPAWNS:
                    for n in ns:
                        r = cell_eff[(cell_eff.contrast == c) & (cell_eff.spawn == sp)
                                     & (cell_eff.target == tg) & (cell_eff.N == n)]
                        if r.empty:
                            continue
                        r = r.iloc[0]
                        tag = c.split("(")[-1].rstrip(")") if "(" in c else c.split(", ")[-1]
                        labels.append(f"{sp} · {tag} · N={n}")
                        vals.append(r.delta_pp)
                        ps.append(r.p_bh)
                        hs.append(hatch[n])
            if not vals:
                ax.axis("off")
                continue
            y = np.arange(len(vals))[::-1]
            ax.barh(y, vals, height=0.66, color=[POS if v > 0 else NEG for v in vals],
                    hatch=hs, edgecolor=SURFACE, linewidth=0.6, zorder=3)
            ax.axvline(0, color=AXIS, linewidth=1.0, zorder=4)
            for yi, v, p in zip(y, vals, ps):
                ax.text(v + (0.7 if v >= 0 else -0.7), yi, f"{v:+.1f}", va="center",
                        ha="left" if v >= 0 else "right", fontsize=7.4, color=INK_2)
            ax.set_yticks(y, labels, fontsize=7.4)
            ax.set_title(f"{title}   —   {TARGET_LABEL[tg]}", fontsize=10.2, color=INK,
                         pad=10, loc="left", fontweight="semibold")
            ax.set_xlim(-lim, lim)
            _style(ax, grid_axis="x")
    for ax in axes[-1]:
        ax.set_xlabel("change in episode success rate (percentage points)",
                      fontsize=9, color=INK_2)
    handles = [Patch(facecolor=POS, label="helps"), Patch(facecolor=NEG, label="hurts")]
    # hatch is drawn in the EDGE colour, so a legend patch without one shows solid
    handles += [Patch(facecolor=MUTED, hatch=hatch[n], edgecolor=SURFACE,
                      linewidth=0.6, label=f"N = {n}") for n in ns]
    fig.legend(handles=handles, loc="lower center", ncol=2 + len(ns), frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return _save(fig, out, "fig1_mechanism_effects")


def fig_success_matrix(df: pd.DataFrame, ns: list[int], out: Path):
    """Section 2: absolute success for every arm across the deployment cells."""
    fig, axes = _fig(len(ns), 1, figsize=(11, 3.9 * len(ns)), sharex=True, squeeze=False)
    axes = axes[:, 0]
    x = np.arange(len(CELLS))
    w = 0.2
    for ax, n in zip(axes, ns):
        for i, arm in enumerate(POLICY_ARMS):
            vals = []
            for sp, tg in CELLS:
                s = df[(df.N == n) & (df.spawn == sp) & (df.target == tg)
                       & (df.radius == TRAIN_RADIUS) & (df.arm == arm)]
                vals.append(s.any_rate.iloc[0] if not s.empty else np.nan)
            pos = x + (i - 1.5) * (w + 0.012)
            ax.bar(pos, vals, width=w, color=ARM_COLOR[arm], zorder=3,
                   label=ARM_LABEL[arm] if n == ns[0] else None)
            for px, v in zip(pos, vals):
                if np.isfinite(v):
                    ax.text(px, v + 0.02, f"{v:.2f}", ha="center", fontsize=6.8, color=INK_2,
                            zorder=6, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.8))
        rnd = []
        for sp, tg in CELLS:
            s = df[(df.N == n) & (df.spawn == sp) & (df.target == tg)
                   & (df.radius == TRAIN_RADIUS) & (df.arm == "random")]
            rnd.append(s.any_rate.iloc[0] if not s.empty else np.nan)
        for xi, v in zip(x, rnd):
            ax.plot([xi - 0.44, xi + 0.44], [v, v], color=MUTED, linewidth=1.6, zorder=5,
                    label="random policy" if (n == ns[0] and xi == 0) else None)
        ax.axvspan(-0.5, 0.5, color=POS, alpha=0.05, zorder=1)
        ax.set_ylim(0, 1.14)
        ax.set_ylabel(f"N = {n}\nepisode success rate", fontsize=9.5, color=INK)
        _style(ax)
    axes[-1].set_xticks(x, [f"{sp}\n{TARGET_LABEL[tg]}" for sp, tg in CELLS], fontsize=9)
    axes[0].text(0, 1.075, "trained here", fontsize=8, color=POS, ha="center", style="italic")
    fig.legend(loc="lower center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return _save(fig, out, "fig2_success_matrix")


def fig_radius(df: pd.DataFrame, ns: list[int], out: Path):
    """Section 3: restricting the comms link at deployment."""
    fig, axes = _fig(1, len(ns), figsize=(5.4 * len(ns), 4.4), sharey=True, squeeze=False)
    axes = axes[0]
    xs = np.arange(len(RADII))
    for ax, n in zip(axes, ns):
        left, right = [], []
        for arm in ["ippo_comm", "mappo_comm"]:
            color = ARM_COLOR[arm]
            for sp, tg in CELLS:
                ys = []
                for r in RADII:
                    s = df[(df.N == n) & (df.spawn == sp) & (df.target == tg)
                           & (df.radius == r) & (df.arm == arm)]
                    ys.append(s.any_rate.iloc[0] if not s.empty else np.nan)
                ax.plot(xs, ys, color=color, linewidth=0.9, alpha=0.22, zorder=2)
            mean = [df[(df.N == n) & (df.radius == r) & (df.arm == arm)].any_rate.mean()
                    for r in RADII]
            ax.plot(xs, mean, color=color, linewidth=2.4, marker="o", markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=4,
                    label=ARM_LABEL[arm] if n == ns[0] else None)
            right.append([mean[-1], mean[-1], color])
            left.append([mean[0], mean[0], color])
        for val, y, color in _spread(right, 0.045):
            ax.text(len(RADII) - 0.94, y, f"{val:.2f}", fontsize=8.5, color=color, va="center")
        for val, y, color in _spread(left, 0.045):
            ax.text(-0.06, y, f"{val:.2f}", fontsize=8.5, color=color, va="center", ha="right")
        ax.set_xticks(xs, ["inf\n(as trained)", "250 m", "0 m\n(link cut)"], fontsize=9)
        ax.set_xlim(-0.35, len(RADII) - 0.65)
        ax.set_title(f"N = {n}", fontsize=10.5, color=INK, loc="left", fontweight="semibold")
        _style(ax)
    axes[0].set_ylabel("episode success rate", fontsize=9.5, color=INK_2)
    axes[0].set_ylim(0, 1.0)
    fig.legend(loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, out, "fig3_comms_range")


def fig_budget(curves: pd.DataFrame, ns: list[int], out: Path):
    """Section 4: success within a step budget."""
    fig, axes = _fig(1, len(ns), figsize=(5.6 * len(ns), 4.6), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, n in zip(axes, ns):
        ends = []
        for arm in ARMS:
            sub = curves[(curves.N == n) & (curves.arm == arm)].sort_values("budget")
            if sub.empty:
                continue
            is_rnd = arm == "random"
            color = MUTED if is_rnd else ARM_COLOR[arm]
            ax.plot(sub.budget, sub.rate, color=color, linewidth=1.6 if is_rnd else 2.2,
                    marker="" if is_rnd else "o", markersize=5, markeredgecolor=SURFACE,
                    markeredgewidth=1.2, zorder=3, label=ARM_LABEL[arm] if n == ns[0] else None)
            ends.append([sub.iloc[-1].rate, sub.iloc[-1].rate, color])
        for rate, y, color in _spread(ends, 0.042):
            ax.text(BUDGETS[-1] * 1.07, y, f"{rate:.2f}", fontsize=8, color=color, va="center")
        ax.set_xscale("log")
        ax.set_xticks(BUDGETS, [str(b) for b in BUDGETS], fontsize=7.5)
        ax.minorticks_off()
        ax.set_xlim(90, BUDGETS[-1] * 1.5)
        ax.set_xlabel("step budget", fontsize=9, color=INK_2)
        ax.set_title(f"N = {n}", fontsize=10.5, color=INK, loc="left", fontweight="semibold")
        _style(ax, grid_axis="both")
    axes[0].set_ylabel("fraction of episodes solved within budget", fontsize=9.5, color=INK_2)
    axes[0].set_ylim(0, 1.0)
    fig.legend(loc="lower center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _save(fig, out, "fig4_success_at_budget")


def fig_time(tm: pd.DataFrame, ns: list[int], out: Path):
    """Section 5: conditional arrival time vs the penalized metric.

    Two panels per swarm size makes the point of the section visible: the
    ordering under a conditional median is not the ordering under a metric that
    charges failures.
    """
    fig, axes = _fig(len(ns), 2, figsize=(12, 3.9 * len(ns)), squeeze=False)
    x = np.arange(len(CELLS))
    w = 0.2
    for row, n in enumerate(ns):
        for col, (field, title, ylab) in enumerate([
                ("t_med_cond", "Median time to first arrival (successes only)",
                 "steps  ·  lower = faster"),
                ("time_weighted_success", "Time-weighted success  $1-E[\\min(t,T)]/T$",
                 "0 = never solved  ·  1 = solved instantly")]):
            ax = axes[row, col]
            for i, arm in enumerate(POLICY_ARMS):
                vals, ann = [], []
                for sp, tg in CELLS:
                    s = tm[(tm.N == n) & (tm.spawn == sp) & (tm.target == tg) & (tm.arm == arm)]
                    vals.append(s[field].iloc[0] if not s.empty else np.nan)
                    ann.append(int(s.n_solved.iloc[0]) if not s.empty else 0)
                pos = x + (i - 1.5) * (w + 0.012)
                ax.bar(pos, vals, width=w, color=ARM_COLOR[arm], zorder=3,
                       label=ARM_LABEL[arm] if (row == 0 and col == 0) else None)
                if field == "t_med_cond":
                    for px, v, k in zip(pos, vals, ann):
                        if np.isfinite(v):
                            ax.text(px, v, f"n={k}", ha="center", va="bottom", fontsize=6,
                                    color=MUTED, rotation=90, zorder=6)
            ax.set_xticks(x, [f"{sp}\n{tg}" for sp, tg in CELLS], fontsize=7.6)
            ax.set_ylabel(f"N = {n}\n{ylab}", fontsize=8.6, color=INK_2)
            ax.set_title(title, fontsize=9.8, color=INK, loc="left", fontweight="semibold")
            if field == "time_weighted_success":
                ax.set_ylim(0, 1.0)
            _style(ax)
    fig.legend(loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return _save(fig, out, "fig5_time_to_first_success")


def _pick(d: pd.DataFrame, field: str, **kw) -> float:
    """Single scalar out of a tidy frame, NaN when that row does not exist."""
    m = np.ones(len(d), dtype=bool)
    for k, v in kw.items():
        m &= (d[k] == v).values
    s = d.loc[m, field]
    return float(s.iloc[0]) if len(s) else np.nan


def fig_spawn(sm: pd.DataFrame, tm: pd.DataFrame, ns: list[int], out: Path):
    """Section 8: success and time to first arrival across the spawn regimes.

    Same idiom as fig3 (faint per-cell lines, bold mean) so the two
    deployment-shift figures read alike, but on the spawn axis and with the
    random-policy control drawn in, because a spawn regime can move every arm at
    once by simply changing how far the target is.
    """
    # Headroom on the success panel is for the "trained here" annotation, which
    # otherwise rides over the panel title.
    metrics = [("success_rate", "Episode success rate (first agent to arrive)",
                "episode success rate", (0.0, 1.18)),
               ("t_penalized", "Penalized time to first arrival  $E[\\min(t,T)]$",
                "steps  ·  lower = faster", None)]
    fig, axes = _fig(len(metrics), len(ns), figsize=(5.8 * len(ns), 4.3 * len(metrics)),
                     squeeze=False, sharex=True)
    xs = np.arange(len(SPAWN_ORDER))
    trained_i = SPAWN_ORDER.index(TRAIN_SPAWN)
    for r_i, (field, title, ylab, ylim) in enumerate(metrics):
        for c_i, n in enumerate(ns):
            ax = axes[r_i, c_i]
            ax.axvspan(trained_i - 0.5, trained_i + 0.5, color=POS, alpha=0.05, zorder=1)
            ends = []
            for arm in ARMS:
                is_rnd = arm == "random"
                color = MUTED if is_rnd else ARM_COLOR[arm]
                if not is_rnd:                       # per-target detail behind the mean
                    for tg in TARGETS:
                        ax.plot(xs, [_pick(tm, field, N=n, spawn=sp, target=tg, arm=arm)
                                     for sp in SPAWN_ORDER],
                                color=color, linewidth=0.9, alpha=0.22, zorder=2)
                ys = [_pick(sm, field, N=n, spawn=sp, arm=arm) for sp in SPAWN_ORDER]
                ax.plot(xs, ys, color=color, linewidth=1.6 if is_rnd else 2.4,
                        marker="" if is_rnd else "o", markersize=7,
                        markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=4,
                        label=ARM_LABEL[arm] if (r_i == 0 and c_i == 0) else None)
                if np.isfinite(ys[-1]):
                    ends.append([ys[-1], ys[-1], color])
            if ylim:
                ax.set_ylim(*ylim)
            else:
                ax.set_ylim(bottom=0)
            ax.set_xlim(-0.4, len(SPAWN_ORDER) - 0.55)
            lo, hi = ax.get_ylim()
            for val, y, color in _spread(ends, 0.05 * (hi - lo)):
                ax.text(len(SPAWN_ORDER) - 0.92, y,
                        f"{val:.2f}" if field == "success_rate" else f"{val:.0f}",
                        fontsize=8.5, color=color, va="center")
            ax.set_title(f"{title}   —   $N = {n}$", fontsize=9.8, color=INK,
                         loc="left", fontweight="semibold")
            if c_i == 0:
                ax.set_ylabel(ylab, fontsize=9.5, color=INK_2)
            _style(ax)
    for ax in axes[-1]:
        ax.set_xticks(xs, [SPAWN_PLOT[sp] for sp in SPAWN_ORDER], fontsize=8.4)
        ax.set_xlabel("initial swarm dispersion  →", fontsize=9, color=INK_2)
    for ax in axes[0]:
        ax.text(trained_i, 1.07, "trained here", fontsize=8, color=POS,
                ha="center", style="italic")
    fig.legend(loc="lower center", ncol=5, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, out, "fig8_spawn_effect")


def fig_targets(eps: dict, df: pd.DataFrame, tmx: pd.DataFrame, ns: list[int], out: Path):
    """Section 9: the ordinary target against the rare one.

    Plotted as the time-to-first-success CDF rather than as two bar charts,
    because that shows BOTH quantities this section is about at once and shows
    them honestly: the height of the plateau is the success rate, the shape of
    the rise is the speed, and an episode that is never solved simply never
    fires instead of being dropped. Left and right columns share their axes, so
    the cost of a rare target is the horizontal shift plus the drop in plateau.
    """
    fig, axes = _fig(len(ns), len(TARGETS), figsize=(6.0 * len(TARGETS), 4.2 * len(ns)),
                     squeeze=False, sharex=True, sharey=True)
    for r_i, n in enumerate(ns):
        for c_i, tg in enumerate(TARGETS):
            ax = axes[r_i, c_i]
            g = pool_eps(eps, n, [(sp, tg) for sp in SPAWNS])
            if g is None:
                ax.axis("off")
                continue
            T = budget_of(df, n, target=tg)
            grid = np.geomspace(BUDGETS[0] * 0.5, T, 160)
            ends = []
            for arm in ARMS:
                A = g[g["mode"] == arm]
                if A.empty:
                    continue
                is_rnd = arm == "random"
                color = MUTED if is_rnd else ARM_COLOR[arm]
                solved = A.solved.values.astype(bool)
                y = ecdf(A.t_first.values, solved, grid)
                ax.plot(grid, y, color=color, linewidth=1.6 if is_rnd else 2.2, zorder=3,
                        label=ARM_LABEL[arm] if (r_i == 0 and c_i == 0) else None)
                # Median of the arm's OWN successes, marked on its curve: the
                # number Table 9 reports conditionally, shown where it actually
                # falls on the distribution it was conditioned out of.
                med = _pick(tmx, "t_med_cond", N=n, target=tg, arm=arm)
                if np.isfinite(med):
                    ax.plot([med], [ecdf(A.t_first.values, solved, np.array([med]))[0]],
                            marker="|", markersize=9, markeredgewidth=1.8, color=color,
                            zorder=5)
                ends.append([y[-1], y[-1], color])
            for val, yy, color in _spread(ends, 0.052):
                ax.text(T * 1.06, yy, f"{val:.2f}", fontsize=8, color=color, va="center")
            ax.set_xscale("log")
            ax.set_xticks(BUDGETS, [str(b) for b in BUDGETS], fontsize=7.5)
            ax.minorticks_off()
            ax.set_xlim(BUDGETS[0] * 0.85, T * 1.35)
            ax.set_ylim(0, 1.0)
            ax.set_title(f"{TARGET_LABEL[tg]}   —   $N = {n}$", fontsize=10.2, color=INK,
                         loc="left", fontweight="semibold")
            if c_i == 0:
                ax.set_ylabel("fraction of episodes solved by $t$", fontsize=9, color=INK_2)
            if r_i == len(ns) - 1:
                ax.set_xlabel("steps to first arrival", fontsize=9, color=INK_2)
            _style(ax, grid_axis="both")
    fig.legend(loc="lower center", ncol=5, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    return _save(fig, out, "fig9_hard_targets")


def fig_scaling(sc: pd.DataFrame, ns: list[int], out: Path):
    """Section 10: success and speed as the swarm grows.

    Levels only. The independent-swarm benchmark and the margin over an
    untrained swarm are still computed and still reported, but in
    Table~\\ref{tab:scaling_n} rather than here.
    """
    metrics = [("success_rate", "Episode success rate",
                "episode success rate", (0.0, 1.0)),
               ("t_penalized", "Penalized time to first arrival  $E[\\min(t,T)]$",
                "steps  ·  lower = faster", None)]
    fig, axes = _fig(1, len(metrics), figsize=(6.0 * len(metrics), 4.5), squeeze=False)
    xs = np.array(ns, dtype=float)
    pad = 0.08 * (xs.max() - xs.min())
    for c_i, (field, title, ylab, ylim) in enumerate(metrics):
        ax = axes[0, c_i]
        ends = []
        for arm in ARMS:
            is_rnd = arm == "random"
            color = MUTED if is_rnd else ARM_COLOR[arm]
            ys = [_pick(sc, field, N=n, arm=arm) for n in ns]
            ax.plot(xs, ys, color=color, linewidth=1.6 if is_rnd else 2.4,
                    marker="" if is_rnd else "o", markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=4,
                    label=ARM_LABEL[arm] if c_i == 0 else None)
            if np.isfinite(ys[-1]):
                ends.append([ys[-1], ys[-1], color])
        if ylim:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(bottom=0)
        lo, hi = ax.get_ylim()
        span = hi - lo
        fmt = (lambda v: f"{v:.2f}") if field == "success_rate" else (lambda v: f"{v:.0f}")
        vals = [e[0] for e in ends]
        if len(vals) > 1 and (max(vals) - min(vals)) < 0.06 * span:
            # Arms that finish on top of one another: stacked labels would end up
            # further from their own lines than from each other and imply a
            # separation that is not in the data. State the range once.
            ax.text(xs.max() + 0.45 * pad, float(np.mean(vals)),
                    f"{fmt(min(vals))}–{fmt(max(vals))}", fontsize=8.5,
                    color=INK_2, va="center")
        else:
            for val, y, color in _spread(ends, 0.045 * span):
                ax.text(xs.max() + 0.45 * pad, y, fmt(val), fontsize=8.5, color=color,
                        va="center")
        ax.set_xticks(xs, [f"$N = {int(n)}$" for n in ns], fontsize=9)
        ax.set_xlim(xs.min() - pad, xs.max() + 2.3 * pad)
        ax.set_title(title, fontsize=9.9, color=INK, loc="left", fontweight="semibold")
        ax.set_ylabel(ylab, fontsize=9, color=INK_2)
        _style(ax)
    fig.legend(loc="lower center", ncol=5, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _save(fig, out, "fig10_scaling_n")


def fig_arrivals(rk: pd.DataFrame, rt: pd.DataFrame, out: Path):
    """Section 6: empirical arrival CDF per arrival rank, ONE FIGURE PER SWARM SIZE.

    Ranks run to N, so a shared grid over several swarm sizes leaves the smaller
    ones with empty columns. Splitting keeps every panel occupied and lets each
    figure be included at its own scale.
    """
    scen = sorted(rk.scenario.unique())
    made = []
    for n in sorted(rk.N.unique()):
        nrank = int(rk[rk.N == n]["rank"].max())
        fig, axes = _fig(len(scen), nrank, figsize=(3.5 * nrank, 3.1 * len(scen)),
                         squeeze=False, sharex=True, sharey=True)
        for r_i, s in enumerate(scen):
            for c_i in range(nrank):
                rank = c_i + 1
                ax = axes[r_i, c_i]
                sub = rk[(rk.scenario == s) & (rk.N == n) & (rk["rank"] == rank)]
                if sub.empty:
                    ax.axis("off")
                    continue
                T = float(sub.max_steps.iloc[0])
                grid = np.linspace(0, T, 120)
                ends = []
                for arm in ARMS:
                    a = sub[sub.arm == arm]
                    if a.empty:
                        continue
                    color = MUTED if arm == "random" else ARM_COLOR.get(arm, MUTED)
                    y = ecdf(a.t.values, a.observed.values, grid)
                    ax.plot(grid, y, color=color, linewidth=1.5 if arm == "random" else 2.0,
                            zorder=3, label=ARM_LABEL[arm] if (r_i == 0 and c_i == 0) else None)
                    ends.append([y[-1], y[-1], color])
                for val, yy, color in _spread(ends, 0.058):
                    ax.text(T * 1.02, yy, f"{val:.2f}", fontsize=7, color=color, va="center")
                ax.set_ylim(0, 1.05)
                ax.set_xlim(0, T * 1.18)
                ax.set_title(f"{s.replace('__', ' / ')} · rank {rank}"
                             + (" (first)" if rank == 1 else ""),
                             fontsize=9, color=INK, loc="left", fontweight="semibold")
                if c_i == 0:
                    ax.set_ylabel("P(arrived by t)", fontsize=8.6, color=INK_2)
                if r_i == len(scen) - 1:
                    ax.set_xlabel("steps", fontsize=8.6, color=INK_2)
                _style(ax, grid_axis="both")
        fig.legend(loc="lower center", ncol=5, frameon=False, fontsize=9,
                   bbox_to_anchor=(0.5, -0.04))
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        made.append(_save(fig, out, f"fig6_arrival_by_rank_N{n}"))
    return made


# ------------------------------------------------------------------- tables
def tab_effects(cell_eff: pd.DataFrame, ns: list[int], out: Path):
    """Same decomposition as fig1 -- per target mode, not pooled over it -- so the
    two artefacts report identical numbers under identical multiplicity."""
    header = ["Mechanism", "Spawn"] + [f"{tg}, $N={n}$" for tg in TARGETS for n in ns]
    rows = []
    for name, _, _ in CONTRASTS:
        for sp in SPAWNS:
            cells = [name, SPAWN_LABEL[sp]]
            for tg in TARGETS:
                for n in ns:
                    r = cell_eff[(cell_eff.contrast == name) & (cell_eff.spawn == sp)
                                 & (cell_eff.target == tg) & (cell_eff.N == n)]
                    cells.append(f"${r.iloc[0].delta_pp:+.1f}$~{stars(r.iloc[0].p_bh)}"
                                 if not r.empty else "--")
            rows.append(cells)
    return latex_table(
        out / "tab1_mechanism_effects.tex",
        "Effect of communication and of the centralized critic on episode success, "
        "by spawn regime, target mode and swarm size",
        "tab:mechanism_effects", header, rows,
        note="Paired exact McNemar within each cell (100 episodes per entry); "
             "\\texttt{***}~$p<0.001$, \\texttt{**}~$p<0.01$, \\texttt{*}~$p<0.05$ after "
             "Benjamini--Hochberg correction over all cells, \\texttt{ns} otherwise. The "
             "\\texttt{tail} columns are the regime coordination is meant to pay for: a rare "
             "target that a lone agent is unlikely to stumble onto. Columns are separate "
             "paired experiments, so differences \\emph{between} them are unpaired and are not "
             "tested. Effects pooled over target mode, which carry more power per test, are in "
             "\\texttt{effects\\_by\\_spawn.csv}.")


def tab_success_matrix(df: pd.DataFrame, ns: list[int], out: Path):
    header = ["$N$", "Deployment cell"] + [ARM_LABEL[a] for a in ARMS]
    rows = []
    for n in ns:
        for sp, tg in CELLS:
            cells = [str(n), CELL_TEX[(sp, tg)]]
            for arm in ARMS:
                s = df[(df.N == n) & (df.spawn == sp) & (df.target == tg)
                       & (df.radius == TRAIN_RADIUS) & (df.arm == arm)]
                cells.append(_tex_num(s.any_rate.iloc[0] if not s.empty else None))
            rows.append(cells)
    return latex_table(
        out / "tab2_success_matrix.tex",
        "Episode success rate (\\texttt{success\\_any}) for every arm across the deployment cells",
        "tab:success_matrix", header, rows,
        note=f"Deployment communication radius = \\texttt{{{TRAIN_RADIUS}}} (as trained). "
             "100 paired episodes per entry; \\texttt{random / random} is the training "
             "condition and every other cell is out of distribution.")


def tab_radius(rad: pd.DataFrame, out: Path):
    header = ["Arm", "$N$", "inf", "250 m", "0 m", "$\\Delta$ pp", "lost", "gained", "$p$"]
    rows = []
    for _, r in rad.iterrows():
        rows.append([ARM_LABEL[r.arm], str(int(r.N)), _tex_num(r.rate_inf), _tex_num(r.rate_250),
                     _tex_num(r.rate_0), f"${r.delta_pp:+.1f}$", str(int(r.lost)),
                     str(int(r.gained)),
                     (f"{r.p:.1e}" if r.p < 1e-4 else f"{r.p:.4f}") + f"~{stars(r.p)}"])
    return latex_table(
        out / "tab3_comms_range.tex",
        "Effect of restricting the communication range at deployment",
        "tab:comms_range", header, rows,
        note="Success rate averaged over the six deployment cells at each range; "
             "$\\Delta$ pp is $0\\,\\mathrm{m}$ minus \\texttt{inf}. \\emph{lost} / \\emph{gained} "
             "count episodes that flip when the link is cut, over all cells (600 paired "
             "episodes), and $p$ is the exact McNemar test on those. Non-communicating arms "
             "are omitted: they cannot observe the radius and are invariant to it by construction.")


def tab_budget(curves: pd.DataFrame, ns: list[int], out: Path):
    header = ["$N$", "Arm"] + [str(b) for b in BUDGETS]
    rows = []
    for n in ns:
        for arm in ARMS:
            sub = curves[(curves.N == n) & (curves.arm == arm)].set_index("budget")
            if sub.empty:
                continue
            rows.append([str(n), ARM_LABEL[arm]]
                        + [_tex_num(sub.rate.get(b)) for b in BUDGETS])
    return latex_table(
        out / "tab4_success_at_budget.tex",
        "Fraction of episodes solved within a step budget",
        "tab:success_at_budget", header, rows,
        note="Pooled over the six deployment cells at the trained communication radius. "
             "Episodes never solved count as failures at every budget, so the columns are "
             "monotone and free of survivorship bias.")


def tab_time(tm: pd.DataFrame, ns: list[int], out: Path):
    """Two tables: the conditional times, and the penalized/normalized pair."""
    header = ["$N$", "Deployment cell"] + [ARM_LABEL[a] for a in POLICY_ARMS]
    rows = []
    for n in ns:
        for sp, tg in CELLS:
            cells = [str(n), CELL_TEX[(sp, tg)]]
            for arm in POLICY_ARMS:
                s = tm[(tm.N == n) & (tm.spawn == sp) & (tm.target == tg) & (tm.arm == arm)]
                if s.empty or not np.isfinite(s.t_med_cond.iloc[0]):
                    cells.append("--")
                else:
                    r = s.iloc[0]
                    cells.append(f"{r.t_med_cond:.0f} / {r.t_mean_cond:.0f} ({int(r.n_solved)})")
            rows.append(cells)
    a = latex_table(
        out / "tab5a_time_conditional.tex",
        "Time to first success, median / mean over solved episodes (number solved in brackets)",
        "tab:time_conditional", header, rows,
        note="Steps; one step is \\texttt{dt}$\\times$\\texttt{frame\\_skip}. These are "
             "\\emph{conditional} on each arm's own successes and must not be read as a speed "
             "comparison: an arm that solves fewer episodes drops its slow ones and its median "
             "improves for free. Table~\\ref{tab:time_penalized} is the unbiased companion.")

    rows = []
    for n in ns:
        for sp, tg in CELLS:
            cells = [str(n), CELL_TEX[(sp, tg)]]
            for arm in POLICY_ARMS:
                s = tm[(tm.N == n) & (tm.spawn == sp) & (tm.target == tg) & (tm.arm == arm)]
                cells.append("--" if s.empty else
                             f"{s.t_penalized.iloc[0]:.0f} ({s.time_weighted_success.iloc[0]:.2f})")
            rows.append(cells)
    b = latex_table(
        out / "tab5b_time_penalized.tex",
        "Penalized time to first success $E[\\min(t,T)]$, with time-weighted success in brackets",
        "tab:time_penalized", header, rows,
        note="Every episode is counted; an episode never solved is charged the full budget "
             "$T=$\\texttt{max\\_steps}. Lower is better. The bracketed value is the unit-free "
             "form $1-E[\\min(t,T)]/T\\in[0,1]$ (higher is better), which equals the normalized "
             "area under the corresponding success-at-budget curve of "
             "Figure~\\ref{fig:success_at_budget}. Unlike Table~\\ref{tab:time_conditional} this "
             "cannot be improved by failing more often.")
    return a, b


def tab_spawn(sm: pd.DataFrame, shift: pd.DataFrame, sp_pair: pd.DataFrame,
              ns: list[int], out: Path):
    """Section 8: the levels the figure draws, plus the two shifts it cannot label.

    Carries the conditional median as well, since the figure deliberately plots
    only the penalized time -- the reader who wants the familiar "how long did a
    success take" number should not have to open a CSV for it.
    """
    header = (["$N$", "Arm"]
              + [SPAWN_LABEL[sp] + (" (trained)" if sp == TRAIN_SPAWN else "")
                 for sp in SPAWN_ORDER]
              + ["$\\Delta$ origin", "$\\Delta$ max\\_dist",
                 "origin $\\to$ max\\_dist"])
    rows = []
    for n in ns:
        for arm in ARMS:
            cells = [str(n), ARM_LABEL[arm]]
            for sp in SPAWN_ORDER:
                s = sm[(sm.N == n) & (sm.spawn == sp) & (sm.arm == arm)]
                if s.empty:
                    cells.append("--")
                    continue
                r = s.iloc[0]
                med = "--" if not np.isfinite(r.t_med_cond) else f"{r.t_med_cond:.0f}"
                cells.append(f"{r.success_rate:.2f} ({r.t_penalized:.0f} / {med})")
            for sp in ["origin", "max_dist"]:
                d = shift[(shift.N == n) & (shift.arm == arm) & (shift.spawn == sp)]
                cells.append("--" if d.empty else
                             f"${d.iloc[0].delta_pp:+.1f}$~{stars(d.iloc[0].p_bh)}")
            q = sp_pair[(sp_pair.N == n) & (sp_pair.arm == arm)]
            cells.append("--" if q.empty else
                         f"${q.iloc[0].delta_pp:+.1f}$~{stars(q.iloc[0].p_bh)}")
            rows.append(cells)
    return latex_table(
        out / "tab8_spawn_effect.tex",
        "Effect of the spawn geometry on first-arrival success and time",
        "tab:spawn_effect", header, rows,
        note="Each spawn column is the episode success rate, with the penalized time "
             "$E[\\min(t,T)]$ and the conditional median over solved episodes, in steps, in "
             "brackets. Pooled over the two target modes (100 episodes each, so 200 per entry); "
             "the per-target breakdown is in \\texttt{spawn\\_levels.csv}. $\\Delta$ columns are "
             "percentage-point changes in success against the trained \\texttt{random} spawn, by "
             "Fisher's exact test; the last column is \\texttt{origin} against \\texttt{max\\_dist} "
             "by paired exact McNemar. All $p$ values are Benjamini--Hochberg corrected within "
             "their own family. The two families differ because the pairing does: "
             "\\texttt{reset()} draws the flow snapshot and the target \\emph{after} placing the "
             "agents, and only the \\texttt{random} regime consumes randomness to place them, so "
             "\\texttt{origin} and \\texttt{max\\_dist} at a given seed face an identical task and "
             "differ only in where the swarm starts, whereas \\texttt{random} faces a different "
             "one and admits only an unpaired comparison. The last column is therefore the "
             "cleanest measurement on this axis: same field, same target, different geometry. "
             "The \\texttt{random}-policy row is the control -- where it moves, the regime changed "
             "the difficulty of the task rather than the value of the learned behaviour.")


def tab_targets(tmx: pd.DataFrame, tpair: pd.DataFrame, ns: list[int], out: Path):
    """Section 9: the levels behind fig9, plus the paired cost of a rare target."""
    header = (["$N$", "Arm"] + [TARGET_LABEL[tg] for tg in TARGETS]
              + ["$\\Delta$ success", "$\\Delta$ penalized time"])
    rows = []
    for n in ns:
        for arm in ARMS:
            cells = [str(n), ARM_LABEL[arm]]
            vals = {}
            for tg in TARGETS:
                s = tmx[(tmx.N == n) & (tmx.target == tg) & (tmx.arm == arm)]
                if s.empty:
                    cells.append("--")
                    continue
                r = s.iloc[0]
                vals[tg] = r
                med = "--" if not np.isfinite(r.t_med_cond) else f"{r.t_med_cond:.0f}"
                cells.append(f"{r.success_rate:.2f} ({r.t_penalized:.0f} / {med})")
            d = tpair[(tpair.N == n) & (tpair.arm == arm)]
            cells.append("--" if d.empty else
                         f"${d.iloc[0].delta_pp:+.1f}$~{stars(d.iloc[0].p_bh)}")
            cells.append("--" if len(vals) < len(TARGETS) else
                         f"${vals['tail'].t_penalized - vals['random'].t_penalized:+.0f}$")
            rows.append(cells)
    return latex_table(
        out / "tab9_hard_targets.tex",
        "Cost of a rare target: first-arrival success and time under the two target regimes",
        "tab:hard_targets", header, rows,
        note="A \\texttt{random} target is any point in the domain; a \\texttt{tail} target is "
             "drawn from the rarest few per cent of the salinity distribution over its own "
             "depth plane, which shrinks the success zone the swarm has to find. Each target "
             "column is the episode success rate, with the penalized time $E[\\min(t,T)]$ and "
             "the conditional median over solved episodes, in steps, in brackets. Pooled over "
             "the three spawn geometries (100 episodes each, so 300 per entry); the breakdown "
             "is in \\texttt{target\\_levels.csv}. $\\Delta$ success is \\texttt{tail} minus "
             "\\texttt{random} in percentage points, by paired exact McNemar with "
             "Benjamini--Hochberg correction. The pairing is exact here: \\texttt{reset()} fixes "
             "the flow field, the frozen snapshot, the spawn positions and the headings before "
             "it draws the target, so the two regimes differ in the rarity of the target and in "
             "nothing else.")


def tab_scaling(sc: pd.DataFrame, indep: pd.DataFrame, mg: pd.DataFrame,
                ns: list[int], out: Path):
    """Section 10: the levels behind fig10, plus the cost of the extra agents."""
    header = ["$N$", "Arm", "success (pen. / med.)", "cov. red.", "swarm path (m)",
              "NN dist. (m)", "margin", "vs. independent"]
    rows = []
    for n in ns:
        for arm in ARMS:
            s = sc[(sc.N == n) & (sc.arm == arm)]
            if s.empty:
                continue
            r = s.iloc[0]
            med = "--" if not np.isfinite(r.t_med_cond) else f"{r.t_med_cond:.0f}"
            g = mg[(mg.N == n) & (mg.arm == arm)]
            i = indep[(indep.N == n) & (indep.arm == arm)]
            rows.append([
                str(n), ARM_LABEL[arm],
                f"{r.success_rate:.2f} ({r.t_penalized:.0f} / {med})",
                _tex_num(r.cov_red), f"{r.swarm_path:.0f}", f"{r.nn_dist:.0f}",
                "--" if g.empty else f"${g.iloc[0].delta_pp:+.1f}$~{stars(g.iloc[0].p_bh)}",
                "--" if i.empty else f"${i.iloc[0].gap_pp:+.1f}$~{stars(i.iloc[0].p_bh)}"])
    return latex_table(
        out / "tab10_scaling_n.tex",
        "Effect of swarm size on first-arrival success, search overlap and travel cost",
        "tab:scaling_n", header, rows,
        note="Pooled over all six deployment cells at the trained communication radius (600 "
             "episodes per row). Success carries the penalized time $E[\\min(t,T)]$ and the "
             "conditional median over solved episodes, in steps. \\emph{cov.\\ red.} is unique "
             "visited voxels over summed per-agent voxels: $1$ means the agents swept disjoint "
             "water, $1/N$ means they all swept the same. \\emph{swarm path} is the distance "
             "flown by the whole swarm and \\emph{NN dist.}\\ the mean nearest-neighbour "
             "separation at the end of the episode; both are descriptive, and note that an "
             "episode stops at the first arrival, so a faster arm also flies less. "
             "\\emph{margin} is percentage points over the untrained swarm at the same $N$, by "
             "paired exact McNemar. \\emph{vs.\\ independent} is percentage points over "
             "$1-(1-p_{N=2})^{N/2}$, what non-interacting copies of the two-agent swarm would "
             "already achieve; positive means the larger swarm searches better together than "
             "apart. Its interval is a parametric bootstrap over both binomials, since the "
             "prediction is itself estimated. All $p$ values Benjamini--Hochberg corrected "
             "within their own family.")


def tab_arrivals(rt: pd.DataFrame, out: Path):
    """Ranks as COLUMNS: the comparison a reader makes is across ranks within an
    arm ("do this arm's stragglers get there?"), and the long form runs to
    hundreds of rows once N=4 is included."""
    maxrank = int(rt["rank"].max())
    header = ["Scenario", "$N$", "Arm"] + [f"rank {r}" for r in range(1, maxrank + 1)]
    rows = []
    for (scen, n), g in rt.groupby(["scenario", "N"], sort=True):
        for arm in ARMS:
            a = g[g.arm == arm]
            if a.empty:
                continue
            cells = [scen.replace("_", "\\_"), str(int(n)), ARM_LABEL.get(arm, arm)]
            for r in range(1, maxrank + 1):
                x = a[a["rank"] == r]
                cells.append("--" if x.empty else
                             f"{x.reach_rate.iloc[0]:.2f} ({x.t_penalized.iloc[0]:.0f})")
            rows.append(cells)
    return latex_table(
        out / "tab6_arrival_by_rank.tex",
        "Arrival of the $r$-th agent: reach probability and penalized arrival time, by arrival rank",
        "tab:arrival_by_rank", header, rows,
        note="Each entry is the fraction of episodes in which an $r$-th agent ever arrived, "
             "with the penalized arrival time $E[\\min(t,T)]$ in steps in brackets. Evaluated "
             "with \\texttt{success\\_all} semantics: the episode runs on past the first arrival "
             "until every agent has arrived or the budget expires, so rank~1 reproduces "
             "\\texttt{success\\_any} and ranks $2\\ldots N$ are the stragglers this table exists "
             "to measure. An $r$-th arrival that never happens is right-censored at "
             "$T=$\\texttt{max\\_steps} and charged the full budget, so neither column can be "
             "improved by a straggler simply never getting there. Conditional medians and the "
             "unit-free $1-E[\\min(t,T)]/T$ are in \\texttt{arrival\\_by\\_rank.csv}.")


# --------------------------------------------------------------------- main
def parse_kv(values, what):
    out = {}
    for v in values or []:
        if "=" not in v:
            raise SystemExit(f"--{what} expects N=DIR, got '{v}'")
        k, p = v.split("=", 1)
        out[int(k)] = Path(p)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", action="append", metavar="N=DIR", required=True,
                    help="success_any sweep directory for swarm size N (repeatable), "
                         "e.g. --sweep 2=stats/out/sweep_full_100_2N_new")
    ap.add_argument("--success-all", action="store_true",
                    help="also build section 6 from the success_all sweeps. Omit while "
                         "those sweeps are still running: sections 1-5 do not need them.")
    ap.add_argument("--success-all-sweep", action="append", metavar="N=DIR", default=[],
                    help="success_all sweep directory for swarm size N (repeatable)")
    ap.add_argument("--out", type=Path, default=Path("stats/out/thesis_report"))
    ap.add_argument("--assets-dir", type=Path, default=None,
                    help="if given, copy the generated PDFs and .tex tables here "
                         "(e.g. thesis/assets) so the thesis can \\input them directly")
    args = ap.parse_args()

    sweeps = parse_kv(args.sweep, "sweep")
    sa_dirs = parse_kv(args.success_all_sweep, "success-all-sweep")
    # Naming a success_all directory IS the request for section 6; requiring the
    # bare --success-all flag as well only creates a silent no-op.
    if sa_dirs and not args.success_all:
        print("[note] --success-all-sweep given, so section 6 is enabled "
              "(--success-all is implied).")
        args.success_all = True
    ns = sorted(sweeps)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"success_any sweeps: " + ", ".join(f"N={n} <- {p}" for n, p in sweeps.items()))
    df, paired = load_any(sweeps)
    eps = load_episodes(sweeps)
    inv = check_invariants(df, paired)
    print("invariants:")
    for s in inv:
        print(f"  - {s}")

    eff = effects_by_spawn(paired, ns)
    cell_eff = effects_by_cell(paired, ns)
    rad = radius_effects(paired, df, ns)
    curves = budget_curves(eps, ns)
    tm = time_matrix(eps, df, ns)
    sm = spawn_matrix(eps, df, ns)
    shift = spawn_shift(sm, ns)
    sp_pair = spawn_paired(paired, ns)
    tmx = target_matrix(eps, df, ns)
    tg_pair = target_paired(paired, ns)
    sc = scaling_matrix(eps, df, ns)
    indep = scaling_independence(sc, ns)
    mg = scaling_margin(paired, ns)

    df.to_csv(args.out / "cells.csv", index=False)
    eff.to_csv(args.out / "effects_by_spawn.csv", index=False)
    cell_eff.to_csv(args.out / "effects_by_cell.csv", index=False)
    rad.to_csv(args.out / "comms_range.csv", index=False)
    curves.to_csv(args.out / "success_at_budget.csv", index=False)
    tm.to_csv(args.out / "time_to_first_success.csv", index=False)
    sm.to_csv(args.out / "spawn_levels.csv", index=False)
    shift.to_csv(args.out / "spawn_shift.csv", index=False)
    sp_pair.to_csv(args.out / "spawn_paired.csv", index=False)
    tmx.to_csv(args.out / "target_levels.csv", index=False)
    tg_pair.to_csv(args.out / "target_paired.csv", index=False)
    sc.to_csv(args.out / "scaling_levels.csv", index=False)
    indep.to_csv(args.out / "scaling_independence.csv", index=False)
    mg.to_csv(args.out / "scaling_margin.csv", index=False)

    made = [tab_effects(cell_eff, ns, args.out),
            tab_success_matrix(df, ns, args.out),
            tab_radius(rad, args.out),
            tab_budget(curves, ns, args.out),
            *tab_time(tm, ns, args.out),
            tab_spawn(sm, shift, sp_pair, ns, args.out),
            tab_targets(tmx, tg_pair, ns, args.out),
            tab_scaling(sc, indep, mg, ns, args.out),
            fig_effects(cell_eff, ns, args.out),
            fig_success_matrix(df, ns, args.out),
            fig_radius(df, ns, args.out),
            fig_budget(curves, ns, args.out),
            fig_time(tm, ns, args.out),
            fig_spawn(sm, tm, ns, args.out),
            fig_targets(eps, df, tmx, ns, args.out),
            fig_scaling(sc, ns, args.out)]

    if args.success_all:
        sa = sa_dirs or {
            n: Path(str(p).replace("sweep_full_100_", "success_all_").replace("_new", ""))
            for n, p in sweeps.items()}
        print("success_all sweeps: " + ", ".join(f"N={n} <- {p}" for n, p in sa.items()))
        arrivals = load_success_all(sa)
        rk = rank_arrival(arrivals)
        rt = rank_table(rk)
        rk.to_csv(args.out / "arrival_rank_episodes.csv", index=False)
        rt.to_csv(args.out / "arrival_by_rank.csv", index=False)
        made += [tab_arrivals(rt, args.out), *fig_arrivals(rk, rt, args.out)]
    else:
        print("[skip] section 6 (success_all) -- pass --success-all once those sweeps finish.")

    made.append(write_figure_captions(made, args.out))

    if args.assets_dir:
        args.assets_dir.mkdir(parents=True, exist_ok=True)
        for p in made:
            shutil.copy2(p, args.assets_dir / p.name)
        print(f"copied {len(made)} artefact(s) -> {args.assets_dir}")

    print(f"\nwrote {len(made)} thesis artefact(s) + CSVs -> {args.out}")
    for p in made:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()

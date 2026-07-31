"""
Efficiency analysis for the swarm comparison — the lens where a MARL policy can
actually beat N independent copies.

`success_any` over N independent agents is best-of-N draws: it rises with N for
free, saturates (measured: random at N=2 already scores ~60%, PPO at N=4 ~98%),
and structurally cannot reward coordination. Efficiency can. This script reads
the per_episode.csv files written by compare_single_vs_multi.py and answers
"is the jointly-trained policy CHEAPER?" on three metrics:

  * t_first_reach  — steps until the first agent arrives (wall-clock to result)
  * swarm_path     — TOTAL metres flown by ALL agents up to that moment. This is
                     the metric that charges a swarm for its size: 4 agents
                     covering the same water cost 4x the battery. A policy that
                     wins on t_first_reach purely by having more searchers loses
                     here, which is exactly the distinction the thesis needs.
  * finder SPL / path_efficiency — how direct the successful agent's route was.

Every comparison is PAIRED on the episodes BOTH arms solved, and tested with a
Wilcoxon signed-rank test. This matters: a median t_first_reach conditioned on
success is biased when the arms have different success rates (a weak arm only
solves the easy episodes, which flatters its median). Restricting to commonly-
solved episodes removes that bias.

Censoring-free companion: T@k% = the step budget at which an arm first reaches
k% episode success. Defined for every arm regardless of success rate, so it is
the one cross-arm timing number that needs no conditioning caveat.

Usage:
    python stats/efficiency_analysis.py stats/out/scaling_2N_4N/N2
    python stats/efficiency_analysis.py stats/out/scaling_2N_4N/N2 \
                                        stats/out/scaling_2N_4N/N4
"""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

LEVELS = (0.25, 0.50, 0.75, 0.90)


def load_group(d):
    """per_episode.csv -> {arm: {episode: {...}}} with episode-level aggregates."""
    path = os.path.join(d, "per_episode.csv")
    if not os.path.isfile(path):
        raise SystemExit(f"no per_episode.csv in {d}")
    rows = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[r["mode"]][int(r["episode"])].append(r)

    def fnum(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    out = {}
    for arm, eps in rows.items():
        per_ep = {}
        for ep, agents in eps.items():
            succ = [a for a in agents if a["success"].lower() in ("true", "1")]
            # Total distance flown by the WHOLE swarm during the episode. Under
            # the first-reach lens the episode stops at the first arrival, so
            # this is the swarm's cost to produce that result.
            swarm_path = float(np.nansum([fnum(a["path_len"]) for a in agents]))
            if succ:
                finder = min(succ, key=lambda a: fnum(a["steps_to_success"]))
                per_ep[ep] = dict(
                    solved=True,
                    t_any=fnum(finder["steps_to_success"]),
                    swarm_path=swarm_path,
                    finder_spl=fnum(finder["spl"]),
                    finder_path_eff=fnum(finder["path_efficiency"]))
            else:
                per_ep[ep] = dict(solved=False, t_any=float("nan"),
                                  swarm_path=swarm_path, finder_spl=float("nan"),
                                  finder_path_eff=float("nan"))
        out[arm] = per_ep
    return out


def t_at_level(per_ep, level):
    """Step budget at which this arm reaches `level` episode success. Censoring-
    free: uses every episode, solved or not, so arms with different success
    rates stay comparable. None if the arm never reaches the level."""
    n = len(per_ep)
    ts = sorted(v["t_any"] for v in per_ep.values() if v["solved"])
    need = int(np.ceil(level * n))
    return float(ts[need - 1]) if need <= len(ts) else None


def paired(per_a, per_b, key):
    """Values of `key` on the episodes BOTH arms solved."""
    eps = sorted(e for e in per_a
                 if e in per_b and per_a[e]["solved"] and per_b[e]["solved"])
    a = np.array([per_a[e][key] for e in eps], float)
    b = np.array([per_b[e][key] for e in eps], float)
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m], int(m.sum())


def paired_test(a, b):
    """Median paired difference (a - b) + Wilcoxon signed-rank p."""
    if a.size < 6:
        return dict(n=int(a.size), median_diff=float("nan"), p=float("nan"),
                    a_faster=0, b_faster=0)
    d = a - b
    try:
        p = float(wilcoxon(a, b, zero_method="zsplit").pvalue)
    except ValueError:
        p = 1.0
    return dict(n=int(a.size), median_diff=float(np.median(d)), p=p,
                a_faster=int(np.sum(d < 0)), b_faster=int(np.sum(d > 0)))


def print_group(name, groups_dir, data):
    arms = list(data)
    n_ep = len(next(iter(data.values())))
    print("\n" + "=" * 100)
    print(f"EFFICIENCY — {name}   ({n_ep} paired episodes)   <- {groups_dir}")
    print("=" * 100)

    w = max(14, max(len(a) for a in arms) + 2)
    hdr = f"{'metric':<34}" + "".join(f"{a:>{w}}" for a in arms)
    print(hdr); print("-" * len(hdr))

    def row(label, fn):
        print(f"{label:<34}" + "".join(f"{fn(a):>{w}}" for a in arms))

    row("episodes solved", lambda a: f"{sum(v['solved'] for v in data[a].values())}/{n_ep}")
    row("t_first_reach med (solved only)",
        lambda a: _f(np.nanmedian([v["t_any"] for v in data[a].values() if v["solved"]])))
    for lv in LEVELS:
        row(f"T@{int(lv*100)}% success (steps)",
            lambda a, lv=lv: _f(t_at_level(data[a], lv)))
    row("swarm_path med (m, all agents)",
        lambda a: _f(np.nanmedian([v["swarm_path"] for v in data[a].values()])))
    row("finder path_eff med",
        lambda a: _f(np.nanmedian([v["finder_path_eff"] for v in data[a].values()
                                   if v["solved"]]), 3))
    print("-" * len(hdr))
    print("  T@k% is censoring-free (uses all episodes) — the safe cross-arm timing number.")
    print("  t_first_reach med is conditioned on success; compare via the paired tests below.")

    print(f"\nPaired on commonly-solved episodes (Wilcoxon signed-rank):")
    print("-" * 100)
    res = {}
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            a, b = arms[i], arms[j]
            for key, unit in (("t_any", "steps"), ("swarm_path", "m")):
                xa, xb, n = paired(data[a], data[b], key)
                t = paired_test(xa, xb)
                res[f"{a}_vs_{b}__{key}"] = t
                if not np.isfinite(t["median_diff"]):
                    continue
                sig = "**" if t["p"] < 0.05 else ("*" if t["p"] < 0.10 else "ns")
                lead = a if t["median_diff"] < 0 else b
                print(f"  {a:>12} vs {b:<12} {key:<11} n={t['n']:3d}  "
                      f"med Δ={t['median_diff']:+8.1f} {unit:<5} p={t['p']:.4f} {sig:<3} "
                      f"(cheaper: {lead})")
    print("-" * 100)
    print("  Δ = first arm minus second; negative = the FIRST arm is cheaper/faster.")
    print("  ** p<0.05   * p<0.10   ns not significant")
    return res


def _f(v, nd=1):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return f"{v:.{nd}f}"


def main():
    dirs = sys.argv[1:]
    if not dirs:
        raise SystemExit(__doc__)
    out = {}
    for d in dirs:
        name = os.path.basename(os.path.normpath(d))
        data = load_group(d)
        out[name] = print_group(name, d, data)

    if len(dirs) > 1:
        print("\n" + "=" * 100)
        print("CROSS-N NOTE")
        print("=" * 100)
        print("  Groups with different N are NOT paired: OceananigansEnv.reset() draws agent")
        print("  spawns before the snapshot and the target, so a seed yields a different field")
        print("  at each N. Compare each arm to the baseline WITHIN its group, then compare")
        print("  those margins — never an N=2 number directly against an N=4 one.")
        print("  swarm_path is the metric that charges the swarm for its size and is the")
        print("  fairest efficiency comparison across N.")

    dest = os.path.join(os.path.dirname(os.path.normpath(dirs[0])),
                        "efficiency_summary.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {dest}")


if __name__ == "__main__":
    main()

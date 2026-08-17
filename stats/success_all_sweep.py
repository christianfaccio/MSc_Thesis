"""
success_all deployment sweep: how long does the WHOLE swarm take, not just its
first agent?

This is the success_all counterpart of stats/scenario_sweep.py, specialized to
the one question it exists to answer: a policy trained under success_any (the
episode ends at the first arrival) is evaluated with the episode running ON
until EVERY agent has arrived or the step budget expires, so the arrival time of
each individual agent — not only the winner's — is observed and recorded.

That makes it a ZERO-SHOT GENERALIZATION probe. Nothing about the policies is
retrained: only the episode-end rule changes. The mechanism under test is that
a teammate who has already arrived is, to a communication policy, a beacon —
its neighbor block carries the winner's body-frame bearing together with its
salinity error S_j - S*, which is ~0 exactly when it sits in the zone. If the
policy learned to weight motion toward the better-reading neighbor, the
stragglers should home in on it and the comm arms should separate from the
no-comm arms on the LATER arrivals while tying on the first one. (Caveat worth
carrying into the analysis: a frozen agent stops swimming but does not
station-keep, so it drifts with the current and the beacon degrades with time.)

Difference from scenario_sweep.py
---------------------------------
scenario_sweep sweeps spawn x target x comms_radius = 18 conditions under
success_any. Here the target and radius axes are FIXED (random target, radius
as-trained) and only the deployment geometry is swept, over the two spawn modes
that model a real deployment:

    origin     the whole swarm is dropped at (0,0,0) — one boat, one point
    max_dist   spread evenly along the two land walls (West x=0, South y=0) at
               z=0, at maximum separation

The uniform-random spawn is deliberately excluded: it is the training
distribution and has no deployment analogue. Everything else (paired seeds per
scenario, McNemar, Wilson CIs, the per-scenario summary.json / per_episode.csv /
figures) comes from compare_single_vs_multi.run_group unchanged.

Censoring
---------
With a finite budget (--max-steps, default 1800) a straggler can simply never
arrive, so arrival times are RIGHT-CENSORED and a median taken over arrivals
only is optimistically biased — it conditions on having arrived. Every CSV
therefore carries the `reached` / `censored` flags and `max_steps` alongside the
times, which is what a survival analysis (Kaplan-Meier, log-rank) needs. Read
`success_all` (did everyone arrive at all) BEFORE reading `t_last`.

Outputs (all under --out-dir)
-----------------------------
  arrival_times.csv     one row per (scenario, N, arm, episode, AGENT) — the raw
                        per-agent arrival data: reached, arrival_step,
                        arrival_time_s, arrival_rank (1 = first to arrive),
                        censored, plus that agent's path geometry
  episodes.csv          one row per (scenario, N, arm, episode): n_arrived,
                        all_arrived, t_first, t_last, gap (t_last - t_first)
  arrival_by_rank.csv   arrival-time distribution per ARRIVAL RANK: rank 1 is
                        the winner (== the success_any metric), ranks 2..N are
                        the stragglers this script exists to measure
  scenario_summary.csv  one row per (scenario, N, arm): success_any/all with
                        Wilson CIs, mean agents arrived, median t_first/t_last/
                        gap, censoring rate, coverage/nn/swarm-path, and the
                        paired margin over the baseline arm on BOTH lenses
  mcnemar_success_all.csv  every pairwise paired McNemar test per scenario
  success_all_sweep.json   the same content, machine-readable

Usage
-----
    python stats/success_all_sweep.py \
        --policy ppo=runs/ppo_buoyancy_history/checkpoints/latest.pt \
        --policy ippo=runs/ippo_buoyancy_history_2N_new/checkpoints/latest.pt \
        --policy ippo_comm=runs/ippo_buoyancy_history_2N_comm_new/checkpoints/latest.pt \
        --policy mappo_comm=runs/mappo_buoyancy_history_2N_comm_new/checkpoints/latest.pt \
        --netcdf-file data/oceananigans/buoyancy_active/test \
        --n-agents 2 --episodes 100 --seed 0 \
        --out-dir stats/out/success_all_2N --workers 2

    # plan only, roll out nothing
    python stats/success_all_sweep.py --policy ... --dry-run
"""
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import compare_single_vs_multi as cs
import eval_oceananigans_stats as ev

# The two deployment geometries. "random" (the training spawn) is deliberately
# not offered: this sweep is about how the swarm is actually put in the water.
SPAWN_MODES = ("origin", "max_dist")


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", action="append", default=[], metavar="[LABEL=]CKPT",
                   help="a policy arm as LABEL=path/to/checkpoint.pt (repeatable). "
                        "PPO / IPPO / MAPPO are auto-detected; a PPO checkpoint is "
                        "deployed as N independent copies. The FIRST policy supplies "
                        "the common task config.")
    p.add_argument("--spawn-modes", nargs="+", default=list(SPAWN_MODES),
                   choices=list(SPAWN_MODES),
                   help=f"the swept axis (default: {' '.join(SPAWN_MODES)}). The "
                        "uniform-random training spawn is not a choice here.")
    p.add_argument("--target-mode", type=str, default="random",
                   choices=["random", "tail"],
                   help="FIXED for the whole sweep, not an axis (default random). "
                        "'tail' shrinks the success zone to a rare salinity tail.")
    p.add_argument("--target-percentile", type=float, default=None,
                   help="tail width in percent (tail mode only; default: the first "
                        "policy's, typically 5.0)")
    p.add_argument("--comms-radius", type=float, default=None,
                   help="FIXED eval-time communication range in metres, not an axis. "
                        "Default: as trained (inf = global sharing). 0 zeroes the "
                        "neighbor block entirely. Ignored by no-comm policies.")
    p.add_argument("--n-agents", type=int, nargs="+", default=[2], metavar="N",
                   help="agent counts; every scenario runs every N (default 2). A "
                        "communication policy joins ONLY the group matching its "
                        "trained n_agents (its neighbor block is N-specific).")
    p.add_argument("--baseline", type=str, default=None, metavar="LABEL",
                   help="reference arm for the margin columns (default: the first "
                        "PPO arm, i.e. the 'N independent copies' control)")
    p.add_argument("--netcdf-file", type=str, default=None,
                   help="field for every scenario — use the held-out split, e.g. "
                        "data/oceananigans/buoyancy_active/test")
    p.add_argument("--episodes", type=int, default=100,
                   help="paired episodes per scenario (default 100)")
    p.add_argument("--seed", type=int, default=0,
                   help="base seed, SHARED by every scenario so cells stay "
                        "comparable (default 0)")
    p.add_argument("--max-steps", type=int, default=1800,
                   help="step budget per episode (default 1800). This is the "
                        "CENSORING horizon: a straggler that has not arrived by "
                        "then is recorded as censored, not as slow.")
    p.add_argument("--success-steps", type=int, default=1,
                   help="consecutive in-zone steps counted as an arrival (default 1)")
    p.add_argument("--spawn-max-tries", type=int, default=200)
    p.add_argument("--max-cached-loaders", type=int, default=None,
                   help="per-arm NetCDF loader LRU cap (default: as trained, 8). "
                        "Every arm holds its own cache (~90 MB per loader), so peak "
                        "RSS scales with arms x workers x this; lower it to ~2 when "
                        "running several workers.")
    p.add_argument("--mode", default="greedy", choices=["greedy", "stochastic"],
                   help="decode for the trained policies (default greedy = deployment)")
    p.add_argument("--no-random", dest="include_random", action="store_false",
                   default=True,
                   help="omit the uniform-random POLICY arm (the luck floor). Note: "
                        "this is the random baseline policy, unrelated to the "
                        "excluded 'random' spawn mode.")
    p.add_argument("--out-dir", type=str,
                   default=os.path.join(ROOT, "stats", "out", "success_all_sweep"))
    p.add_argument("--workers", type=int, default=1,
                   help="(scenario, N) units evaluated in parallel (default 1). "
                        "Units are independent; with >1 each unit's console output "
                        "goes to <out-dir>/<scenario>__N<N>.log. Keep it <= physical "
                        "cores, and watch RSS (see --max-cached-loaders).")
    p.add_argument("--resume", action="store_true",
                   help="skip units whose summary.json already exists")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and roll out nothing")
    return p.parse_args()


def scenario_name(spawn_mode, target_mode):
    return f"{spawn_mode}__{target_mode}"


def unit_cli(base, spawn_mode, out_dir):
    """A cli namespace for compare_single_vs_multi.run_group. success_all is
    hardwired True — it is the whole point of this driver — and the target and
    comms-radius axes are fixed rather than swept."""
    return SimpleNamespace(
        policy=[], sa_checkpoint=None, ma_checkpoint=None,
        include_random=base.include_random,
        n_agents=base.n_agents, baseline=base.baseline,
        netcdf_file=base.netcdf_file, episodes=base.episodes, seed=base.seed,
        max_steps=base.max_steps,
        target_mode=base.target_mode, target_percentile=base.target_percentile,
        success_steps=base.success_steps, success_all=True,
        spawn_mode=spawn_mode,
        # The fixed spawn modes place agents deterministically; reject-sampling a
        # spawn "far from the zone" on top of that is contradictory.
        min_spawn_distance=0.0, spawn_max_tries=base.spawn_max_tries,
        comms_radius=base.comms_radius, mode=base.mode,
        max_cached_loaders=base.max_cached_loaders,
        out_dir=out_dir)


# --------------------------------------------------------------------------- #
# per-agent arrival extraction
# --------------------------------------------------------------------------- #
def unit_rows(spawn_mode, target_mode, N, g, dt_s, max_steps):
    """Flatten one run_group result into per-agent and per-episode rows.

    The per-agent view is what the episode-level any_ts/all_ts summaries throw
    away: WHICH agent arrived WHEN. Arrival rank is assigned within the episode
    by arrival step (ties broken by agent index), so rank 1 is the success_any
    winner and ranks 2..N are the stragglers."""
    scen = scenario_name(spawn_mode, target_mode)
    agent_rows, episode_rows = [], []
    for arm in g["labels"]:
        by_ep = {}
        for m in g["per_agent_by_label"][arm]:
            by_ep.setdefault(m["_episode"], []).append(m)
        for ep, ms in sorted(by_ep.items()):
            ms.sort(key=lambda m: m["_agent"])
            arrivals = sorted((m["steps_to_success"], m["_agent"])
                              for m in ms if m["success"])
            rank_of = {agent: i + 1 for i, (_, agent) in enumerate(arrivals)}
            n_arrived = len(arrivals)
            all_arrived = bool(arrivals) and n_arrived == len(ms)
            t_first = arrivals[0][0] if arrivals else None
            # t_last is defined ONLY when nobody is censored: the last arrival
            # among a partially-arrived swarm is not the swarm's completion time.
            t_last = arrivals[-1][0] if all_arrived else None
            for m in ms:
                step = m["steps_to_success"]
                agent_rows.append(dict(
                    scenario=scen, spawn_mode=spawn_mode, target_mode=target_mode,
                    n_agents=N, arm=arm, kind=g["kinds"][arm],
                    episode=ep, agent=m["_agent"],
                    reached=int(m["success"]), censored=int(not m["success"]),
                    arrival_step=step,
                    arrival_time_s=(step * dt_s if step is not None else None),
                    arrival_rank=rank_of.get(m["_agent"]),
                    n_arrived=n_arrived, all_arrived=int(all_arrived),
                    max_steps=max_steps,
                    spawn_dist=m["spawn_dist"], path_len=m["path_len"],
                    spl=m["spl"], path_efficiency=m["path_efficiency"],
                    min_dist=m["min_dist"], tortuosity=m["tortuosity"],
                    monotonic_frac=m["monotonic_frac"],
                    final_dS=m["final_dS"], final_dtau=m["final_dtau"],
                    nc_file=m["nc_file"]))
            episode_rows.append(dict(
                scenario=scen, spawn_mode=spawn_mode, target_mode=target_mode,
                n_agents=N, arm=arm, kind=g["kinds"][arm], episode=ep,
                success_any=int(n_arrived > 0), all_arrived=int(all_arrived),
                n_arrived=n_arrived, n_censored=len(ms) - n_arrived,
                t_first=t_first,
                t_first_s=(t_first * dt_s if t_first is not None else None),
                t_last=t_last,
                t_last_s=(t_last * dt_s if t_last is not None else None),
                # The straggler penalty: how much longer the swarm needed after
                # its first agent got there. Only defined when all arrived.
                gap_steps=(t_last - t_first if t_last is not None else None),
                gap_s=((t_last - t_first) * dt_s if t_last is not None else None),
                max_steps=max_steps))
    return agent_rows, episode_rows


def run_unit(spawn_mode, N, specs, base_dict, ckpt_args, out_root, quiet):
    """Evaluate ONE (spawn_mode, N) unit. Runs in a worker process when
    --workers > 1, so it takes only picklable arguments and reloads the task
    checkpoint itself."""
    base = SimpleNamespace(**base_dict)
    if quiet:
        # One worker per unit, each running tiny MLPs on a batch of n_agents:
        # torch's default intra-op pool would put ~8 threads in every worker and
        # oversubscribe the machine, costing more than the parallelism gains.
        torch.set_num_threads(1)
    device = torch.device("cpu")
    src_raw = torch.load(specs[0][1], map_location=device, weights_only=False)
    dt_s = float(src_raw["args"]["dt"]) * float(src_raw["args"]["frame_skip"])
    max_steps = base.max_steps or int(src_raw["args"]["max_steps"])

    scen = scenario_name(spawn_mode, base.target_mode)
    out_dir = os.path.join(out_root, scen)
    os.makedirs(out_dir, exist_ok=True)
    cli = unit_cli(base, spawn_mode, out_dir)

    sub = cs.group_specs(specs, N, ckpt_args)
    if not sub:
        return [], [], [], 0.0

    t0 = time.time()

    def _work():
        g = cs.run_group(N, sub, cli, device, src_raw, base.seed)
        a_rows, e_rows = unit_rows(spawn_mode, base.target_mode, N, g, dt_s, max_steps)
        return g, a_rows, e_rows

    if quiet:
        log = os.path.join(out_root, f"{scen}__N{N}.log")
        with open(log, "w") as f, redirect_stdout(f), redirect_stderr(f):
            g, a_rows, e_rows = _work()
    else:
        g, a_rows, e_rows = _work()

    summary_rows, mcnemar_rows = summarize_unit(
        spawn_mode, base.target_mode, N, g, e_rows, a_rows, base.baseline, dt_s)
    return a_rows, e_rows, (summary_rows, mcnemar_rows), time.time() - t0


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def _median(vals):
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
    return float(np.median(a)) if a.size else None


def _mean(vals):
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
    return float(a.mean()) if a.size else None


def pick_baseline(g, explicit):
    """The arm every other arm is measured against: the 'N independent copies'
    control (single-agent PPO deployed as N agents), unless overridden."""
    if explicit is not None:
        return explicit if explicit in g["labels"] else None
    for l in g["pol_labels"]:
        if g["kinds"][l] == "ppo":
            return l
    return g["pol_labels"][0] if g["pol_labels"] else None


def summarize_unit(spawn_mode, target_mode, N, g, e_rows, a_rows, baseline, dt_s):
    """One summary row per arm + the pairwise McNemar rows, for this unit."""
    scen = scenario_name(spawn_mode, target_mode)
    base = pick_baseline(g, baseline)
    n_ep = len(g["any_by_label"][g["labels"][0]])
    summary_rows = []
    for arm in g["labels"]:
        er = [r for r in e_rows if r["arm"] == arm]
        ar = [r for r in a_rows if r["arm"] == arm]
        st = g["ep_stats"][arm]
        k_any = int(sum(g["any_by_label"][arm]))
        k_all = int(sum(g["all_by_label"][arm]))
        any_lo, any_hi = ev.wilson_ci(k_any, n_ep)
        all_lo, all_hi = ev.wilson_ci(k_all, n_ep)
        margin_any = margin_all = None
        if base and arm != base:
            margin_any = cs.paired_diff_ci(g["any_by_label"][arm],
                                           g["any_by_label"][base])
            margin_all = cs.paired_diff_ci(g["all_by_label"][arm],
                                           g["all_by_label"][base])
        row = dict(
            scenario=scen, spawn_mode=spawn_mode, target_mode=target_mode,
            n_agents=N, arm=arm, kind=g["kinds"][arm], episodes=n_ep,
            success_any=k_any / n_ep, success_any_lo=any_lo, success_any_hi=any_hi,
            success_all=k_all / n_ep, success_all_lo=all_lo, success_all_hi=all_hi,
            per_agent_success=g["results"][arm]["success_rate"],
            mean_agents_arrived=_mean([r["n_arrived"] for r in er]),
            # Fraction of agent-episodes whose arrival time is right-censored by
            # the step budget — the number that makes the medians below readable.
            censored_frac=_mean([r["censored"] for r in ar]),
            t_first_med=_median([r["t_first"] for r in er]),
            t_first_med_s=_median([r["t_first_s"] for r in er]),
            t_last_med=_median([r["t_last"] for r in er]),
            t_last_med_s=_median([r["t_last_s"] for r in er]),
            gap_med=_median([r["gap_steps"] for r in er]),
            gap_med_s=_median([r["gap_s"] for r in er]),
            spl=g["results"][arm]["spl_mean"],
            coverage_redundancy=st.get("coverage_redundancy"),
            nn_distance=st.get("nn_distance"), swarm_path=st.get("swarm_path"),
            baseline=base,
            margin_any_pp=(margin_any["diff"] * 100 if margin_any else None),
            margin_any_lo=(margin_any["lo"] * 100 if margin_any else None),
            margin_any_hi=(margin_any["hi"] * 100 if margin_any else None),
            margin_all_pp=(margin_all["diff"] * 100 if margin_all else None),
            margin_all_lo=(margin_all["lo"] * 100 if margin_all else None),
            margin_all_hi=(margin_all["hi"] * 100 if margin_all else None),
        )
        # Median arrival time per rank, conditioned on that many agents arriving.
        for r in range(1, N + 1):
            times = [x["arrival_step"] for x in ar if x["arrival_rank"] == r]
            row[f"t_rank{r}_med"] = _median(times)
            row[f"n_rank{r}"] = len(times)
        summary_rows.append(row)

    mcnemar_rows = []
    for lens, flags in (("success_any", g["any_by_label"]),
                        ("success_all", g["all_by_label"])):
        for i in range(len(g["labels"])):
            for j in range(i + 1, len(g["labels"])):
                a, b = g["labels"][i], g["labels"][j]
                xa = np.asarray(flags[a], bool)
                xb = np.asarray(flags[b], bool)
                a_only = int(np.sum(xa & ~xb))
                b_only = int(np.sum(xb & ~xa))
                d = cs.paired_diff_ci(flags[a], flags[b])
                mcnemar_rows.append(dict(
                    scenario=scen, n_agents=N, lens=lens, arm_a=a, arm_b=b,
                    a_only=a_only, b_only=b_only,
                    p_value=cs.mcnemar_exact(a_only, b_only),
                    lead=(a if a_only > b_only else (b if b_only > a_only else "tie")),
                    diff_pp=d["diff"] * 100, diff_lo=d["lo"] * 100,
                    diff_hi=d["hi"] * 100))
    return summary_rows, mcnemar_rows


def rank_table(agent_rows):
    """Arrival-time distribution per (scenario, N, arm, rank). Rank 1 is the
    success_any winner; ranks 2..N are the stragglers. Each rank is conditioned
    on at least that many agents having arrived, so `n_episodes` shrinks with
    rank exactly as much as the censoring does — read them together."""
    keyed = {}
    for r in agent_rows:
        if r["arrival_rank"] is None:
            continue
        k = (r["scenario"], r["n_agents"], r["arm"], r["kind"], r["arrival_rank"])
        keyed.setdefault(k, []).append(r)
    out = []
    for (scen, n, arm, kind, rank), rows in sorted(keyed.items()):
        steps = np.array([x["arrival_step"] for x in rows], float)
        secs = np.array([x["arrival_time_s"] for x in rows], float)
        out.append(dict(
            scenario=scen, n_agents=n, arm=arm, kind=kind, arrival_rank=rank,
            n_episodes=len(rows),
            steps_mean=float(steps.mean()), steps_median=float(np.median(steps)),
            steps_std=float(steps.std()),
            steps_p25=float(np.percentile(steps, 25)),
            steps_p75=float(np.percentile(steps, 75)),
            steps_min=float(steps.min()), steps_max=float(steps.max()),
            seconds_median=float(np.median(secs))))
    return out


def write_csv(path, rows, fields=None):
    if not rows:
        return
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


# --------------------------------------------------------------------------- #
# console reporting
# --------------------------------------------------------------------------- #
def _fmt(v, kind="num"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    if kind == "pct":
        return f"{v * 100:.1f}"
    if kind == "int":
        return f"{v:.0f}"
    return f"{v:.3f}"


def print_unit_tables(summary_rows, N, scenarios):
    rows = [r for r in summary_rows if r["n_agents"] == N]
    if not rows:
        return
    arms = []
    for r in rows:
        if r["arm"] not in arms:
            arms.append(r["arm"])
    by = {(r["arm"], r["scenario"]): r for r in rows}
    lab = max(16, max(len(a) for a in arms) + 2)
    w = max(18, max(len(s) for s in scenarios) + 2)
    total = lab + w * len(scenarios)

    def table(title, note, field, kind):
        print("\n" + "=" * total)
        print(title)
        if note:
            print(note)
        print("=" * total)
        print(("{:<%d}" % lab).format("arm")
              + "".join(("{:>%d}" % w).format(s) for s in scenarios))
        print("-" * total)
        for a in arms:
            line = ("{:<%d}" % lab).format(a)
            for s in scenarios:
                r = by.get((a, s))
                line += ("{:>%d}" % w).format(_fmt(r[field] if r else None, kind))
            print(line)
        print("=" * total)

    print("\n\n" + "#" * total)
    print(f"# SUCCESS_ALL SWEEP — N={N}")
    print("#" * total)
    table(f"success_any (%)  N={N}",
          "at least one agent arrived — the first-reach lens, usually saturated",
          "success_any", "pct")
    table(f"success_all (%)  N={N}",
          "EVERY agent arrived within the budget — the metric this sweep exists for",
          "success_all", "pct")
    table(f"agents arrived (mean of {N})  N={N}",
          "the graded version of success_all; less brittle than the all-or-nothing rate",
          "mean_agents_arrived", "num")
    table(f"censored agent-episodes (fraction)  N={N}",
          "arrival time cut off by the step budget; read this BEFORE any median below",
          "censored_frac", "num")
    table(f"t_first (median steps)  N={N}",
          "first arrival, over episodes with any arrival",
          "t_first_med", "int")
    table(f"t_last (median steps)  N={N}",
          "LAST arrival, over fully-arrived episodes only — the swarm completion time",
          "t_last_med", "int")
    table(f"gap = t_last - t_first (median steps)  N={N}",
          "the straggler penalty: extra time after the winner arrived",
          "gap_med", "int")
    for r_i in range(1, N + 1):
        table(f"arrival #{r_i} (median steps)  N={N}",
              f"rank {r_i} of {N}, conditioned on at least {r_i} agent(s) arriving",
              f"t_rank{r_i}_med", "int")
    table(f"margin over '{rows[0]['baseline']}' on success_all (pp)  N={N}",
          "paired within each scenario; see scenario_summary.csv for the CI",
          "margin_all_pp", "num")
    table(f"coverage redundancy  N={N}",
          "1.0 = disjoint search, 1/N = everyone swept the same water",
          "coverage_redundancy", "num")


# --------------------------------------------------------------------------- #
def main():
    cli = parse_args()
    specs = cs.collect_policy_specs(SimpleNamespace(
        policy=cli.policy, sa_checkpoint=None, ma_checkpoint=None))
    ns = sorted(set(int(n) for n in cli.n_agents))
    os.makedirs(cli.out_dir, exist_ok=True)

    device = torch.device("cpu")
    ckpt_args = {}
    for _, path in specs:
        c = torch.load(path, map_location=device, weights_only=False)
        if "netcdf_file" not in c["args"]:
            raise SystemExit(f"'{path}' is not an Oceananigans checkpoint.")
        ckpt_args[path] = c["args"]
        del c

    scenarios = [scenario_name(s, cli.target_mode) for s in cli.spawn_modes]
    units = [(s, n) for s in cli.spawn_modes for n in ns]
    if cli.resume:
        keep = [(s, n) for s, n in units
                if not os.path.exists(os.path.join(
                    cli.out_dir, scenario_name(s, cli.target_mode),
                    f"N{n}", "summary.json"))]
        if len(units) - len(keep):
            print(f"[resume] skipping {len(units) - len(keep)} completed unit(s)")
        units = keep

    # ---- plan ---------------------------------------------------------- #
    per_n = {n: cs.group_specs(specs, n, ckpt_args) for n in ns}
    rollouts = sum(len(per_n[n]) + (1 if cli.include_random else 0)
                   for _, n in units)
    print("=" * 78)
    print(f"SUCCESS_ALL SWEEP — {len(cli.spawn_modes)} spawn mode(s) x "
          f"{len(ns)} agent count(s) = {len(units)} unit(s)")
    print("=" * 78)
    print(f"  episode end  : success_ALL (runs past the first arrival)")
    print(f"  spawn axis   : {' '.join(cli.spawn_modes)}   (uniform-random spawn excluded)")
    print(f"  target       : {cli.target_mode} (fixed)")
    print(f"  comms radius : {'as trained (inf)' if cli.comms_radius is None else cli.comms_radius} (fixed)")
    print(f"  max_steps    : {cli.max_steps}   <- the censoring horizon")
    for n in ns:
        cm = [os.path.basename(os.path.dirname(os.path.dirname(p)))
              for _, p in per_n[n] if ckpt_args[p].get("communication", False)]
        print(f"  N={n}: {len(per_n[n])} policy arm(s)"
              f"{' + random' if cli.include_random else ''}, {len(cm)} with comms")
        if not per_n[n]:
            print(f"        (no policy can be deployed at N={n})")
    print(f"  episodes     : {cli.episodes} paired, base seed {cli.seed} (shared)")
    print(f"  rollouts     : ~{rollouts} arm-units x {cli.episodes} episodes")
    print(f"  decode       : {cli.mode}")
    print(f"  workers      : {cli.workers}")
    print(f"  out          : {cli.out_dir}")
    if any(n == 1 for n in ns):
        print("  NOTE: at N=1 success_all == success_any (nothing to straggle).")
    if cli.dry_run:
        print("\n[dry-run] nothing rolled out.")
        return
    if not units:
        print("\nnothing to do.")
        return

    base_dict = vars(cli)
    quiet = cli.workers > 1
    t0 = time.time()
    agent_rows, episode_rows, summary_rows, mcnemar_rows = [], [], [], []

    def _collect(res):
        a_rows, e_rows, summaries, _ = res
        agent_rows.extend(a_rows)
        episode_rows.extend(e_rows)
        if summaries:
            summary_rows.extend(summaries[0])
            mcnemar_rows.extend(summaries[1])

    if cli.workers > 1:
        print(f"\nrunning {cli.workers} unit(s) in parallel; per-unit output "
              f"-> <out-dir>/<scenario>__N<N>.log\n")
        with ProcessPoolExecutor(max_workers=cli.workers) as pool:
            futs = {pool.submit(run_unit, s, n, specs, base_dict, ckpt_args,
                                cli.out_dir, True): (s, n) for s, n in units}
            for i, fut in enumerate(as_completed(futs), 1):
                s, n = futs[fut]
                res = fut.result()
                _collect(res)
                print(f"  [{i}/{len(units)}] {s:<10} N={n}  {res[3] / 60:6.1f} min")
    else:
        for i, (s, n) in enumerate(units, 1):
            print(f"\n>>> [{i}/{len(units)}] spawn '{s}', N={n}")
            _collect(run_unit(s, n, specs, base_dict, ckpt_args, cli.out_dir, False))

    total_min = (time.time() - t0) / 60

    # ---- write the CSVs ------------------------------------------------- #
    ranks = rank_table(agent_rows)
    write_csv(os.path.join(cli.out_dir, "arrival_times.csv"), agent_rows)
    write_csv(os.path.join(cli.out_dir, "episodes.csv"), episode_rows)
    write_csv(os.path.join(cli.out_dir, "arrival_by_rank.csv"), ranks)
    write_csv(os.path.join(cli.out_dir, "scenario_summary.csv"), summary_rows)
    write_csv(os.path.join(cli.out_dir, "mcnemar_success_all.csv"), mcnemar_rows)

    for n in ns:
        print_unit_tables(summary_rows, n, scenarios)

    with open(os.path.join(cli.out_dir, "success_all_sweep.json"), "w") as f:
        json.dump(dict(
            generated=datetime.now().isoformat(timespec="seconds"),
            episodes=cli.episodes, base_seed=cli.seed, decode=cli.mode,
            max_steps=cli.max_steps, success_steps=cli.success_steps,
            success_all=True, netcdf_file=cli.netcdf_file,
            spawn_modes=list(cli.spawn_modes), target_mode=cli.target_mode,
            comms_radius=cli.comms_radius, n_agents=ns,
            scenarios=scenarios, elapsed_min=total_min,
            summary=summary_rows, by_rank=ranks, mcnemar=mcnemar_rows),
            f, indent=2, default=float)

    print(f"\nSweep finished in {total_min:.1f} min.")
    print(f"Wrote arrival_times.csv, episodes.csv, arrival_by_rank.csv,")
    print(f"      scenario_summary.csv, mcnemar_success_all.csv, "
          f"success_all_sweep.json -> {cli.out_dir}")
    print("Per-scenario detail (tables, McNemar, figures) is under "
          "<out-dir>/<scenario>/N<N>/.")
    print("\nArrival times are RIGHT-CENSORED at max_steps "
          f"({cli.max_steps} steps): every CSV carries `reached`/`censored`, so "
          "prefer a survival estimate (Kaplan-Meier) over a plain mean.")


if __name__ == "__main__":
    main()

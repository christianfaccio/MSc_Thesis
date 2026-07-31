"""
Run ONE set of policies across MANY evaluation scenarios in a single invocation,
and report a cross-scenario matrix.

Motivation: on the training task (random target, random spawn, first-reach lens)
the 2N/4N policies sit at 88-100% greedy success_any on the held-out split. That
metric is SATURATED — with 100 episodes, "98.3% vs 100%" is two episodes and
McNemar returns `ns` for almost every pair. So the interesting question is not
"who wins on the training task" but "under which DEPLOYMENT CONDITIONS does CTDE
(MAPPO vs IPPO) or communication (comm vs no-comm) actually buy something?".

This driver answers that by sweeping the scenario axes and reporting each arm's
margin over a baseline in every cell, so a claim like "communication only pays
when the swarm is deployed from a single point" becomes a row you can read.

It reuses compare_single_vs_multi.run_group verbatim, so every scenario also
produces the usual per-scenario summary.json / per_episode.csv / paired_episodes.csv
/ figures/ under <out-dir>/<scenario>/N<N>/.

The grid
--------
  spawn_mode    random | origin | max_dist    where the swarm starts
  target_mode   random | tail                 how rare the success zone is
  comms_radius  inf | 250 | 0  [metres]       eval-time communication range

3 x 2 x 3 = 18 conditions. The first two axes define the TASK (a "cell"); the
third is an observation-side ablation that only a communication policy can
react to, since a no-comm policy has no neighbor block in its observation at
all. So the radius axis is nested inside each cell and the cost is asymmetric:

  comm policies      18 evaluated conditions each
  no-comm + random    6 rollouts each, reused across the radius axis

The reuse is exact rather than an approximation — within a cell the task, the
episode seeds and the greedy decode are all identical, so a radius-invariant
arm would produce byte-identical episodes. All 18 output directories still
contain a full table with every arm and every McNemar test.

The radius axis is what turns "is communication used?" into a testable
question: these policies all trained at radius inf, so if success is flat from
inf to 0 — where the neighbor block is entirely zeroed — the communication
channel is decorative, whatever the comm-vs-no-comm comparison shows.

Usage
-----
    python stats/scenario_sweep.py \
        --policy ppo=runs/ppo_buoyancy_history/checkpoints/latest.pt \
        --policy ippo=runs/ippo_buoyancy_history_2N/checkpoints/latest.pt \
        --policy ippo_comm=runs/ippo_buoyancy_history_2N_comm/checkpoints/latest.pt \
        --policy mappo=runs/mappo_buoyancy_history_2N/checkpoints/latest.pt \
        --policy mappo_comm=runs/mappo_buoyancy_history_2N_comm/checkpoints/latest.pt \
        --netcdf-file data/oceananigans/buoyancy_active/test \
        --n-agents 2 --episodes 100 --seed 0 \
        --out-dir stats/out/sweep_2N --workers 3

    # see the plan and the cost split without rolling anything out
    python stats/scenario_sweep.py --policy ... --dry-run
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


# --------------------------------------------------------------------------- #
# scenario grid
# --------------------------------------------------------------------------- #
SPAWN_MODES = ("random", "origin", "max_dist")
TARGET_MODES = ("random", "tail")
COMMS_RADII = (None, 250.0, 0.0)      # None = as trained (inf)

# The grid is spawn x target x comms_radius. The first two axes define the TASK
# (a "cell"); the third is an observation-side ablation that only a
# communication policy can react to, so it is nested INSIDE the cell rather than
# treated as a third independent task axis:
#
#   comm policies      -> 3 x 2 x 3 = 18 evaluated conditions
#   no-comm + random   -> 3 x 2     =  6 rollouts, reused across the radius axis
#
# The reuse is exact, not an approximation: within a cell the task, the seeds and
# the greedy decode are identical, and a policy with no neighbor block in its
# observation cannot see comms_radius at all. Every one of the 18 output
# directories still contains a full table with all arms and all McNemar tests.


def radius_tag(r):
    return "inf" if r is None else f"{int(r)}"


def build_cells(spawn_modes, target_modes):
    """The 6 task cells. Radii are expanded inside run_scenario so the no-comm
    arms can be rolled out once per cell and shared across the radius axis."""
    return [dict(name=f"{s}__{t}", spawn_mode=s, target_mode=t)
            for s in spawn_modes for t in target_modes]


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", action="append", default=[], metavar="[LABEL=]CKPT",
                   help="a policy arm, as in compare_single_vs_multi (repeatable). "
                        "The FIRST supplies the common task config.")
    p.add_argument("--spawn-modes", nargs="+", default=list(SPAWN_MODES),
                   choices=list(SPAWN_MODES),
                   help=f"spawn axis (default: {' '.join(SPAWN_MODES)})")
    p.add_argument("--target-modes", nargs="+", default=list(TARGET_MODES),
                   choices=list(TARGET_MODES),
                   help=f"target axis (default: {' '.join(TARGET_MODES)})")
    p.add_argument("--comms-radii", nargs="+", default=None, metavar="R",
                   help="eval-time communication range axis, in metres; 'inf' means "
                        "as-trained (default: inf 250 0). Applied ONLY to policies "
                        "trained with communication — no-comm arms are rolled out "
                        "once per task cell and reused across this axis. Pass a "
                        "single value to disable the axis.")
    p.add_argument("--only", type=str, default=None,
                   help="comma-separated task-cell names to keep (substring match "
                        "on '<spawn>__<target>')")
    p.add_argument("--n-agents", type=int, nargs="+", default=[2], metavar="N",
                   help="agent counts; every scenario runs every N (default 2)")
    p.add_argument("--baseline", type=str, default=None, metavar="LABEL",
                   help="reference arm for the margin matrix (default: the first "
                        "PPO arm, i.e. the 'N independent copies' control)")
    p.add_argument("--netcdf-file", type=str, default=None,
                   help="default field for every scenario that does not override it")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0,
                   help="base seed, SHARED by every scenario so cells stay "
                        "comparable (default 0)")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--success-steps", type=int, default=1)
    p.add_argument("--spawn-max-tries", type=int, default=200)
    p.add_argument("--max-cached-loaders", type=int, default=None,
                   help="per-arm NetCDF loader LRU cap (default: as trained, 8). "
                        "Each arm holds its own cache, so peak RSS scales with "
                        "arms x this; lower it to ~2 to fit more --workers in RAM.")
    p.add_argument("--target-percentile", type=float, default=None)
    p.add_argument("--mode", default="greedy", choices=["greedy", "stochastic"])
    p.add_argument("--no-random", dest="include_random", action="store_false",
                   default=True)
    p.add_argument("--no-end-on-any-success", dest="end_on_any_success",
                   action="store_false", default=True,
                   help="swarm-truth lens for the WHOLE sweep: episodes run until "
                        "every agent arrives, de-censoring success_all. Under the "
                        "default first-reach lens success_any is best-of-N and "
                        "success_all is ~0 by construction.")
    p.add_argument("--out-dir", type=str,
                   default=os.path.join(ROOT, "stats", "out", "sweep"))
    p.add_argument("--workers", type=int, default=1,
                   help="scenarios evaluated in parallel (default 1). Scenarios are "
                        "independent, so this scales nearly linearly; with >1 each "
                        "scenario's console output goes to <scenario>/run.log instead "
                        "of the terminal. Keep it <= physical cores.")
    p.add_argument("--resume", action="store_true",
                   help="skip scenarios whose output already has every N group")
    p.add_argument("--dry-run", action="store_true",
                   help="print the scenario plan and cost estimate, roll out nothing")
    return p.parse_args()


def parse_radii(values):
    if values is None:
        return list(COMMS_RADII)
    out = []
    for v in values:
        s = str(v).strip().lower()
        out.append(None if s in ("inf", "none", "trained") else float(s))
    return out


def build_cells_filtered(cli):
    cells = build_cells(cli.spawn_modes, cli.target_modes)
    if cli.only:
        keys = [k.strip() for k in cli.only.split(",") if k.strip()]
        cells = [c for c in cells if any(k in c["name"] for k in keys)]
    if not cells:
        raise SystemExit("no task cells selected.")
    return cells


def scenario_cli(base, scenario, out_dir):
    """A cli namespace for compare_single_vs_multi.run_group: the base settings
    with this scenario's overrides applied."""
    c = SimpleNamespace(
        policy=[], sa_checkpoint=None, ma_checkpoint=None,
        include_random=base.include_random, n_agents=base.n_agents,
        baseline=base.baseline, netcdf_file=base.netcdf_file,
        episodes=base.episodes, seed=base.seed, max_steps=base.max_steps,
        target_mode=None, target_percentile=base.target_percentile,
        success_steps=base.success_steps, spawn_mode=None,
        min_spawn_distance=0.0, spawn_max_tries=base.spawn_max_tries,
        comms_radius=None, mode=base.mode,
        max_cached_loaders=base.max_cached_loaders,
        end_on_any_success=base.end_on_any_success, out_dir=out_dir)
    for key, val in scenario.items():
        if key == "name":
            continue
        if not hasattr(c, key):
            raise SystemExit(f"scenario '{scenario['name']}': unknown key '{key}'")
        setattr(c, key, val)
    return c


def scenario_done(out_dir, ns):
    return all(os.path.exists(os.path.join(out_dir, f"N{n}", "summary.json"))
               for n in ns)


# --------------------------------------------------------------------------- #
def _strip(name, scenario, elapsed, groups):
    """Reduce a run_group result to the picklable, matrix-relevant subset."""
    return dict(
        name=name, scenario=scenario, elapsed_s=elapsed,
        groups={str(n): dict(labels=g["labels"], pol_labels=g["pol_labels"],
                             kinds=g["kinds"], ep_stats=g["ep_stats"],
                             any_by_label=g["any_by_label"],
                             all_by_label=g["all_by_label"],
                             any_ts=g["any_ts"], out_dir=g["out_dir"],
                             spl={l: g["results"][l]["spl_mean"] for l in g["labels"]})
                for n, g in groups.items()})


def run_cell(cell, specs, base_dict, ckpt_args, out_root, quiet):
    """Evaluate ONE task cell (spawn x target) across every comms radius and every
    N group, and return one result per radius.

    The radius axis is handled here rather than by the caller so that a single
    rollout cache can be shared across it: no-comm policies and the random arm
    are rolled out once for the cell and reused for every radius, which is what
    keeps the cost at 18 comm conditions + 6 no-comm rollouts instead of 18 of
    each. Runs in a worker process when --workers > 1, so it takes only picklable
    arguments and re-loads the task checkpoint itself."""
    base = SimpleNamespace(**base_dict)
    if quiet:
        # One worker per cell, each running tiny MLPs on a batch of n_agents:
        # torch's default intra-op pool would put ~8 threads in every worker and
        # oversubscribe the machine several times over, which costs more than the
        # parallelism gains. One thread per worker is strictly better here.
        torch.set_num_threads(1)
    radii = parse_radii(base.comms_radii)
    ns = sorted(set(int(x) for x in base.n_agents))
    device = torch.device("cpu")
    src_raw = torch.load(specs[0][1], map_location=device, weights_only=False)

    # Which N groups actually contain a communication policy? For the others the
    # radius axis is a no-op, so only the first (as-trained) radius is run.
    comm_ns = {n for n in ns
               if any(ckpt_args[p].get("communication", False)
                      for _, p in cs.group_specs(specs, n, ckpt_args))}

    def _work():
        cache = {}                     # shared across radii within this cell
        out = []
        for r in radii:
            name = f"{cell['name']}__radius-{radius_tag(r)}"
            scenario = dict(cell, name=name, comms_radius=r)
            out_dir = os.path.join(out_root, name)
            os.makedirs(out_dir, exist_ok=True)
            cli = scenario_cli(base, scenario, out_dir)
            t0 = time.time()
            groups = {}
            for n in ns:
                if r is not radii[0] and n not in comm_ns:
                    continue           # radius is a no-op for this N group
                sub = cs.group_specs(specs, n, ckpt_args)
                if not sub:
                    continue
                groups[n] = cs.run_group(n, sub, cli, device, src_raw, base.seed,
                                         cache=cache)
            if groups:
                out.append(_strip(name, scenario, time.time() - t0, groups))
        return out

    t0 = time.time()
    if quiet:
        os.makedirs(out_root, exist_ok=True)
        log = os.path.join(out_root, f"{cell['name']}.log")
        with open(log, "w") as f, redirect_stdout(f), redirect_stderr(f):
            results = _work()
    else:
        results = _work()
    for r in results:
        r["cell"] = cell["name"]
    return results, time.time() - t0


# --------------------------------------------------------------------------- #
# cross-scenario reporting
# --------------------------------------------------------------------------- #
def _pick_baseline(group, explicit):
    if explicit is not None:
        return explicit if explicit in group["labels"] else None
    for l in group["pol_labels"]:
        if group["kinds"][l] == "ppo":
            return l
    return group["pol_labels"][0]


def matrix(results, n, baseline_label):
    """rows = arm, cols = scenario. Returns (arms, scen_names, cells) where a
    cell holds every reported quantity for that (arm, scenario) pair."""
    names = [r["name"] for r in results if str(n) in r["groups"]]
    arms = []
    for r in results:
        g = r["groups"].get(str(n))
        if g:
            for l in g["labels"]:
                if l not in arms:
                    arms.append(l)
    cells = {}
    for r in results:
        g = r["groups"].get(str(n))
        if not g:
            continue
        base = _pick_baseline(g, baseline_label)
        for l in g["labels"]:
            st = g["ep_stats"][l]
            margin = None
            if base and base in g["labels"] and l != base:
                margin = cs.paired_diff_ci(g["any_by_label"][l],
                                           g["any_by_label"][base])
            cells[(l, r["name"])] = dict(
                any_rate=st["any_rate"], all_rate=st["all_rate"],
                t_any=cs.median_t_any(g["any_ts"][l]),
                coverage_redundancy=st.get("coverage_redundancy"),
                swarm_path=st.get("swarm_path"), spl=g["spl"][l],
                margin=margin, baseline=base)
    return arms, names, cells


def _cell(v, kind):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    if kind == "pct":
        return f"{v * 100:.1f}"
    if kind == "margin":
        return f"{v['diff'] * 100:+.1f}[{v['lo'] * 100:+.0f},{v['hi'] * 100:+.0f}]"
    if kind == "int":
        return f"{v:.0f}"
    return f"{v:.3f}"


def print_matrix(title, arms, names, cells, field, kind, note=""):
    w = max(20, max((len(n) for n in names), default=10) + 2)
    lab = max(16, max((len(a) for a in arms), default=8) + 2)
    total = lab + w * len(names)
    print("\n" + "=" * total)
    print(title)
    if note:
        print(note)
    print("=" * total)
    print(("{:<%d}" % lab).format("arm") + "".join(("{:>%d}" % w).format(n) for n in names))
    print("-" * total)
    for a in arms:
        row = ("{:<%d}" % lab).format(a)
        for n in names:
            c = cells.get((a, n))
            row += ("{:>%d}" % w).format(_cell(c[field] if c else None, kind))
        print(row)
    print("=" * total)


def write_matrix_csv(path, arms, names, cells, n):
    fields = ["any_rate", "all_rate", "t_any", "coverage_redundancy",
              "swarm_path", "spl"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_agents", "arm", "scenario", "baseline",
                                          "margin_pp", "margin_lo", "margin_hi"] + fields)
        w.writeheader()
        for a in arms:
            for s in names:
                c = cells.get((a, s))
                if not c:
                    continue
                m = c["margin"]
                row = dict(n_agents=n, arm=a, scenario=s, baseline=c["baseline"],
                           margin_pp=(m["diff"] * 100 if m else None),
                           margin_lo=(m["lo"] * 100 if m else None),
                           margin_hi=(m["hi"] * 100 if m else None))
                row.update({k: c[k] for k in fields})
                w.writerow(row)


# --------------------------------------------------------------------------- #
def main():
    cli = parse_args()
    cells = build_cells_filtered(cli)
    radii = parse_radii(cli.comms_radii)
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

    n_comm = sum(1 for _, p in specs if ckpt_args[p].get("communication", False))
    if n_comm == 0 and len(radii) > 1:
        print("[no communication policy among the arms — collapsing the radius "
              "axis to 'inf']")
        radii = radii[:1]
        cli.comms_radii = ["inf"]

    scen_names = [f"{c['name']}__radius-{radius_tag(r)}" for c in cells for r in radii]
    pending = cells
    if cli.resume:
        pending = [c for c in cells
                   if not all(scenario_done(
                       os.path.join(cli.out_dir, f"{c['name']}__radius-{radius_tag(r)}"),
                       ns) for r in radii)]
        if len(cells) - len(pending):
            print(f"[resume] skipping {len(cells) - len(pending)} completed cell(s)")

    # ---- plan ---------------------------------------------------------- #
    per_n = {n: cs.group_specs(specs, n, ckpt_args) for n in ns}
    comm_rollouts = nocomm_rollouts = 0
    for n, sub in per_n.items():
        c = sum(1 for _, p in sub if ckpt_args[p].get("communication", False))
        nc = len(sub) - c + (1 if cli.include_random else 0)
        comm_rollouts += c * len(radii) * len(pending)
        nocomm_rollouts += nc * len(pending)
    total = (comm_rollouts + nocomm_rollouts) * cli.episodes

    print("=" * 78)
    print(f"SCENARIO SWEEP — {len(pending)} task cell(s) x {len(radii)} comms "
          f"radius(es) = {len(pending) * len(radii)} conditions")
    print("=" * 78)
    print(f"  spawn x target : {' '.join(cli.spawn_modes)}  x  "
          f"{' '.join(cli.target_modes)}")
    print(f"  comms radii    : {', '.join(radius_tag(r) for r in radii)} "
          f"(comm policies only)")
    for c in pending:
        print(f"    {c['name']}")
    print("-" * 78)
    for n in ns:
        sub = per_n[n]
        cm = [os.path.basename(os.path.dirname(os.path.dirname(p)))
              for _, p in sub if ckpt_args[p].get("communication", False)]
        print(f"  N={n}: {len(sub)} policy arm(s)"
              f"{' + random' if cli.include_random else ''}, "
              f"{len(cm)} with comms")
    print(f"  episodes     : {cli.episodes} paired, base seed {cli.seed} (shared)")
    print(f"  rollouts     : {comm_rollouts} comm-arm x {cli.episodes} + "
          f"{nocomm_rollouts} no-comm x {cli.episodes} = ~{total}")
    print(f"                 (no-comm arms are rolled out once per cell and "
          f"reused across radii)")
    print(f"  workers      : {cli.workers}")
    print(f"  out          : {cli.out_dir}")
    if cli.dry_run:
        print("\n[dry-run] nothing rolled out.")
        return
    if not pending:
        print("\nnothing to do.")
        return

    base_dict = vars(cli)
    quiet = cli.workers > 1
    t0 = time.time()
    results = []
    if cli.workers > 1:
        print(f"\nrunning {cli.workers} cells in parallel; per-cell output "
              f"-> <out-dir>/<cell>.log\n")
        with ProcessPoolExecutor(max_workers=cli.workers) as pool:
            futs = {pool.submit(run_cell, c, specs, base_dict, ckpt_args,
                                cli.out_dir, True): c for c in pending}
            for i, fut in enumerate(as_completed(futs), 1):
                res, elapsed = fut.result()
                results.extend(res)
                print(f"  [{i}/{len(pending)}] {res[0]['cell'] if res else '?':<24} "
                      f"{len(res)} condition(s)  {elapsed / 60:6.1f} min")
    else:
        for i, c in enumerate(pending, 1):
            print(f"\n>>> [{i}/{len(pending)}] cell '{c['name']}'")
            res, _ = run_cell(c, specs, base_dict, ckpt_args, cli.out_dir, False)
            results.extend(res)

    order = {n: i for i, n in enumerate(scen_names)}
    results.sort(key=lambda r: order.get(r["name"], 1 << 30))
    total_min = (time.time() - t0) / 60

    # ---- cross-scenario matrices --------------------------------------- #
    payload = dict(generated=datetime.now().isoformat(timespec="seconds"),
                   episodes=cli.episodes, base_seed=cli.seed, decode=cli.mode,
                   n_agents=ns, elapsed_min=total_min,
                   end_on_any_success=cli.end_on_any_success,
                   spawn_modes=list(cli.spawn_modes),
                   target_modes=list(cli.target_modes),
                   comms_radii=[radius_tag(r) for r in radii],
                   cells=[c["name"] for c in pending],
                   scenarios=[r["name"] for r in results], per_n={})
    for n in ns:
        arms, names, cells = matrix(results, n, cli.baseline)
        if not names:
            continue
        base = next((cells[(a, names[0])]["baseline"] for a in arms
                     if (a, names[0]) in cells), None)
        print("\n\n" + "#" * 78)
        print(f"# CROSS-SCENARIO MATRIX — N={n}")
        print("#" * 78)
        print_matrix(f"success_any (%)  N={n}", arms, names, cells,
                     "any_rate", "pct",
                     note="the headline metric; saturates on easy cells")
        print_matrix(f"margin over '{base}' (percentage points, 95% CI)  N={n}",
                     arms, names, cells, "margin", "margin",
                     note="paired within each cell. A CI spanning 0 = no detectable edge.")
        print_matrix(f"success_all (%)  N={n}", arms, names, cells,
                     "all_rate", "pct",
                     note="only meaningful in swarm-truth cells (lens preset); "
                          "censored to ~0 under the first-reach lens")
        print_matrix(f"t_first_reach (median steps, successes only)  N={n}",
                     arms, names, cells, "t_any", "int",
                     note="trajectory efficiency; compare only between arms with "
                          "similar success_any in the same cell")
        print_matrix(f"coverage redundancy  N={n}", arms, names, cells,
                     "coverage_redundancy", "num",
                     note="1.0 = perfectly disjoint search, 1/N = everyone swept "
                          "the same water. What comms/CTDE should move.")
        print_matrix(f"swarm path (m, all agents)  N={n}", arms, names, cells,
                     "swarm_path", "int",
                     note="total metres flown by the whole swarm — charges a swarm "
                          "for its size, unlike per-agent SPL")
        write_matrix_csv(os.path.join(cli.out_dir, f"sweep_matrix_N{n}.csv"),
                         arms, names, cells, n)
        payload["per_n"][str(n)] = dict(
            arms=arms, scenarios=names, baseline=base,
            cells={f"{a}||{s}": cells[(a, s)] for a in arms for s in names
                   if (a, s) in cells})

    with open(os.path.join(cli.out_dir, "sweep_summary.json"), "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"\nSweep finished in {total_min:.1f} min.")
    print(f"Wrote sweep_summary.json + sweep_matrix_N*.csv -> {cli.out_dir}")
    print("Per-scenario detail (tables, McNemar, figures) is under "
          "<out-dir>/<scenario>/N<N>/.")


if __name__ == "__main__":
    main()

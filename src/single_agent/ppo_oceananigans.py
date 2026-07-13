'''
Usage (from root):
    python -m src.single_agent.ppo_oceananigans
    tensorboard --logdir runs --port 6006
'''
import random
import time
from collections import deque
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from rich.console import Console
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn,
)
from torch.utils.tensorboard import SummaryWriter

from src.single_agent.policy import PpoPolicy
from src.envs.env_pool import AsyncEnvPool, SyncEnvPool
from src.envs.oceananigans_factory import (
    make_raw_env_from_cfg,
    make_wrapped_single_env_from_cfg,
)

DEBUG = True
console = Console()
STATS_WINDOW = 100


# --- Normalization-state checkpoint helpers -------------------------------------
# Implemented through the vector-env get_attr/set_attr API (which resolves the
# attribute through each env's wrapper stack), NOT by walking envs.envs: with
# AsyncVectorEnv the per-env wrappers live in worker processes and envs.envs
# does not exist. get_attr/set_attr work identically on SyncVectorEnv.
def get_obs_rms_state(envs):
    return [{
        "mean": np.asarray(rms.mean).copy(),
        "var": np.asarray(rms.var).copy(),
        "count": float(rms.count),
    } for rms in envs.get_attr("obs_rms")]


def set_obs_rms_state(envs, states):
    rms_objs = list(envs.get_attr("obs_rms"))
    for rms, state in zip(rms_objs, states):
        rms.mean = np.asarray(state["mean"]).copy()
        rms.var = np.asarray(state["var"]).copy()
        rms.count = state["count"]
    envs.set_attr("obs_rms", rms_objs)


def get_return_rms_state(envs):
    return [{
        "mean": float(rms.mean),
        "var": float(rms.var),
        "count": float(rms.count),
    } for rms in envs.get_attr("return_rms")]


def set_return_rms_state(envs, states):
    rms_objs = list(envs.get_attr("return_rms"))
    for rms, state in zip(rms_objs, states):
        rms.mean = np.array(state["mean"])
        rms.var = np.array(state["var"])
        rms.count = state["count"]
    envs.set_attr("return_rms", rms_objs)


@dataclass
class Args:
    exp_name: str = "ppo_oceananigans"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances"""

    # Environment arguments 
    env_id: str = "OceananigansSingleAgent-ppo"
    """the id of the environment"""
    xml_file: str = "config/simulation.xml"
    """SwarmSwIM simulation XML"""
    netcdf_file: str = "data/oceananigans/buoyancy_active"
    """Oceananigans NetCDF source: a directory (all *.nc), a glob, or a single file.
    Each reset draws a random file and (static mode) a random snapshot within it, so
    every episode sees a different frozen field sampled from the whole set. Default is
    the buoyancy-active dataset (salinity filaments -> real local minima); the passive
    baseline lives at data/oceananigans/no_buoyancy. NOTE: epsilon_salinity / sigma_s
    below are sized to the field's per-snapshot span — switch them together with the
    dataset (buoyancy_active ~5 PSU: 0.15 / 1.5; no_buoyancy ~10 PSU: 0.3 / 3.0)."""
    epsilon_salinity: float = 0.15
    """success tolerance on |S - S*| (PSU), ~3% of the per-snapshot span so the relative
    difficulty of the success test matches the validated no_buoyancy runs (0.3 on the
    ~10 PSU passive span -> 0.15 on the ~4.9 PSU median buoyancy-active span, measured
    2026-07-10 across the dataset). Also sizes the narrow shaping kernels (1.5x/5x eps)
    and the trivial-target rejection (2x eps) in the env."""
    epsilon_turbidity: float = 0.05
    """success tolerance on |τ - τ*| — unchanged: τ depends only on depth and the
    domain/turbidity model are identical across the two datasets."""
    sigma_s: float = 1.5
    """wide shaping-kernel width in S (PSU), ~0.3x the field span (validated rule from
    the no_buoyancy runs: 3.0 on ~10 PSU). On the ~5 PSU buoyancy-active span 3.0 would
    cover most of the field and flatten the far-field guidance."""
    sigma_tau: float = 0.3
    """wide shaping-kernel width in τ — unchanged (depth parity, see reward_func)."""
    max_cached_loaders: int = 8
    """per-env LRU cap on cached FieldLoaders (~90 MB each). Memory ≈
    num_envs · max_cached_loaders · 90 MB (12·8 ≈ 8.6 GB). Raise on big-RAM machines
    to cut file re-opens; lower if memory-constrained."""
    k: int = 12
    """observation history depth: last k (action direction, ΔS, Δτ) tuples appended to
    the 12-dim sensor frame (obs = 12 + 5k values; incl. dead-reckoned displacement from spawn). On the buoyancy-active filament
    fields the gradient's basin of attraction covers only ~40% of the plane
    (2026-07-11 analysis), so a memoryless gradient-follower is insufficient — the
    history is what enables dead-reckoned escape from filament local optima."""
    target_mode: str = "random"
    """'random' = target (S*, τ*) read at a uniform random field point (typical S* ->
    the |ΔS|<ε zone covers ~10-20% of the plane at z*, large luck floor: a depth-only
    drifting baseline scores ~0.5); 'tail' = S* from a rare tail of the salinity
    distribution over the target's own depth plane — LOW or HIGH side drawn 50/50
    per episode (3D rarity does NOT work: the fields are depth-stratified, so a
    3D-rare S* can still cover much of its plane), shrinking the zone to a rare
    filament so success requires actual navigation (2026-06-29 meeting scenario).
    Spawn stays uniform."""
    target_percentile: float = 5.0
    """tail mode only: tail width in percent — S* below this percentile (low side)
    or above 100 minus it (high side) of the salinity values on its depth plane
    (Monte Carlo estimate, 256 plane points per reset)"""
    reward_potential: str = "error"
    """shaping potential Φ: 'error' = Gaussian over the (ΔS, Δτ) measurement error
    (agent-sensible, but every filament with S ≈ S* is a reward local optimum);
    'distance' = 1 − d/diag with d the distance to the nearest success-zone cell
    of the episode's snapshot — monotone toward the zone, NO local optima.
    Training-time privileged info: feeds only the reward, never the observation;
    potential-based shaping keeps the optimal policy identical (Ng et al. 1999)."""
    v_agent: float = 1.0
    """agent commanded speed (m/s)"""
    max_steps: int = 1440
    """maximum env steps per episode before truncation. One env step ≈ 1 m of travel
    (v_agent·dt·frame_skip = 1 m), the domain is 1 km and targets spawn ~0.3·diagonal
    ≈ 425 m away, so 1440 steps (~24 min sim) is ~3.4× the optimal path — enough slack
    without burning compute on 7200-step failed episodes. Also matches the γ=0.999
    effective horizon (~1000 steps). 7200 (run 1783528628) let the STOCHASTIC policy
    rack up ~0.5 'success' by pure diffusion (7.2 km of travel vs a zone covering
    ~5.5% of the xy-plane) while the greedy policy sat in no-op/ping-pong loops — a
    gradient-following oracle needs only ~350 steps, so 1440 keeps diffusion from
    masquerading as navigation."""
    dt: float = 0.1
    """simulator timestep (s) per sim sub-step"""
    domain: tuple[float, float, float] = (1000.0, 1000.0, 100.0)
    """domain extent in (x, y, z) meters"""
    frame_skip: int = 10
    """sim sub-steps per env step; one env step = dt·frame_skip = 1 s of sim time,
    so distance per step ≈ v_agent·dt·frame_skip = 1 m (1000 m domain -> ~1000 steps
    to cross; targets spawn ≥30% of the diagonal away)"""
    success_bonus: float = 10.0
    """reward bonus on reaching the target zone (shaped potential otherwise)"""
    static_frame: bool = True 
    """Either static or dynamic mode"""
    min_band_grad: float = 0.004
    """reject targets whose success band is ~flat (median |grad_xy S| < this, PSU/m) at
    reset, so every episode has a local gradient to home on; <=0 disables the guard"""
    target_min_dist_frac: float = 0.0
    """minimum spawn→target distance as a fraction of the domain diagonal. 0 = no
    distance check, so targets may land close to the spawn — episode difficulty then
    varies (some easy, near-target episodes), giving a stuck sparse-reward policy
    denser success signal to bootstrap from. Raise (e.g. 0.3) to force far targets."""
    wall_penalty: float = 0.05
    """per-step reward penalty for pinning against a domain wall, scaled by the
    fraction of the step's frame_skip ticks that were clamped. Discourages the
    degenerate 'drive into a boundary and stall' local optimum. 0 disables."""
    success_steps_required: int = 3
    """consecutive in-zone steps required to count as success. 1 (run 1783508432) let
    a single lucky in-zone step terminate: on the turbulent LES field the STOCHASTIC
    policy wiggles across the thin |ΔS|<ε band and clips it by chance, which (a)
    inflates train success, (b) makes success DECAY as entropy anneals down and the
    wiggle vanishes (the ~5M-step regression), and (c) never reproduces under a greedy
    rollout (agent parks just outside the band). Requiring 3 consecutive steps (=30 s
    of dwell at dt·frame_skip=10 s) forces the agent to arrive AND hold, so the metric
    is honest and matches rollouts. Holding is feasible: no-op action + monotonic depth
    + ~0.003 m/s currents on the frozen field."""

    # Algorithm specific arguments
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 3.0e-4
    """the learning rate of the optimizer"""
    num_envs: int = 12
    """the number of parallel environments"""
    async_envs: bool = True
    """step the parallel envs in worker processes (AsyncVectorEnv, spawn context)
    instead of a single-core loop. The env step is the wall-clock bottleneck
    (~3 ms of SwarmSwIM+scipy per env-step vs a negligible policy forward), so this
    is a ~num_envs× rollout speedup up to the core count, with IDENTICAL training
    semantics (same batch, same autoreset mode, per-env RNG unchanged). Workers
    build envs from the torch-free factory (src/envs/oceananigans_factory.py), so
    each costs ~an env's memory, not a torch import. --no-async-envs restores the
    single-process SyncVectorEnv for debugging."""
    num_steps: int = 512
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """OFF for this field. When ON (run 1783459789) lr decayed to 0 by the end and
    FROZE the policy while it was still improving: success PEAKED 0.69 @7.5M then
    regressed to 0.53 @10M as lr/kl/clipfrac all went to 0. The task is still learning
    at 7.5M, so a full lr→0 decay by 10M throws away the best policy. Base runs never
    annealed lr either. Keep a flat lr and let the entropy anneal do the late-stage
    sharpening instead."""
    gamma: float = 0.999
    """discount factor; effective horizon 1/(1-γ) = 1000 steps ≈ 1000 m, matched to
    the ~1 m/step, up-to-1280-step episodes. MUST equal the env's γ for the
    potential-based shaping to stay policy-invariant (passed to the env below)."""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 12
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.25
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function"""
    ent_coef: float = 0.01
    """starting entropy coefficient. The turbulent LES salinity is deceptive, so the
    early failure mode is premature lock-in (wall-pinning / circling) — 0.01 keeps a
    high exploration floor at the start. But held CONSTANT (run 1783449635) entropy
    plateaus at ~2.79 of a 3.30 max (eff. ~16/27 actions, near-uniform): the policy
    never commits and success caps at ~0.40. So anneal DOWN to a nonzero floor to let
    it sharpen late without the run 1783417603 collapse-to-0 lock-in."""
    anneal_ent: bool = True
    """ON: anneal ent_coef → ent_coef_final over the FIRST `ent_anneal_frac` of
    training, then HOLD the floor. For this field we anneal to a NONZERO floor (not 0)
    so late-stage commitment coexists with a residual exploration bonus — full
    anneal-to-0 collapsed entropy to 0.62 and locked in at 37%."""
    ent_anneal_frac: float = 0.5
    """fraction of training over which ent_coef anneals from ent_coef → ent_coef_final;
    after that it HOLDS the floor. The old linear-over-full-run schedule (run
    1783499930) kept ent_coef ~0.007 at 3.5M so entropy stayed ~2.4 and the policy
    never reached a low-entropy COMMIT phase — base spent its whole 2nd half at
    entropy ~1.0 refining 0.48→0.76. 0.5 = explore hard early (escape the deceptive
    local optima that locked run-1 at 37%), then give the back half at the 0.001 floor
    to commit + refine like base. Lower toward 0.3 if it still won't commit; raise
    toward 0.7 if entropy collapses too early and locks in."""
    ent_coef_final: float = 0.0
    """entropy coefficient at end of training (anneal_ent on). 0.001 (run 1783528628)
    still left final entropy at 1.47 of 3.30 — the 'policy' was a biased random walk
    whose train success came from stochastic diffusion, and its greedy argmax
    collapsed to no-op/ping-pong (2026-07-09 diagnosis). Anneal fully to 0 so the
    back half of training must commit; the greedy_success_rate eval (below) is the
    honest metric to watch for lock-in. (The earlier anneal-to-0 lock-in at 37%,
    run 1783417603, predates the frame fix in the env's _measure and the shorter
    max_steps, so its caveat no longer binds.)"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.02
    """the target KL divergence threshold"""

    # Greedy evaluation (the honest metric — see 2026-07-09 diagnosis: the training
    # success_rate measures the STOCHASTIC policy, which can score ~0.5 by diffusion
    # alone; deployment/plot_trajectories.py uses greedy argmax)
    eval_every_iterations: int = 50
    """run a deterministic (argmax) evaluation every N iterations; 0 disables.
    Cost: eval_episodes × up to max_steps env steps on one env (~worst case a couple
    of minutes per eval at 1440 steps), amortized over ~11 min of training."""
    eval_episodes: int = 20
    """greedy episodes per evaluation. Fixed seeds (reused every eval) so the
    logged charts/greedy_success_rate is comparable across the run. 20 keeps the
    binomial noise at ~±0.11 (4 episodes gave ±0.25 — unreadable trends)."""
    eval_workers: int = 4
    """parallel worker envs for the greedy evaluation (episodes fanned out in
    waves; greedy + fixed seeds, so the metric is identical to a sequential eval).
    20 sequential episodes × up to 1440 steps would otherwise serialize ~half the
    wall-clock of the 50 (parallel) training iterations between evals. Each worker
    holds its own FieldLoader cache (~90 MB per cached file), so keep this modest
    on small-RAM machines. 1 = sequential (previous behavior)."""

    # Checkpointing
    save_model: bool = True
    """if toggled, periodically save model + optimizer + RNG state checkpoints"""
    save_every_iterations: int = 20
    """save a checkpoint every N PPO iterations (and always on the final iteration)"""
    checkpoint_dir: str = "runs"
    """parent directory for checkpoints"""
    resume: str = None
    """path to a checkpoint .pt file to resume training from"""

    # to be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


# Args fields forwarded to the OceananigansEnv constructor, extracted to a plain
# dict so worker processes can unpickle the env factory without importing this
# (torch-heavy) module — see src/envs/oceananigans_factory.py.
ENV_CFG_KEYS = (
    "xml_file", "netcdf_file", "k", "v_agent", "max_steps", "dt", "domain",
    "frame_skip", "gamma", "success_bonus", "static_frame", "min_band_grad",
    "target_min_dist_frac", "wall_penalty", "success_steps_required",
    "max_cached_loaders", "epsilon_salinity", "epsilon_turbidity",
    "sigma_s", "sigma_tau", "target_mode", "target_percentile",
    "reward_potential",
)


def env_cfg(args) -> dict:
    """Plain-dict env configuration (picklable without this module)."""
    return {key: getattr(args, key) for key in ENV_CFG_KEYS}


def make_raw_env(args):
    """Bare OceananigansEnv (n_agents=1) with the training configuration (no gym wrappers)."""
    return make_raw_env_from_cfg(env_cfg(args))


def combine_obs_rms(states):
    """Count-weighted merge of the per-env NormalizeObservation RMS states
    (same helper as scripts/plot_trajectories.py) -> (mean, var)."""
    counts = np.array([s["count"] for s in states], dtype=np.float64)
    w = counts / counts.sum()
    mean = sum(wi * np.asarray(s["mean"], np.float64) for wi, s in zip(w, states))
    var = sum(wi * np.asarray(s["var"], np.float64) for wi, s in zip(w, states))
    return mean, var


def greedy_eval(agent, eval_pool, obs_rms_states, device, n_episodes, max_steps,
                base_seed=1_000_000):
    """Deterministic (argmax) rollouts on raw envs with the current training
    obs normalization applied — the same conditions as plot_trajectories.py.
    Episodes are fanned out in waves across the eval pool's workers; greedy
    actions + fixed per-episode seeds make the result identical to a sequential
    eval, only parallel. Returns the success rate under the TRAINING bar
    (env's success_steps_required)."""
    mean, var = combine_obs_rms(obs_rms_states)
    successes = 0
    was_training = agent.training
    agent.eval()
    with torch.no_grad():
        for wave_start in range(0, n_episodes, eval_pool.num_envs):
            episodes = range(wave_start, min(wave_start + eval_pool.num_envs, n_episodes))
            workers = list(range(len(episodes)))
            results = eval_pool.reset_where(
                workers, seeds=[base_seed + ep for ep in episodes])
            obs = {w: res[0] for w, res in zip(workers, results)}
            active = workers
            for _ in range(max_steps):
                if not active:
                    break
                stack = np.stack([np.asarray(obs[w]) for w in active])
                norm = np.clip((stack - mean) / np.sqrt(var + 1e-8),
                               -10.0, 10.0).astype(np.float32)
                acts = agent.actor(torch.tensor(norm, device=device)).argmax(dim=-1).cpu().numpy()
                results = eval_pool.step_where(active, [int(a) for a in acts])
                still_active = []
                for w, (o, _, term, trunc, _) in zip(active, results):
                    obs[w] = o
                    if term:
                        successes += 1
                    elif not trunc:
                        still_active.append(w)
                active = still_active
    if was_training:
        agent.train()
    return successes / n_episodes


def train(args):
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if DEBUG:
        print("--- INFO ---\n")
        print(f"Run name: {run_name}\nBatch size: {args.batch_size}\n"
              f"Minibatch size: {args.minibatch_size}\nIterations: {args.num_iterations}\n")

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name, entity=args.wandb_entity,
            sync_tensorboard=True, config=vars(args), name=run_name,
            monitor_gym=True, save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if DEBUG:
        print(f"Device: {device}")
        print("--- Setting up the environment...")
    # SAME_STEP autoreset restores the classic CleanRL semantics this loop assumes:
    # when an episode ends, next_obs is already the NEW episode's reset obs and the
    # true terminal obs arrives in infos["final_obs"]. Gymnasium 1.x's default
    # NEXT_STEP mode would instead hand back the old episode's final obs, pair it
    # with an ignored action and a 0 reward on the following step, polluting the
    # on-policy batch with one bogus transition per episode.
    # Async vs sync only changes WHERE envs step (worker processes vs this one):
    # same wrapper stack, same autoreset mode, same per-env seeding. The factory
    # partials pickle by reference to the torch-free factory module, so spawn
    # workers never import this module (or torch).
    cfg = env_cfg(args)
    env_fns = [partial(make_wrapped_single_env_from_cfg, cfg)
               for _ in range(args.num_envs)]
    if args.async_envs:
        envs = gym.vector.AsyncVectorEnv(
            env_fns,
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
            context="spawn",  # fork is unsafe on macOS and with threaded parents
        )
    else:
        envs = gym.vector.SyncVectorEnv(
            env_fns,
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
        )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), \
        "only discrete action space is supported"

    # Dedicated raw-env pool (no wrappers) for the periodic greedy evaluation;
    # created once so each worker's FieldLoader LRU cache persists across evals.
    # The eval seeds are fixed, so each worker only ever touches its own episodes'
    # files — cap its loader cache accordingly instead of args.max_cached_loaders.
    eval_pool = None
    if args.eval_every_iterations > 0:
        n_workers = max(1, min(args.eval_workers, args.eval_episodes))
        eval_cfg = dict(cfg)
        episodes_per_worker = -(-args.eval_episodes // n_workers)  # ceil
        eval_cfg["max_cached_loaders"] = min(args.max_cached_loaders,
                                             max(2, episodes_per_worker))
        eval_fns = [partial(make_raw_env_from_cfg, eval_cfg) for _ in range(n_workers)]
        pool_cls = AsyncEnvPool if (args.async_envs and n_workers > 1) else SyncEnvPool
        eval_pool = pool_cls(eval_fns)

    agent = PpoPolicy(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    start_iteration = 1
    global_step = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        agent.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_iteration = ckpt["iteration"] + 1
        global_step = ckpt["global_step"]
        torch.set_rng_state(ckpt["torch_rng"])
        np.random.set_state(ckpt["np_rng"])
        random.setstate(ckpt["py_rng"])
        if "obs_rms" in ckpt:
            set_obs_rms_state(envs, ckpt["obs_rms"])
        if "return_rms" in ckpt:
            set_return_rms_state(envs, ckpt["return_rms"])
        if DEBUG:
            print(f"Resumed from {args.resume}: iteration={start_iteration}, global_step={global_step}")

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)

    if DEBUG:
        print("--- GAME START ---")
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    ep_returns = deque(maxlen=STATS_WINDOW)
    ep_lengths = deque(maxlen=STATS_WINDOW)
    ep_terminated = deque(maxlen=STATS_WINDOW)  # 1.0 if reached target, 0.0 if truncated

    progress = Progress(
        TextColumn("[bold blue]iter"), MofNCompleteColumn(), BarColumn(),
        TextColumn("ret={task.fields[ret]:>6.2f}  len={task.fields[len]:>5.1f}  "
                   "term={task.fields[term]:>3.0f}%  eps={task.fields[eps]:>4d}  "
                   "SPS={task.fields[sps]:>5d}"),
        TextColumn("•"), TimeElapsedColumn(), TextColumn("<"), TimeRemainingColumn(),
        console=console, refresh_per_second=4,
    )
    task_id = progress.add_task(
        "train", total=args.num_iterations, completed=start_iteration - 1,
        ret=float("nan"), len=float("nan"), term=0.0, eps=0, sps=0,
    )

    progress.start()
    for iteration in range(start_iteration, args.num_iterations + 1):
        frac = 1.0 - (iteration - 1.0) / args.num_iterations
        if args.anneal_lr:
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate
        if args.anneal_ent:
            # anneal ent_coef -> ent_coef_final over the first ent_anneal_frac of
            # training, then HOLD the floor (explore early, commit late).
            train_frac = (iteration - 1.0) / args.num_iterations
            a = min(train_frac / max(args.ent_anneal_frac, 1e-8), 1.0)
            ent_coef_now = args.ent_coef + a * (args.ent_coef_final - args.ent_coef)
        else:
            ent_coef_now = args.ent_coef

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, _ = agent.get_action_and_value(next_obs)
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)

            # Truncation is not termination: the potential-based shaping relies on
            # bootstrapping from the truncated state's value (the env keeps the real
            # Φ(s') there for exactly this reason). With SAME_STEP autoreset next_obs
            # is already the NEW episode's reset obs, so fold the bootstrap into the
            # reward using the recorded final observation (already normalized/clipped
            # by the wrapper stack). done=1 then correctly stops GAE at this step.
            if truncations.any() and "final_obs" in infos:
                final_obs = np.stack([np.asarray(o, dtype=np.float32)
                                      for o in infos["final_obs"][truncations]])
                with torch.no_grad():
                    final_v = agent.get_value(torch.Tensor(final_obs).to(device)).view(-1)
                rewards[step][torch.as_tensor(truncations, device=device)] += args.gamma * final_v

            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(next_done).to(device)

            # SAME_STEP vector format: episode stats arrive as dict-of-arrays under
            # infos["final_info"]["episode"], with the "_episode" boolean mask
            # marking which envs actually finished this step.
            if "final_info" in infos and "episode" in infos["final_info"]:
                fin = infos["final_info"]
                ep_stats = fin["episode"]
                for i in np.where(fin["_episode"])[0]:
                    ep_returns.append(float(ep_stats["r"][i]))
                    ep_lengths.append(float(ep_stats["l"][i]))
                    succ = 1.0 if bool(terminations[i]) else 0.0
                    ep_terminated.append(succ)
                    writer.add_scalar("charts/episodic_return", float(ep_stats["r"][i]), global_step)
                    writer.add_scalar("charts/episodic_length", float(ep_stats["l"][i]), global_step)
                    writer.add_scalar("charts/episode_success", succ, global_step)

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            with torch.no_grad():
                new_values = agent.get_value(b_obs).view(args.num_steps, args.num_envs)
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = new_values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - new_values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + new_values
                b_advantages = advantages.reshape(-1)
                b_returns = returns.reshape(-1)
                b_values = new_values.reshape(-1)

            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef_now * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/ent_coef", ent_coef_now, global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)
        if ep_terminated:
            writer.add_scalar("charts/success_rate", float(np.mean(ep_terminated)), global_step)

        # Periodic deterministic evaluation — charts/success_rate above tracks the
        # STOCHASTIC policy, which can score ~0.5 by diffusion alone (2026-07-09
        # diagnosis); greedy argmax is what plot_trajectories.py and deployment use.
        if eval_pool is not None and (iteration % args.eval_every_iterations == 0
                                      or iteration == args.num_iterations):
            greedy_sr = greedy_eval(agent, eval_pool, get_obs_rms_state(envs), device,
                                    args.eval_episodes, args.max_steps)
            writer.add_scalar("charts/greedy_success_rate", greedy_sr, global_step)
            console.log(f"iter {iteration}: greedy eval success "
                        f"{greedy_sr:.2f} ({args.eval_episodes} episodes)")

        progress.update(
            task_id, completed=iteration,
            ret=(float(np.mean(ep_returns)) if ep_returns else float("nan")),
            len=(float(np.mean(ep_lengths)) if ep_lengths else float("nan")),
            term=(100.0 * float(np.mean(ep_terminated)) if ep_terminated else 0.0),
            eps=len(ep_returns), sps=sps,
        )

        if args.save_model and (iteration % args.save_every_iterations == 0 or iteration == args.num_iterations):
            ckpt_dir = Path(args.checkpoint_dir) / run_name / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"iter_{iteration:04d}.pt"
            torch.save({
                "iteration": iteration,
                "global_step": global_step,
                "model_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "np_rng": np.random.get_state(),
                "py_rng": random.getstate(),
                "obs_rms": get_obs_rms_state(envs),
                "return_rms": get_return_rms_state(envs),
                "args": vars(args),
            }, ckpt_path)
            latest = ckpt_dir / "latest.pt"
            if latest.exists() or latest.is_symlink():
                latest.unlink()
            latest.symlink_to(ckpt_path.name)
            console.log(f"Saved checkpoint: {ckpt_path}")

    progress.stop()
    envs.close()
    if eval_pool is not None:
        eval_pool.close()
    writer.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)

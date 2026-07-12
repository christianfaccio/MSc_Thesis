'''
IPPO (Independent PPO, de Witt et al. 2020) with parameter sharing on the
Oceananigans NetCDF env (src/envs/oceananigans.py: OceananigansEnv, n_agents=2).

This is to src/multi_agent/ippo_base.py what ppo_oceananigans.py is to
ppo_base.py: the SAME IPPO machinery (one actor-critic shared by every agent, a
fully DECENTRALIZED critic on the LOCAL obs, per-(env, agent, step) samples,
frozen-agent masking, manual RunningMeanStd normalization), pointed at the
turbulent LES salinity field instead of the synthetic Gaussian baseline. The
env's info["global_state"] is deliberately IGNORED here — only MAPPO uses it.

Env configuration and the oceananigans-specific learnings follow
ppo_oceananigans.py (2026-07-09 diagnosis): max_steps=1440 (diffusion can't
masquerade as navigation), success_steps_required=3 (arrive AND hold),
wall_penalty, entropy annealed 0.01 -> 0 over the first half, and a periodic
GREEDY (argmax) evaluation as the honest metric.

Multi-agent semantics: ONE shared (S*, tau*) target per episode;
end_on_any_success=True (2026-06-29 meeting) ends the episode as soon as any
agent holds the zone, so success_rate == success_any.

Usage (from root):
    - train       -> `python -m src.multi_agent.ippo_oceananigans`
    - tensorboard -> `tensorboard --logdir runs --port 6006`
'''
import random
import time
from collections import deque
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.utils.tensorboard import SummaryWriter

from src.multi_agent.policy import IppoPolicy
from src.envs.env_pool import AsyncEnvPool, SyncEnvPool
from src.envs.oceananigans_factory import make_raw_env_from_cfg

DEBUG = True
console = Console()
STATS_WINDOW = 100


class RunningMeanStd:
    '''Welford running mean/variance, batched (Parallel algorithm). Used for
    observation and reward normalization in place of the gym wrappers, which
    cannot handle the per-agent axis. (Inlined rather than imported from
    src.multi_agent.ippo, whose src.envs.multi_agent import no longer exists.)'''
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.mean = new_mean
        self.count = tot_count

    def state_dict(self):
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, state):
        self.mean = np.array(state["mean"], dtype=np.float64)
        self.var = np.array(state["var"], dtype=np.float64)
        self.count = float(state["count"])


@dataclass
class Args:
    exp_name: str = "ippo_oceananigans"
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

    # Environment arguments (mirror ppo_oceananigans.py, plus the agent axis)
    env_id: str = "OceananigansMultiAgent-ippo"
    """the id of the environment"""
    xml_file: str = "config/simulation.xml"
    """SwarmSwIM simulation XML (environment physics only; agents are created
    programmatically, any <agents> block in the XML is ignored)"""
    netcdf_file: str = "data/oceananigans/buoyancy_active"
    """Oceananigans NetCDF source: a directory (all *.nc), a glob, or a single file.
    Each reset draws a random file and (static mode) a random snapshot within it, so
    every episode sees a different frozen field sampled from the whole set. Default is
    the buoyancy-active dataset, matching ppo_oceananigans so single- vs multi-agent
    runs are directly comparable (run 1783718789 accidentally trained buoyancy_active
    with the no_buoyancy ε/σ sizing — double-relative-width success zone — making its
    success rate incomparable to PPO's). NOTE: epsilon_salinity / sigma_s below are
    sized to the field's per-snapshot span — switch them together with the dataset
    (buoyancy_active ~5 PSU: 0.15 / 1.5; no_buoyancy ~10 PSU: 0.3 / 3.0)."""
    epsilon_salinity: float = 0.15
    """success tolerance on |S - S*| (PSU), ~3% of the per-snapshot field span
    (buoyancy_active median span ~4.9 PSU, measured 2026-07-10)"""
    epsilon_turbidity: float = 0.05
    """success tolerance on |τ - τ*| (τ depends only on depth)"""
    sigma_s: float = 1.5
    """wide shaping-kernel width in S (PSU), ~0.3x the field span"""
    sigma_tau: float = 0.3
    """wide shaping-kernel width in τ (depth parity, see reward_func)"""
    max_cached_loaders: int = 8
    """per-env LRU cap on cached FieldLoaders (~90 MB each). Memory ≈
    (num_envs + 1 eval env) · max_cached_loaders · 90 MB (7·8 ≈ 5 GB)."""
    n_agents: int = 2
    """number of agents in the swarm (parameter-shared policy, one shared target)"""
    k: int = 12
    """observation history depth: last k (action direction, ΔS, Δτ) tuples appended to
    the 9-dim sensor frame (obs = 9 + 5k values per agent). On the buoyancy-active
    filament fields the gradient's basin of attraction covers only ~40% of the plane
    (2026-07-11 analysis), so a memoryless gradient-follower is insufficient — the
    history is what enables dead-reckoned escape from filament local optima."""
    target_mode: str = "random"
    """'random' = target (S*, τ*) read at a uniform random field point (typical S* ->
    the |ΔS|<ε zone covers ~10-20% of the plane at z*, large luck floor: a depth-only
    drifting baseline scores ~0.5 per agent); 'tail' = S* from a rare tail of the salinity
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
    """maximum env steps per episode before truncation. One env step ≈ 1 m of travel,
    the domain is 1 km and targets spawn up to ~1 diagonal away, so 1440 steps is
    generous slack for a navigator while keeping the STOCHASTIC policy from racking
    up 'success' by pure diffusion (the run-1783528628 failure mode at 7200 steps).
    Matches the γ=0.999 effective horizon (~1000 steps) and ppo_oceananigans."""
    dt: float = 0.1
    """simulator timestep (s) per sim sub-step"""
    frame_skip: int = 10
    """sim sub-steps per env step; one env step = dt·frame_skip = 1 s of sim time,
    so distance per step ≈ v_agent·dt·frame_skip = 1 m"""
    domain: tuple[float, float, float] = (1000.0, 1000.0, 100.0)
    """domain extent in (x, y, z) meters"""
    success_bonus: float = 10.0
    """reward bonus on reaching the target zone (shaped potential otherwise)"""
    static_frame: bool = True
    """NetCDF time handling: static (freeze one random snapshot per episode) first,
    dynamic later"""
    min_band_grad: float = 0.004
    """reject targets whose success band is ~flat (median |grad_xy S| < this, PSU/m) at
    reset, so every episode has a local gradient to home on; <=0 disables the guard"""
    target_min_dist_frac: float = 0.0
    """minimum spawn→target distance as a fraction of the domain diagonal; 0 = no
    check (targets may land near the spawn -> varied episode difficulty)"""
    wall_penalty: float = 0.05
    """per-step reward penalty for pinning against a domain wall, scaled by the
    fraction of the step's frame_skip ticks that were clamped. 0 disables."""
    success_steps_required: int = 3
    """consecutive in-zone steps required to count as success (arrive AND hold —
    kills single-step luck crossings on the turbulent field; see ppo_oceananigans)"""
    end_on_any_success: bool = True
    """end the episode as soon as ANY agent holds the target zone (the
    first-agent-to-find-it / success_any criterion, 2026-06-29 meeting)"""

    # Algorithm specific arguments (ppo_oceananigans values; batch adds the agent axis)
    total_timesteps: int = 10000000
    """total timesteps of the experiment (counts agent-env steps)"""
    learning_rate: float = 3.0e-4
    """the learning rate of the optimizer"""
    num_envs: int = 6
    """the number of parallel environments (6 envs · 2 agents = 12 agent-streams,
    the same rollout width as ppo_oceananigans's 12 envs)"""
    async_envs: bool = True
    """step the parallel envs in worker processes (src/envs/env_pool.py, spawn
    context) instead of a single-core loop. The env step is the wall-clock
    bottleneck (~n_agents·3 ms of SwarmSwIM+scipy per env-step vs a negligible
    policy forward), so this is a ~num_envs× rollout speedup up to the core count,
    with IDENTICAL training semantics (same batch, same manual reset points,
    per-env RNG unchanged). Workers build envs from the torch-free factory
    (src/envs/oceananigans_factory.py), so each costs ~an env's memory, not a
    torch import. --no-async-envs restores the in-process loop for debugging."""
    num_steps: int = 512
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """OFF for this field (see ppo_oceananigans: lr→0 froze the policy while it was
    still improving). Flat lr; the entropy anneal does the late-stage sharpening."""
    gamma: float = 0.999
    """discount factor; effective horizon 1/(1-γ) = 1000 steps ≈ 1000 m. MUST equal
    the env's γ for the potential-based shaping to stay policy-invariant."""
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
    """starting entropy coefficient — high exploration floor early on the deceptive
    turbulent field (see ppo_oceananigans for the full rationale)"""
    anneal_ent: bool = True
    """anneal ent_coef → ent_coef_final over the FIRST `ent_anneal_frac` of training,
    then HOLD the floor (explore early, commit late)"""
    ent_anneal_frac: float = 0.5
    """fraction of training over which ent_coef anneals; after that it HOLDS"""
    ent_coef_final: float = 0.0
    """entropy coefficient floor. 0 so the back half of training must commit; watch
    charts/greedy_success_rate for lock-in (the honest metric)"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.02
    """the target KL divergence threshold"""

    # Greedy evaluation (honest metric: the training success_rate measures the
    # STOCHASTIC policy; deployment/plot_trajectories.py uses greedy argmax)
    eval_every_iterations: int = 50
    """run a deterministic (argmax) evaluation every N iterations; 0 disables"""
    eval_episodes: int = 20
    """greedy episodes per evaluation. Fixed seeds (reused every eval) so the
    logged charts/greedy_success_rate is comparable across the run. Success =
    ANY agent terminated (same success_any bar as training). 20 keeps the
    binomial noise at ~±0.11 (4 episodes gave ±0.25 — unreadable trends)."""
    eval_workers: int = 4
    """parallel worker envs for the greedy evaluation (episodes fanned out in
    waves; greedy + fixed seeds, so the metric is identical to a sequential eval).
    Otherwise 20 sequential episodes × up to 1440 steps serialize a large slice of
    the wall-clock once rollouts are parallel. Each worker holds its own
    FieldLoader cache (~90 MB per cached file). 1 = sequential (previous
    behavior)."""

    # Checkpointing
    save_model: bool = True
    """if toggled, periodically save model + optimizer + RNG + normalization state"""
    save_every_iterations: int = 20
    """save a checkpoint every N PPO iterations (and always on the final iteration)"""
    checkpoint_dir: str = "runs"
    """parent directory for checkpoints; full path is <checkpoint_dir>/<run_name>/checkpoints/"""
    resume: str = None
    """path to a checkpoint .pt file to resume training from"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime): num_envs · num_steps · n_agents"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


# Args fields forwarded to the OceananigansEnv constructor, extracted to a plain
# dict so worker processes can unpickle the env factory without importing this
# (torch-heavy) module — see src/envs/oceananigans_factory.py.
ENV_CFG_KEYS = (
    "xml_file", "netcdf_file", "k", "n_agents", "v_agent", "max_steps", "dt",
    "domain", "frame_skip", "gamma", "success_bonus", "static_frame",
    "min_band_grad", "target_min_dist_frac", "wall_penalty",
    "success_steps_required", "max_cached_loaders", "end_on_any_success",
    "epsilon_salinity", "epsilon_turbidity", "sigma_s", "sigma_tau",
    "target_mode", "target_percentile", "reward_potential",
)


def env_cfg(args) -> dict:
    '''Plain-dict env configuration (picklable without this module).'''
    return {key: getattr(args, key) for key in ENV_CFG_KEYS}


def make_raw_env(args):
    '''One bare OceananigansEnv (n_agents=N) with the training configuration.'''
    return make_raw_env_from_cfg(env_cfg(args))


def make_env_pool(args):
    '''Pool of `num_envs` raw multi-agent envs — worker processes when
    async_envs (the env step is the wall-clock bottleneck; the gym vector
    wrappers can't carry the per-agent axis, hence the custom pool), else the
    previous in-process loop behind the same API.'''
    fns = [partial(make_raw_env_from_cfg, env_cfg(args)) for _ in range(args.num_envs)]
    return (AsyncEnvPool if args.async_envs else SyncEnvPool)(fns)


def greedy_eval(agent, eval_pool, obs_rms, device, n_episodes, max_steps,
                base_seed=1_000_000):
    '''Deterministic (argmax) rollouts on raw envs with the current training
    obs normalization applied — the same conditions as plot_trajectories.py.
    Episodes are fanned out in waves across the eval pool's workers; greedy
    actions + fixed per-episode seeds make the result identical to a sequential
    eval, only parallel. Success = ANY agent terminated under the TRAINING bar
    (env's success_steps_required).'''
    mean = obs_rms.mean
    std = np.sqrt(obs_rms.var + 1e-8)
    successes = 0
    was_training = agent.training
    agent.eval()
    with torch.no_grad():
        for wave_start in range(0, n_episodes, eval_pool.num_envs):
            episodes = range(wave_start, min(wave_start + eval_pool.num_envs, n_episodes))
            workers = list(range(len(episodes)))
            results = eval_pool.reset_where(
                workers, seeds=[base_seed + ep for ep in episodes])
            obs = {w: res[0] for w, res in zip(workers, results)}  # each (N, local_dim)
            active = workers
            for _ in range(max_steps):
                if not active:
                    break
                stack = np.concatenate([np.asarray(obs[w]) for w in active])  # (n_active·N, D)
                norm = np.clip((stack - mean) / std, -10.0, 10.0).astype(np.float32)
                logits = agent.actor(torch.tensor(norm, device=device))
                acts = logits.argmax(dim=-1).cpu().numpy().astype(np.int64)
                acts = acts.reshape(len(active), -1)  # (n_active, N)
                results = eval_pool.step_where(active, list(acts))
                still_active = []
                for w, (o, _, term, trunc, _) in zip(active, results):
                    obs[w] = o
                    if term.any():
                        successes += 1
                    elif not np.logical_or(term, trunc).all():
                        still_active.append(w)
                active = still_active
    if was_training:
        agent.train()
    return successes / n_episodes


def train(args):
    # batch_size collapses (num_steps, num_envs, n_agents): every agent-step is a sample.
    args.batch_size = int(args.num_envs * args.num_steps * args.n_agents)
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
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if DEBUG:
        print(f"Device: {device}")

    # env setup
    if DEBUG:
        print("--- Setting up the environments...")
    envs = make_env_pool(args)
    n_agents = args.n_agents
    local_dim = int(np.array(envs.attr(0, "local_observation_space").shape).prod())
    n_actions = envs.attr(0, "action_space").n

    # Dedicated raw-env pool for the periodic greedy evaluation; created once so
    # each worker's FieldLoader LRU cache persists across evals. The eval seeds
    # are fixed, so each worker only ever touches its own episodes' files — cap
    # its loader cache accordingly instead of args.max_cached_loaders.
    eval_pool = None
    if args.eval_every_iterations > 0:
        n_workers = max(1, min(args.eval_workers, args.eval_episodes))
        eval_cfg = env_cfg(args)
        episodes_per_worker = -(-args.eval_episodes // n_workers)  # ceil
        eval_cfg["max_cached_loaders"] = min(args.max_cached_loaders,
                                             max(2, episodes_per_worker))
        eval_fns = [partial(make_raw_env_from_cfg, eval_cfg) for _ in range(n_workers)]
        pool_cls = AsyncEnvPool if (args.async_envs and n_workers > 1) else SyncEnvPool
        eval_pool = pool_cls(eval_fns)

    # Parameter-shared actor-critic; critic uses the LOCAL obs (IPPO is decentralized).
    agent = IppoPolicy(local_dim, n_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Manual normalization (replaces gym wrappers, which cannot handle per-agent data).
    obs_rms = RunningMeanStd(shape=(local_dim,))
    return_rms = RunningMeanStd(shape=())
    obs_clip, rew_clip, var_eps = 10.0, 10.0, 1e-8

    def normalize_obs(raw, update=True):
        '''raw: (..., local_dim). Updates the running stats with the raw obs
        (when update) and returns the clipped, normalized obs.'''
        if update:
            obs_rms.update(raw.reshape(-1, local_dim))
        norm = (raw - obs_rms.mean) / np.sqrt(obs_rms.var + var_eps)
        return np.clip(norm, -obs_clip, obs_clip).astype(np.float32)

    # Discounted-return accumulator for reward normalization, per (env, agent).
    return_acc = np.zeros((args.num_envs, n_agents), dtype=np.float64)

    def normalize_reward(raw, done_after):
        '''raw / done_after: (num_envs, n_agents). Mirrors gym NormalizeReward:
        track a running discounted return, scale reward by its std, reset the
        accumulator where the (env, agent) episode just ended.'''
        return_acc[:] = return_acc * args.gamma + raw
        return_rms.update(return_acc.reshape(-1))
        norm = raw / np.sqrt(return_rms.var + var_eps)
        return_acc[:] = return_acc * (1.0 - done_after)  # reset finished trajectories
        return np.clip(norm, -rew_clip, rew_clip).astype(np.float32)

    # Resume from checkpoint if requested
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
            obs_rms.load_state_dict(ckpt["obs_rms"])
        if "return_rms" in ckpt:
            return_rms.load_state_dict(ckpt["return_rms"])
        if DEBUG:
            print(f"Resumed from {args.resume}: iteration={start_iteration}, global_step={global_step}")

    # ALGO Logic: Storage setup (explicit agent axis).
    obs = torch.zeros((args.num_steps, args.num_envs, n_agents, local_dim)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs, n_agents)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs, n_agents)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs, n_agents)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs, n_agents)).to(device)

    if DEBUG:
        print("--- GAME START ---")
    start_time = time.time()

    # Reset every env (in parallel); stack to (num_envs, n_agents, local_dim).
    raw_obs = np.zeros((args.num_envs, n_agents, local_dim), dtype=np.float32)
    for e, (o, _) in enumerate(envs.reset(seeds=[args.seed + e for e in range(args.num_envs)])):
        raw_obs[e] = o
    next_obs = torch.tensor(normalize_obs(raw_obs)).to(device)
    next_done = torch.zeros((args.num_envs, n_agents)).to(device)

    # Per-env episode accumulators (raw rewards) for logging.
    env_ep_return = np.zeros(args.num_envs, dtype=np.float64)
    env_ep_len = np.zeros(args.num_envs, dtype=np.int64)

    # Rolling stats over the last STATS_WINDOW finished episodes.
    ep_returns = deque(maxlen=STATS_WINDOW)   # per-agent mean return
    ep_lengths = deque(maxlen=STATS_WINDOW)
    # With end_on_any_success the episode is a SUCCESS as soon as ANY agent holds
    # the zone, so success_rate == success_any (still the STOCHASTIC policy —
    # charts/greedy_success_rate is the honest deployment metric).
    ep_success = deque(maxlen=STATS_WINDOW)

    progress = Progress(
        TextColumn("[bold blue]iter"),
        MofNCompleteColumn(),
        BarColumn(),
        TextColumn(
            "ret={task.fields[ret]:>6.2f}  len={task.fields[len]:>5.1f}  "
            "succ={task.fields[succ]:>3.0f}%  eps={task.fields[eps]:>4d}  "
            "SPS={task.fields[sps]:>5d}"
        ),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("<"),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=4,
    )
    task_id = progress.add_task(
        "train",
        total=args.num_iterations,
        completed=start_iteration - 1,
        ret=float("nan"),
        len=float("nan"),
        succ=0.0,
        eps=0,
        sps=0,
    )

    progress.start()
    for iteration in range(start_iteration, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        if args.anneal_ent:
            # anneal ent_coef -> ent_coef_final over the first ent_anneal_frac of
            # training, then HOLD the floor (explore early, commit late).
            train_frac = (iteration - 1.0) / args.num_iterations
            a = min(train_frac / max(args.ent_anneal_frac, 1e-8), 1.0)
            ent_coef_now = args.ent_coef + a * (args.ent_coef_final - args.ent_coef)
        else:
            ent_coef_now = args.ent_coef

        for step in range(0, args.num_steps):
            global_step += args.num_envs * n_agents
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: sample actions for the whole swarm in one batched call.
            with torch.no_grad():
                flat_obs = next_obs.reshape(args.num_envs * n_agents, local_dim)
                action, logprob, _, _ = agent.get_action_and_value(flat_obs)
            action = action.reshape(args.num_envs, n_agents)
            logprob = logprob.reshape(args.num_envs, n_agents)
            actions[step] = action
            logprobs[step] = logprob

            # Step every env, collect per-agent transitions.
            act_np = action.cpu().numpy().astype(np.int64)
            raw_next_obs = np.zeros((args.num_envs, n_agents, local_dim), dtype=np.float32)
            raw_reward = np.zeros((args.num_envs, n_agents), dtype=np.float32)
            done_after = np.zeros((args.num_envs, n_agents), dtype=np.float32)
            # Truncation is not termination: the potential-based shaping relies on
            # bootstrapping from the truncated state's value (the env keeps the real
            # Φ(s') there). The envs don't auto-reset, so the obs returned on the
            # ending step IS the true final obs — record it (and which agents were
            # truncated) and fold γ·V(final_obs) into their normalized reward below.
            trunc_flags = np.zeros((args.num_envs, n_agents), dtype=bool)
            final_obs = np.zeros((args.num_envs, n_agents, local_dim), dtype=np.float32)
            # All envs step concurrently in their workers; the loop below only
            # unpacks results (and issues the occasional per-env reset).
            step_results = envs.step(list(act_np))
            for e, (o, r, term, trunc, _) in enumerate(step_results):
                d = np.logical_or(term, trunc)
                raw_next_obs[e] = o
                raw_reward[e] = r
                done_after[e] = d
                env_ep_return[e] += float(r.sum())
                env_ep_len[e] += 1

                # The env does not auto-reset: when all its agents are done, log
                # the episode and reset it. done_after stays 1 so GAE stops at the
                # boundary; next_obs becomes the NEW episode's reset obs.
                if d.all():
                    trunc_flags[e] = np.logical_and(trunc, np.logical_not(term))
                    final_obs[e] = o
                    succeeded = float(term.any())   # success = ANY agent reached the target
                    ep_returns.append(env_ep_return[e] / n_agents)
                    ep_lengths.append(float(env_ep_len[e]))
                    ep_success.append(succeeded)
                    writer.add_scalar("charts/episodic_return", env_ep_return[e] / n_agents, global_step)
                    writer.add_scalar("charts/episodic_length", float(env_ep_len[e]), global_step)
                    writer.add_scalar("charts/episode_success", succeeded, global_step)
                    env_ep_return[e] = 0.0
                    env_ep_len[e] = 0
                    o, _ = envs.reset_at(e)
                    raw_next_obs[e] = o

            norm_reward = normalize_reward(raw_reward, done_after)
            if trunc_flags.any():
                fin_norm = normalize_obs(final_obs[trunc_flags], update=False)
                with torch.no_grad():
                    final_v = agent.get_value(
                        torch.tensor(fin_norm, device=device)).view(-1).cpu().numpy()
                norm_reward[trunc_flags] += args.gamma * final_v.astype(np.float32)

            rewards[step] = torch.tensor(norm_reward).to(device)
            next_obs = torch.tensor(normalize_obs(raw_next_obs)).to(device)
            next_done = torch.tensor(done_after).to(device)

        # ---- update -------------------------------------------------------
        # A frozen (already-succeeded) agent keeps emitting zero-reward steps
        # until its env resets. dones[step] is the done state recorded BEFORE the
        # action, so (1 - dones) masks out those frozen steps from the loss while
        # keeping each agent's true terminal step (where dones[step]==0).
        masks = 1.0 - dones

        # flatten the batch over (num_steps, num_envs, n_agents)
        b_obs = obs.reshape(-1, local_dim)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_masks = masks.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        # Recompute advantages with the current critic each epoch
        # (Andrychowicz et al. 2021, §3.5).
        for epoch in range(args.update_epochs):
            with torch.no_grad():
                new_values = agent.get_value(b_obs).reshape(args.num_steps, args.num_envs, n_agents)
                flat_next = next_obs.reshape(args.num_envs * n_agents, local_dim)
                next_value = agent.get_value(flat_next).reshape(args.num_envs, n_agents)
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
                mb_mask = b_masks[mb_inds]
                mask_sum = mb_mask.sum().clamp(min=1.0)

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio * mb_mask).sum() / mask_sum
                    approx_kl = (((ratio - 1) - logratio) * mb_mask).sum() / mask_sum
                    clipfracs += [
                        ((((ratio - 1.0).abs() > args.clip_coef).float() * mb_mask).sum()
                         / mask_sum).item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    # Normalize using only the active (unmasked) advantages.
                    active = mb_advantages[mb_mask.bool()]
                    if active.numel() > 1:
                        mb_advantages = (mb_advantages - active.mean()) / (active.std() + 1e-8)

                # Policy loss (mask out frozen agent-steps)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = (torch.max(pg_loss1, pg_loss2) * mb_mask).sum() / mask_sum

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * (v_loss_max * mb_mask).sum() / mask_sum
                else:
                    v_loss = 0.5 * (((newvalue - b_returns[mb_inds]) ** 2) * mb_mask).sum() / mask_sum

                entropy_loss = (entropy * mb_mask).sum() / mask_sum
                loss = pg_loss - ent_coef_now * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Explained variance over active steps only.
        active_mask = b_masks.bool().cpu().numpy()
        y_pred = b_values.cpu().numpy()[active_mask]
        y_true = b_returns.cpu().numpy()[active_mask]
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
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
        if ep_success:
            # Rolling success_any over the last STATS_WINDOW episodes (matches the console bar).
            writer.add_scalar("charts/success_rate", float(np.mean(ep_success)), global_step)

        # Periodic deterministic evaluation — charts/success_rate above tracks the
        # STOCHASTIC policy; greedy argmax is what plot_trajectories.py and
        # deployment use (2026-07-09 diagnosis).
        if eval_pool is not None and (iteration % args.eval_every_iterations == 0
                                     or iteration == args.num_iterations):
            greedy_sr = greedy_eval(agent, eval_pool, obs_rms, device,
                                    args.eval_episodes, args.max_steps)
            writer.add_scalar("charts/greedy_success_rate", greedy_sr, global_step)
            console.log(f"iter {iteration}: greedy eval success "
                        f"{greedy_sr:.2f} ({args.eval_episodes} episodes)")

        # Live UI update
        progress.update(
            task_id,
            completed=iteration,
            ret=(float(np.mean(ep_returns)) if ep_returns else float("nan")),
            len=(float(np.mean(ep_lengths)) if ep_lengths else float("nan")),
            succ=(100.0 * float(np.mean(ep_success)) if ep_success else 0.0),
            eps=len(ep_returns),
            sps=sps,
        )

        # Checkpoint save
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
                "obs_rms": obs_rms.state_dict(),
                "return_rms": return_rms.state_dict(),
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

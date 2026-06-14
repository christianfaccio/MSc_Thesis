'''
IPPO (Independent PPO, de Witt et al. 2020) with parameter sharing.

This is single-agent PPO (CleanRL lineage) lifted to a homogeneous swarm:
    - ONE actor-critic network is shared by every agent (parameter sharing).
    - The critic is fully DECENTRALIZED: it takes the agent's LOCAL observation,
      exactly like the actor. There is NO centralized critic — that is MAPPO,
      which only differs by feeding the critic the global state (see
      src/multi_agent/policy.py: MappoPolicy and env info["global_state"]).
    - PPO's clipped objective is what keeps learning stable under the
      non-stationarity of other agents learning at the same time.

Every (env, agent, step) tuple is an independent training sample: the rollout
buffers carry an explicit agent axis (num_steps, num_envs, n_agents, ...) and
are flattened over all three for the update.

The multi-agent env (src/envs/multi_agent.py) follows the PettingZoo-parallel
convention flattened over agents and does NOT auto-reset, so this loop manages a
plain Python list of `num_envs` MultiAgentEnv instances, resets an env when all
its agents are done, and normalizes observations/rewards with its own
RunningMeanStd (the gym.vector + NormalizeObservation/Reward wrapper stack used
in the single-agent code cannot handle a per-agent obs/reward layout).

Usage (from root):
    - train       -> `python -m src.multi_agent.ippo`
    - tensorboard -> `tensorboard --logdir runs --port 6006`
'''
import random
import time
from collections import deque
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
from src.envs.multi_agent import MultiAgentEnv

from dataclasses import dataclass

DEBUG = True
console = Console()
STATS_WINDOW = 100


@dataclass
class Args:
    exp_name: str = "ippo"
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

    # Environment arguments
    env_id: str = "MultiAgent-v0"
    """the id of the environment"""
    xml_file: str = "config/simulation.xml"
    """SwarmSwIM simulation XML (environment physics only; agents are created
    programmatically, any <agents> block in the XML is ignored)"""
    netcdf_file: str = "data/oceananigans/"
    """optional Oceananigans NetCDF data: single file, glob pattern (quote it in the
    shell, e.g. --netcdf-file 'data/oceananigans/hydrostatic_winter_run*.nc'), or
    directory; a random file + snapshot is sampled each episode reset"""
    n_agents: int = 2
    """number of agents in the swarm (parameter-shared policy)"""
    n_sources: int = 4
    """number of pollution sources spawned each reset"""
    k: int = 12
    """history buffer length for (action, reward) pairs; 12 steps × 10 s = 120 s of context"""
    v_agent: float = 1.0
    """agent commanded speed (m/s)"""
    max_steps: int = 720
    """maximum env steps per episode before truncation"""
    dt: float = 1.0
    """simulator timestep (s) per env step"""
    frame_skip: int = 10
    """sim sub-steps per env step (action repeated); one env step is dt · frame_skip = 10 s"""
    domain: tuple[float, float, float] = (5000.0, 5000.0, 40.0)
    """domain extent in (x, y, z) meters"""
    sigma_h: float = 500.0
    """salinity plume horizontal std [m] — scale with the domain"""
    sigma_v: float = 12.0
    """salinity plume vertical std [m]"""
    eddy_length_scale: float = 1000.0
    """vortex eddy radius [m] — scale with the domain"""

    # Algorithm specific arguments
    total_timesteps: int = 2000000
    """total timesteps of the experiment (counts agent-env steps)"""
    learning_rate: float = 3.0e-4
    """the learning rate of the optimizer"""
    num_envs: int = 6
    """the number of parallel environments"""
    num_steps: int = 512
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.995
    """the discount factor gamma; effective horizon 1/(1-γ) = 200 steps ≈ 2000 s"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation (0.95 recommended for MARL)"""
    num_minibatches: int = 2
    """the number of mini-batches (IPPO/MAPPO papers recommend very few: 1-2)"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient (MARL papers recommend ≤ 0.2)"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function"""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

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


class RunningMeanStd:
    '''Welford running mean/variance, batched (Parallel algorithm). Used for
    observation and reward normalization in place of the gym wrappers.'''
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


def make_envs(args):
    '''Build a list of `num_envs` raw MultiAgentEnv instances (no gym wrappers).'''
    envs = []
    for _ in range(args.num_envs):
        envs.append(MultiAgentEnv(
            xml_file=args.xml_file,
            netcdf_file=args.netcdf_file,
            n_agents=args.n_agents,
            n_sources=args.n_sources,
            k=args.k,
            v_agent=args.v_agent,
            max_steps=args.max_steps,
            dt=args.dt,
            frame_skip=args.frame_skip,
            domain=args.domain,
            sigma_h=args.sigma_h,
            sigma_v=args.sigma_v,
            eddy_length_scale=args.eddy_length_scale,
        ))
    return envs


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
    envs = make_envs(args)
    n_agents = args.n_agents
    local_dim = int(np.array(envs[0].local_observation_space.shape).prod())
    n_actions = envs[0].action_space.n

    # Parameter-shared actor-critic; critic uses the LOCAL obs (IPPO is decentralized).
    agent = IppoPolicy(local_dim, n_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Manual normalization (replaces gym wrappers, which cannot handle per-agent data).
    obs_rms = RunningMeanStd(shape=(local_dim,))
    return_rms = RunningMeanStd(shape=())
    obs_clip, rew_clip, var_eps = 10.0, 10.0, 1e-8

    def normalize_obs(raw, update=True):
        '''raw: (num_envs, n_agents, local_dim). Updates the running stats with
        the raw obs (when update) and returns the clipped, normalized obs.'''
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

    # Reset every env; stack to (num_envs, n_agents, local_dim).
    raw_obs = np.zeros((args.num_envs, n_agents, local_dim), dtype=np.float32)
    for e, env in enumerate(envs):
        o, _ = env.reset(seed=args.seed + e)
        raw_obs[e] = o
    next_obs = torch.tensor(normalize_obs(raw_obs)).to(device)
    next_done = torch.zeros((args.num_envs, n_agents)).to(device)

    # Per-env episode accumulators (raw rewards) for logging.
    env_ep_return = np.zeros(args.num_envs, dtype=np.float64)
    env_ep_len = np.zeros(args.num_envs, dtype=np.int64)

    # Rolling stats over the last STATS_WINDOW finished episodes
    ep_returns = deque(maxlen=STATS_WINDOW)   # per-agent mean return
    ep_lengths = deque(maxlen=STATS_WINDOW)
    ep_success = deque(maxlen=STATS_WINDOW)   # fraction of the swarm that reached the target
    ep_success_all = deque(maxlen=STATS_WINDOW)  # 1.0 if EVERY agent reached the target
    ep_success_any = deque(maxlen=STATS_WINDOW)  # 1.0 if AT LEAST ONE agent reached the target

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
            for e, env in enumerate(envs):
                o, r, term, trunc, _ = env.step(act_np[e])
                d = np.logical_or(term, trunc)
                raw_next_obs[e] = o
                raw_reward[e] = r
                done_after[e] = d
                env_ep_return[e] += float(r.sum())
                env_ep_len[e] += 1

                # The env does not auto-reset: when all its agents are done, log
                # the episode and reset it.
                if d.all():
                    frac = float(term.mean())   # fraction of the swarm that reached the target
                    ep_returns.append(env_ep_return[e] / n_agents)
                    ep_lengths.append(float(env_ep_len[e]))
                    ep_success.append(frac)
                    ep_success_all.append(float(term.all()))
                    ep_success_any.append(float(term.any()))
                    writer.add_scalar("charts/episodic_return", env_ep_return[e] / n_agents, global_step)
                    writer.add_scalar("charts/episodic_length", float(env_ep_len[e]), global_step)
                    writer.add_scalar("charts/episode_success", frac, global_step)
                    env_ep_return[e] = 0.0
                    env_ep_len[e] = 0
                    o, _ = env.reset()
                    raw_next_obs[e] = o
                    done_after[e] = 0.0  # fresh episode: next state is not terminal

            rewards[step] = torch.tensor(normalize_reward(raw_reward, done_after)).to(device)
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
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

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
            # Rolling means over the last STATS_WINDOW episodes (smooth, matches the console bar).
            writer.add_scalar("charts/success_rate", float(np.mean(ep_success)), global_step)        # per-agent mean
            writer.add_scalar("charts/success_rate_all", float(np.mean(ep_success_all)), global_step)  # whole swarm
            writer.add_scalar("charts/success_rate_any", float(np.mean(ep_success_any)), global_step)  # ≥1 agent

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
    for env in envs:
        env.close()
    writer.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)

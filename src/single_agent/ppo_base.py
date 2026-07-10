'''
PPO on the synthetic Gaussian-field baseline env (src/envs/base.py via
BaseSingleAgentEnv). Separate from src/single_agent/ppo.py (which trains on the
Oceananigans/analytical SingleAgentEnv) so the two don't overlap.

The env keeps the SwarmSwIM currents (+ Ekman) and Beer-Lambert turbidity, but the
salinity is a randomized sum of 3D Gaussians spanning ~10 PSU — a strong, navigable
gradient — to test whether the agent learns when the field gradient is NOT ~0.

Usage (from root):
    python -m src.single_agent.ppo_base
    tensorboard --logdir runs --port 6006
'''
import random
import time
from collections import deque
from dataclasses import dataclass
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
from src.envs.base import BaseEnv
# Reuse the env-agnostic normalization-state helpers from the main PPO trainer.
from src.single_agent.ppo import (
    get_obs_rms_state, set_obs_rms_state,
    get_return_rms_state, set_return_rms_state,
)

DEBUG = True
console = Console()
STATS_WINDOW = 100


@dataclass
class Args:
    exp_name: str = "ppo_base"
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

    # Environment arguments (synthetic Gaussian-field baseline)
    env_id: str = "BaseSingleAgent-ppo"
    """the id of the environment"""
    xml_file: str = "config/simulation.xml"
    """SwarmSwIM simulation XML"""
    k: int = 12
    """history buffer length for (action, potential) pairs"""
    v_agent: float = 1.0
    """agent commanded speed (m/s)"""
    max_steps: int = 5120
    """maximum env steps per episode before truncation"""
    dt: float = 0.1
    """simulator timestep (s) per sim sub-step"""
    frame_skip: int = 10
    """sim sub-steps per env step; one env step = dt·frame_skip = 1 s of sim time,
    so distance per step ≈ v_agent·dt·frame_skip = 1 m (1000 m domain -> ~1000 steps
    to cross; targets spawn ≥30% of the diagonal away)"""
    domain: tuple[float, float, float] = (1000.0, 1000.0, 100.0)
    """domain extent in (x, y, z) meters"""
    eddy_length_scale: float = 300.0
    """vortex eddy radius [m]"""
    salinity_sigma_h: float = 300.0
    """salinity Gaussian horizontal std [m] (domain-scale -> navigable gradient)"""
    salinity_sigma_v: float = 40.0
    """salinity Gaussian vertical std [m]"""
    salinity_span: float = 10.0
    """salinity field span [PSU] across the domain"""
    n_blobs: int = 3
    """per episode a random 2..n_blobs Gaussian blobs"""
    field_grid_n: int = 32
    """grid resolution used to normalize the field to span"""
    success_bonus: float = 10.0
    """reward bonus on reaching the target zone (shaped potential otherwise)"""

    # Algorithm specific arguments
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 3.0e-4
    """the learning rate of the optimizer"""
    num_envs: int = 6
    """the number of parallel environments"""
    num_steps: int = 512
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
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
    ent_coef: float = 0.003
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

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


def make_env(args):
    def thunk():
        env = BaseEnv(
            xml_file=args.xml_file,
            k=args.k,
            v_agent=args.v_agent,
            max_steps=args.max_steps,
            dt=args.dt,
            frame_skip=args.frame_skip,
            domain=args.domain,
            gamma=args.gamma,  # MUST match the trainer's γ for shaping invariance
            success_bonus=args.success_bonus,
            eddy_length_scale=args.eddy_length_scale,
            salinity_sigma_h=args.salinity_sigma_h,
            salinity_sigma_v=args.salinity_sigma_v,
            salinity_span=args.salinity_span,
            n_blobs=args.n_blobs,
            field_grid_n=args.field_grid_n,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env, lambda obs: np.clip(obs, -10.0, 10.0), env.observation_space
        )
        env = gym.wrappers.NormalizeReward(env, gamma=args.gamma)
        env = gym.wrappers.TransformReward(env, lambda r: float(np.clip(r, -10.0, 10.0)))
        return env

    return thunk


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
    envs = gym.vector.SyncVectorEnv([make_env(args) for _ in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), \
        "only discrete action space is supported"

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
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

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
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(next_done).to(device)

            if "episode" in infos:
                ep_r = infos["episode"]["r"]
                ep_l = infos["episode"]["l"]
                mask = infos.get("_episode", [True] * len(ep_r))
                for i, finished in enumerate(mask):
                    if finished:
                        ep_returns.append(float(ep_r[i]))
                        ep_lengths.append(float(ep_l[i]))
                        succ = 1.0 if bool(terminations[i]) else 0.0
                        ep_terminated.append(succ)
                        writer.add_scalar("charts/episodic_return", float(ep_r[i]), global_step)
                        writer.add_scalar("charts/episodic_length", float(ep_l[i]), global_step)
                        writer.add_scalar("charts/episode_success", succ, global_step)
            elif "final_info" in infos:
                for i, info in enumerate(infos["final_info"]):
                    if info and "episode" in info:
                        ep_returns.append(float(info["episode"]["r"]))
                        ep_lengths.append(float(info["episode"]["l"]))
                        succ = 1.0 if bool(terminations[i]) else 0.0
                        ep_terminated.append(succ)
                        writer.add_scalar("charts/episodic_return", float(info["episode"]["r"]), global_step)
                        writer.add_scalar("charts/episodic_length", float(info["episode"]["l"]), global_step)
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
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

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
    writer.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)
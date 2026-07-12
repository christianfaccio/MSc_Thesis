'''
Torch-free, picklable env factories for the Oceananigans trainers.

Worker processes (gymnasium AsyncVectorEnv, src/envs/env_pool.py) build their
envs by unpickling a zero-arg callable. If that callable closes over a trainer
module's Args dataclass, unpickling imports the trainer module — and torch —
in EVERY worker (hundreds of MB of RSS each). These factories instead take a
plain dict, so `functools.partial(make_raw_env_from_cfg, cfg)` pickles by
reference to THIS module and workers only import the env stack (gymnasium,
numpy, SwarmSwIM, scipy).

Trainers produce `cfg` with their `env_cfg(args)` helper (a plain-dict subset
of Args); every key here mirrors an OceananigansEnv constructor argument.
'''
import gymnasium as gym
import numpy as np

from src.envs.oceananigans import OceananigansEnv


def make_raw_env_from_cfg(cfg: dict) -> OceananigansEnv:
    '''Bare OceananigansEnv (single- or multi-agent via cfg["n_agents"]).'''
    return OceananigansEnv(
        xml_file=cfg["xml_file"],
        netcdf_file=cfg["netcdf_file"],
        k=cfg["k"],
        n_agents=cfg.get("n_agents", 1),
        v_agent=cfg["v_agent"],
        max_steps=cfg["max_steps"],
        dt=cfg["dt"],
        domain=tuple(cfg["domain"]),
        frame_skip=cfg["frame_skip"],
        gamma=cfg["gamma"],  # MUST match the trainer's γ for shaping invariance
        success_bonus=cfg["success_bonus"],
        static_frame=cfg["static_frame"],
        min_band_grad=cfg["min_band_grad"],
        target_min_dist_frac=cfg["target_min_dist_frac"],
        wall_penalty=cfg["wall_penalty"],
        success_steps_required=cfg["success_steps_required"],
        max_cached_loaders=cfg["max_cached_loaders"],
        end_on_any_success=cfg.get("end_on_any_success", True),
        epsilon_salinity=cfg["epsilon_salinity"],
        epsilon_turbidity=cfg["epsilon_turbidity"],
        sigma_s=cfg["sigma_s"],
        sigma_tau=cfg["sigma_tau"],
        target_mode=cfg["target_mode"],
        target_percentile=cfg["target_percentile"],
        reward_potential=cfg.get("reward_potential", "error"),
    )


def make_wrapped_single_env_from_cfg(cfg: dict) -> gym.Env:
    '''Single-agent env with the ppo_oceananigans training wrapper stack.'''
    env = make_raw_env_from_cfg(cfg)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(
        env, lambda obs: np.clip(obs, -10.0, 10.0), env.observation_space
    )
    env = gym.wrappers.NormalizeReward(env, gamma=cfg["gamma"])
    env = gym.wrappers.TransformReward(env, lambda r: float(np.clip(r, -10.0, 10.0)))
    return env

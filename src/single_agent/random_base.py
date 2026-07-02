'''
Random-policy baseline on the synthetic Gaussian-field env (src/envs/base.py).
Gives a success-rate floor to compare ppo_base.py against.

Usage (from root):
    python -m src.single_agent.random_base --iterations 100
'''
from dataclasses import dataclass
from typing import Optional

import numpy as np
import tyro

from src.envs.base import BaseEnv


@dataclass
class Args:
    iterations: int = 100
    """number of episodes to roll out"""
    seed: Optional[int] = None
    """episode reset seed (default: random)"""

    # Environment arguments (mirrors ppo_base.py's Args so results are comparable)
    xml_file: str = "config/simulation.xml"
    k: int = 12
    v_agent: float = 1.0
    max_steps: int = 5120
    dt: float = 0.1
    frame_skip: int = 10
    domain: tuple[float, float, float] = (1000.0, 1000.0, 100.0)
    gamma: float = 0.999
    success_bonus: float = 10.0
    eddy_length_scale: float = 300.0
    salinity_sigma_h: float = 300.0
    salinity_sigma_v: float = 40.0
    salinity_span: float = 10.0
    n_blobs: int = 3
    field_grid_n: int = 32


def build_env(args: Args) -> BaseEnv:
    return BaseEnv(
        xml_file=args.xml_file,
        k=args.k,
        v_agent=args.v_agent,
        max_steps=args.max_steps,
        dt=args.dt,
        frame_skip=args.frame_skip,
        domain=args.domain,
        gamma=args.gamma,
        success_bonus=args.success_bonus,
        eddy_length_scale=args.eddy_length_scale,
        salinity_sigma_h=args.salinity_sigma_h,
        salinity_sigma_v=args.salinity_sigma_v,
        salinity_span=args.salinity_span,
        n_blobs=args.n_blobs,
        field_grid_n=args.field_grid_n,
    )


def main():
    args = tyro.cli(Args)

    env = build_env(args)
    rng = np.random.default_rng(args.seed)
    n_actions = env.action_space.n

    successes = 0

    for _ in range(args.iterations):
        env.reset(seed=args.seed)
        steps = 0
        while steps < args.max_steps:
            a = rng.integers(0, n_actions)
            _, _, term, trunc, _ = env.step(a)
            steps += 1
            if bool(term):
                successes += 1
                break
            if bool(trunc):
                break

    success_rate = successes / args.iterations
    print(f"Success rate: {success_rate}")


if __name__ == "__main__":
    main()

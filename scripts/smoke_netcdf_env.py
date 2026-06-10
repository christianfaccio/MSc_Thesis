"""Smoke test: SingleAgentEnv in NetCDF (Oceananigans) and analytical modes.

Runs a few episodes in each mode and checks observations, rewards, physics
consistency and reset randomization. Not part of the pytest suite (needs the
~1 GB hydrostatic NetCDF file); run manually:

    python scripts/smoke_netcdf_env.py
"""

import time
import numpy as np
from src.envs.single_agent import SingleAgentEnv

NC = "data/oceananigans/hydrostatic_winter.nc"
XML = "config/simulation.xml"


def run_mode(netcdf_file, episodes=3, steps=20):
    label = "netcdf" if netcdf_file else "analytical"
    env = SingleAgentEnv(xml_file=XML, netcdf_file=netcdf_file, max_steps=steps,
                         dt=0.5, frame_skip=10)
    targets, snapshots = [], []
    t0 = time.time()
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        assert obs.shape == env.observation_space.shape, obs.shape
        assert np.isfinite(obs).all(), f"non-finite obs at reset: {obs}"
        targets.append((env.target_salinity, env.target_turbidity))
        if netcdf_file:
            snapshots.append(env.sim.current_3d.time_index)
            # data mode must disable the synthetic surface stack
            assert not env.sim.environment["is_vortex_currents"]
            assert not env.sim.environment["is_uniform_current"]
            assert env.sim.environment["current_3d_model"] == "data"
            # salinity must be near the Gulf baseline, not 0
            assert 35.0 < env.current_salinity < 50.0, env.current_salinity
            assert 35.0 < env.target_salinity < 50.0, env.target_salinity
        ep_rew = 0.0
        for t in range(steps):
            action = env.action_space.sample()
            obs, r, term, trunc, _ = env.step(action)
            assert np.isfinite(obs).all(), f"non-finite obs at step {t}"
            assert np.isfinite(r), r
            ep_rew += r
            if term or trunc:
                break
        print(f"  [{label}] ep{ep}: target S*={env.target_salinity:.3f} "
              f"tau*={env.target_turbidity:.3f} sum_r={ep_rew:.3f} "
              f"pos={np.round(env.sim.agents[0].pos, 1)}")
    # episode randomization sanity
    assert len(set(targets)) > 1, "targets identical across episodes"
    if netcdf_file:
        print(f"  [{label}] snapshots used: {snapshots}")
    print(f"  [{label}] OK ({time.time() - t0:.1f}s)")
    env.close()


if __name__ == "__main__":
    print("Analytical mode:")
    run_mode(None)
    print("NetCDF mode:")
    run_mode(NC)
    print("All smoke checks passed.")

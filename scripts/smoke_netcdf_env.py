"""Smoke test: SingleAgentEnv in NetCDF (Oceananigans) and analytical modes.

Runs a few episodes in each mode and checks observations, rewards, physics
consistency, reset randomization and time interpolation. Not part of the
pytest suite (needs the ~1 GB hydrostatic NetCDF file); run manually:

    python scripts/smoke_netcdf_env.py
"""

import glob
import time
import numpy as np
from src.envs.single_agent import SingleAgentEnv

NC = "data/oceananigans/hydrostatic_winter.nc"
NC_GLOB = "data/oceananigans/hydrostatic_winter_run*.nc"
XML = "config/simulation.xml"

PROBE = (2500.0, 2500.0, 20.0)  # fixed mid-domain point, depth positive-down


def run_time_interp(nc):
    """FieldLoader-level checks of the two-snapshot time blend (no env)."""
    from SwarmSwIM.ocean_data import FieldLoader

    class _Probe:
        pos = np.array(PROBE)

    t0 = time.time()
    L = FieldLoader(nc)
    # Frozen mode reference values (indices >=1 dodge the irregular first interval)
    L.set_snapshot(1)
    s1 = L.salinity_at(*PROBE)
    L.set_snapshot(2)
    s2 = L.salinity_at(*PROBE)
    L.set_snapshot(3)
    s3 = L.salinity_at(*PROBE)
    assert s1 != s2, "snapshots 1 and 2 identical at probe point"

    # Window mode: linear blend within [1, 2]
    dt_snap = L.times[2] - L.times[1]
    L.set_window(1)
    L.calculate(_Probe, sim_time=0.0)
    assert abs(L.salinity_at(*PROBE) - s1) < 1e-9
    L.calculate(_Probe, sim_time=0.5 * dt_snap)
    assert abs(L.salinity_at(*PROBE) - 0.5 * (s1 + s2)) < 1e-9, "blend not linear"
    L.calculate(_Probe, sim_time=dt_snap)
    assert abs(L.salinity_at(*PROBE) - s2) < 1e-9

    # Boundary crossing into [2, 3]
    L.calculate(_Probe, sim_time=1.5 * dt_snap)
    assert (L._idx_lo, L._idx_hi) == (2, 3), (L._idx_lo, L._idx_hi)
    assert L.window_start == 1
    mid = L.salinity_at(*PROBE)
    assert min(s2, s3) <= mid <= max(s2, s3), "value outside bracketing snapshots"

    # Frozen mode still works after window mode (cache invalidation, alpha reset)
    L.set_snapshot(1)
    assert abs(L.salinity_at(*PROBE) - s1) < 1e-9, "set_snapshot after set_window broken"

    # End-of-record: time past the last snapshot clamps (fields freeze, no raise)
    L.set_window(L.n_times - 2)
    L.calculate(_Probe, sim_time=10 * dt_snap)
    s_end = L.salinity_at(*PROBE)
    L.set_snapshot(L.n_times - 1)
    assert abs(L.salinity_at(*PROBE) - s_end) < 1e-9, "end-of-record clamp broken"

    # max_window_start covers the requested duration; huge durations clamp to 0
    i = L.max_window_start(7200.0)
    assert L.times[i] + 7200.0 <= L.times[-1] + 1e-6
    assert L.max_window_start(1e12) == 0
    L.close()
    print(f"  [time-interp] OK ({time.time() - t0:.1f}s)")


def run_mode(netcdf_file, episodes=3, steps=20, label=None):
    label = label or ("netcdf" if netcdf_file else "analytical")
    env = SingleAgentEnv(xml_file=XML, netcdf_file=netcdf_file, max_steps=steps,
                         dt=1.0, frame_skip=10)
    targets, windows, paths = [], [], set()
    t0 = time.time()
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        assert obs.shape == env.observation_space.shape, obs.shape
        assert np.isfinite(obs).all(), f"non-finite obs at reset: {obs}"
        targets.append((env.target_salinity, env.target_turbidity))
        if netcdf_file:
            windows.append(env.sim.current_3d.window_start)
            paths.add(env.active_netcdf_path)
            # data mode must disable the synthetic surface stack
            assert not env.sim.environment["is_vortex_currents"]
            assert not env.sim.environment["is_uniform_current"]
            assert env.sim.environment["current_3d_model"] == "data"
            # salinity must be near the Gulf baseline, not 0
            assert 35.0 < env.current_salinity < 50.0, env.current_salinity
            assert 35.0 < env.target_salinity < 50.0, env.target_salinity
            s_probe0 = env.sim.current_3d.salinity_at(*PROBE)  # alpha = 0 here
        ep_rew = 0.0
        for t in range(steps):
            action = env.action_space.sample()
            obs, r, term, trunc, _ = env.step(action)
            assert np.isfinite(obs).all(), f"non-finite obs at step {t}"
            assert np.isfinite(r), r
            pos = env.sim.agents[0].pos
            for ax in range(3):
                assert 0.0 <= pos[ax] <= env.domain[ax], \
                    f"position out of bounds on axis {ax}: {pos}"
            ep_rew += r
            if term or trunc:
                break
        if netcdf_file:
            # The field at a fixed point must evolve within the episode, unless
            # the two bracketing snapshots happen to be identical there.
            L = env.sim.current_3d
            assert L._alpha > 0.0, "time blend never advanced during episode"
            zq, yq, xq = -PROBE[2], PROBE[1], PROBE[0]
            lo_v = float(L._fields_lo["S"]((zq, yq, xq)))
            hi_v = float(L._fields_hi["S"]((zq, yq, xq)))
            if abs(hi_v - lo_v) > 1e-9:
                assert L.salinity_at(*PROBE) != s_probe0, \
                    "salinity frozen despite time interpolation"
        print(f"  [{label}] ep{ep}: target S*={env.target_salinity:.3f} "
              f"tau*={env.target_turbidity:.3f} sum_r={ep_rew:.3f} "
              f"pos={np.round(env.sim.agents[0].pos, 1)}")
    # episode randomization sanity
    assert len(set(targets)) > 1, "targets identical across episodes"
    if netcdf_file:
        print(f"  [{label}] windows used: {windows}")
        print(f"  [{label}] files used: {sorted(paths)}")
        if len(env._nc_files) > 1 and episodes >= 4:
            assert len(paths) > 1, "multi-file mode never switched file"
            assert len(env._loaders) >= 2, "loader cache should hold >=2 entries"
    print(f"  [{label}] OK ({time.time() - t0:.1f}s)")
    env.close()


if __name__ == "__main__":
    print("FieldLoader time interpolation:")
    run_time_interp(NC)
    print("Analytical mode:")
    run_mode(None)
    print("NetCDF mode (single file):")
    run_mode(NC)
    if len(glob.glob(NC_GLOB)) >= 2:
        print("NetCDF mode (multi-file glob):")
        run_mode(NC_GLOB, episodes=6, label="glob")
    else:
        print(f"Skipping glob mode: <2 files match {NC_GLOB} "
              "(generate with oceananigans/hydrostatic.jl --n-runs 2)")
    print("All smoke checks passed.")

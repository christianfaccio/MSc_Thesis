'''Tests for the success_all evaluation mode of OceananigansEnv
(end_on_any_success=False): the episode runs past the first arrival until every
agent has arrived or time runs out, and per-agent arrival times are reported in
the terminal info dict. The default (end_on_any_success=True) must reproduce
the success_any training semantics byte-for-byte.

These run against a real NetCDF file, so they are skipped when the dataset is
not present (e.g. on a checkout without data/).
'''
import numpy as np
import pytest

from src.envs.oceananigans import OceananigansEnv

NC = "data/oceananigans/buoyancy_active/train/nonhydro_winter_run001.nc"
XML = "config/simulation.xml"

pytestmark = pytest.mark.skipif(
    not __import__("os").path.isfile(NC), reason="Oceananigans dataset not available")

NOOP = np.full(3, 13, dtype=np.int64)   # action 13 = (0,0,0)


def make_env(**kw):
    cfg = dict(xml_file=XML, netcdf_file=NC, k=0, n_agents=3, max_steps=12,
               dt=0.1, domain=(1000.0, 1000.0, 100.0), frame_skip=10,
               gamma=0.9997, success_bonus=20.0, epsilon_salinity=0.15,
               epsilon_turbidity=0.05, reward_potential="distance")
    cfg.update(kw)
    return OceananigansEnv(**cfg)


def force_first_success(env):
    '''Make agent 0 (and only agent 0) terminate on the NEXT step: the whole
    domain becomes the success zone, but agent 0 starts one hold-step ahead of
    the success_steps_required=2 requirement the env was built with.'''
    env.epsilon_salinity = 1e9
    env.epsilon_turbidity = 1e9
    env._in_zone_steps[0] = 1


def test_default_reproduces_success_any():
    '''Same seed, same actions: the default env and an explicit
    end_on_any_success=True env must be indistinguishable, and the episode must
    end on the first arrival with the stragglers truncated.'''
    for kw in ({}, dict(end_on_any_success=True)):
        env = make_env(success_steps_required=2, **kw)
        env.reset(seed=7)
        force_first_success(env)
        _, r, term, trunc, info = env.step(NOOP)
        assert list(term) == [True, False, False]
        assert list(trunc) == [False, True, True]      # team-terminal, not timeout
        assert info["timeout"] is False
        tts = np.asarray(info["time_to_success"], dtype=float)
        assert tts[0] == 1.0 and np.isnan(tts[1:]).all()
        assert info["n_succeeded"] == 1
        assert np.isnan(info["time_to_all_success"])


def test_success_all_runs_past_first_arrival():
    env = make_env(success_steps_required=2, end_on_any_success=False)
    env.reset(seed=7)
    force_first_success(env)

    # Step 1: agent 0 arrives; the episode must NOT end.
    _, r, term, trunc, info = env.step(NOOP)
    assert list(term) == [True, False, False]
    assert not trunc.any()
    assert "time_to_success" not in info               # episode still running

    # Step 2: the stragglers complete their 2-step hold; frozen agent 0 must
    # collect exactly zero reward while the episode runs on without it.
    _, r, term, trunc, info = env.step(NOOP)
    assert r[0] == 0.0
    assert term.all() and not trunc.any()
    tts = np.asarray(info["time_to_success"], dtype=float)
    np.testing.assert_allclose(tts, [1.0, 2.0, 2.0])
    assert info["n_succeeded"] == 3
    assert info["time_to_all_success"] == 2.0
    assert info["timeout"] is False                    # genuine team-terminal
    assert np.isfinite(info["time_to_first_success"])
    assert info["time_to_first_success"] == 1.0


def test_success_all_partial_then_timeout():
    '''One arrival, then nobody else can succeed: the episode must run to
    max_steps and end as a TIMEOUT (bootstrappable), with the straggler slots
    still NaN and success_all not granted.'''
    env = make_env(success_steps_required=2, end_on_any_success=False, max_steps=3)
    env.reset(seed=7)
    force_first_success(env)
    _, _, term, trunc, info = env.step(NOOP)           # agent 0 arrives at step 1
    assert list(term) == [True, False, False] and not trunc.any()
    # Slam the zone shut so agents 1 and 2 can never satisfy the in-zone test.
    env.epsilon_salinity = -1.0
    env.epsilon_turbidity = -1.0
    _, _, term, trunc, info = env.step(NOOP)           # step 2: nothing happens
    assert not trunc.any() and "time_to_success" not in info
    _, _, term, trunc, info = env.step(NOOP)           # step 3 = max_steps
    assert list(term) == [True, False, False]
    assert list(trunc) == [False, True, True]
    assert info["timeout"] is True                     # time limit, not terminal
    tts = np.asarray(info["time_to_success"], dtype=float)
    assert tts[0] == 1.0 and np.isnan(tts[1:]).all()
    assert info["n_succeeded"] == 1
    assert np.isnan(info["time_to_all_success"])


def test_success_step_state_resets_between_episodes():
    env = make_env(success_steps_required=2, end_on_any_success=False)
    env.reset(seed=7)
    force_first_success(env)
    env.step(NOOP)
    env.step(NOOP)                                     # everyone arrived
    env.reset(seed=8)
    assert np.isnan(env._success_step).all()
    assert not env._success.any()

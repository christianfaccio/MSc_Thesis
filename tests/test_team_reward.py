'''Tests for the team-reward block of OceananigansEnv (difference reward,
separation potential, shared success bonus) and the _zone_tree freshness fix.

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


def make_env(**kw):
    cfg = dict(xml_file=XML, netcdf_file=NC, k=0, n_agents=3, max_steps=12,
               dt=0.1, domain=(1000.0, 1000.0, 100.0), frame_skip=10,
               gamma=0.9997, success_bonus=20.0, epsilon_salinity=0.15,
               epsilon_turbidity=0.05, reward_potential="distance",
               end_on_any_success=False)
    cfg.update(kw)
    return OceananigansEnv(**cfg)


def rollout(env, seed, steps=12):
    '''Fixed action sequence so two envs are compared on identical trajectories.'''
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(0)
    rewards, infos = [], []
    for _ in range(steps):
        a = rng.integers(0, 27, size=env.n_agents)
        obs, r, term, trunc, info = env.step(a)
        rewards.append(np.asarray(r, dtype=float).copy())
        infos.append(info)
        if np.logical_or(term, trunc).all():
            break
    return np.array(rewards), infos


def test_defaults_reproduce_individual_baseline():
    '''beta=lambda=0 and per-agent bonus must leave the reward untouched.'''
    r_default, _ = rollout(make_env(), seed=7)
    r_explicit, _ = rollout(make_env(alpha_individual=1.0, beta_difference=0.0,
                                     lambda_separation=0.0,
                                     shared_success_bonus=False), seed=7)
    np.testing.assert_allclose(r_default, r_explicit, rtol=0, atol=0)


def test_difference_reward_is_zero_except_for_the_leader():
    env = make_env(beta_difference=1.0)
    env.reset(seed=3)
    D, _ = env._team_potentials()
    assert np.count_nonzero(D) <= 1, "D must credit at most one agent (the leader)"
    d = env._zone_distances()
    lead = int(np.argmin(d))
    assert D[lead] > 0.0 and np.all(np.delete(D, lead) == 0.0)
    # D_lead == g(d_lead) - g(d_runner_up): the margin over the second best.
    runner = int(np.argsort(d)[1])
    assert D[lead] == pytest.approx(env._g(d[lead]) - env._g(d[runner]))


def test_difference_reward_ignores_frozen_agents():
    '''A succeeded agent sits at d~0; if it were counted it would be the
    permanent leader and zero out everybody else's D forever.'''
    env = make_env(beta_difference=1.0)
    env.reset(seed=3)
    d = env._zone_distances()
    lead = int(np.argmin(d))
    env._success[lead] = True
    D, _ = env._team_potentials()
    assert D[lead] == 0.0
    assert np.count_nonzero(D) == 1  # leadership passed to the best live agent
    # Only one live agent left -> no runner-up -> the whole term vanishes.
    env._success[:] = [True, True, False]
    D, _ = env._team_potentials()
    assert np.all(D == 0.0)


def test_separation_potential_saturates():
    env = make_env(lambda_separation=1.0, separation_scale=150.0)
    env.reset(seed=5)
    # Far apart in every pair -> everyone saturates at the 10.0 cap.
    for a, p in zip(env.sim.agents, [(0., 0., 10.), (900., 0., 10.), (0., 900., 10.)]):
        a.pos[:] = p
    _, sep = env._team_potentials()
    np.testing.assert_allclose(sep, 10.0)
    # Co-located -> zero separation reward.
    for a in env.sim.agents:
        a.pos[:] = (500., 500., 10.)
    _, sep = env._team_potentials()
    np.testing.assert_allclose(sep, 0.0)
    # Half a scale length apart -> exactly half the cap (linear below saturation).
    for a, p in zip(env.sim.agents, [(0., 0., 10.), (75., 0., 10.), (900., 900., 10.)]):
        a.pos[:] = p
    _, sep = env._team_potentials()
    assert sep[0] == pytest.approx(5.0) and sep[1] == pytest.approx(5.0)


def test_shared_success_bonus_pays_every_live_agent():
    '''Force a success by making the whole domain the success zone, then check
    who gets paid under each bonus mode.'''
    for shared, expect_all in ((False, False), (True, True)):
        env = make_env(shared_success_bonus=shared, success_bonus=20.0)
        env.reset(seed=11)
        # Everything is in-zone: agent 0 terminates on the next step.
        env.epsilon_salinity = 1e9
        env.epsilon_turbidity = 1e9
        _, r, term, _, _ = env.step(np.zeros(env.n_agents, dtype=np.int64))
        assert term.all()  # all three land in the (now infinite) zone together
        # Per-agent mode: each agent is paid for its OWN success only (1x).
        # Shared mode: each live agent is paid for every success in the step (3x).
        assert np.all(r > 50.0) if expect_all else np.all(r < 50.0)


def test_zone_tree_is_rebuilt_every_reset():
    '''Regression: _respawn_far_from_zone only built the tree when it was None,
    so with reward_potential="error" episode 2+ measured spawn distance against
    episode 1's zone.'''
    env = make_env(reward_potential="error", min_spawn_distance=100.0)
    env.reset(seed=1)
    first = env._zone_tree
    env.reset(seed=2)
    assert env._zone_tree is not first, "stale zone tree reused across episodes"


def test_episode_stats_reported_on_final_step():
    env = make_env(max_steps=4, beta_difference=1.0, lambda_separation=0.25)
    _, infos = rollout(env, seed=13, steps=4)
    last = infos[-1]
    assert "time_to_first_success" in last
    assert "coverage_redundancy" in last and 0.0 < last["coverage_redundancy"] <= 1.0
    assert "nn_distance" in last and last["nn_distance"] > 0.0
    # Intermediate steps must stay lean (no per-step set unions).
    assert "coverage_redundancy" not in infos[0]


def test_team_terms_rejected_for_single_agent():
    with pytest.raises(ValueError):
        make_env(n_agents=1, beta_difference=1.0)


def test_shaping_telescopes_to_a_constant():
    '''Ng invariance: with no success, the summed shaping over an episode must
    equal γ^T·Φ(s_T) − Φ(s_0) (up to the per-step γ factors), independent of the
    actions taken. Checked on the joint terms as well as the individual one.'''
    env = make_env(max_steps=6, beta_difference=1.0, lambda_separation=0.5,
                   epsilon_salinity=1e-9)  # unreachable zone -> no termination
    env.reset(seed=17)
    phi0 = env._prev_potential.copy()
    D0, sep0 = env._prev_difference.copy(), env._prev_separation.copy()
    rewards, _ = rollout_no_reset(env, steps=6)
    phiT = env._prev_potential.copy()
    DT, sepT = env._prev_difference.copy(), env._prev_separation.copy()
    g = env.gamma
    # Per-step terms are γΦ(s')−Φ(s); the undiscounted sum telescopes to
    # Φ(s_T)−Φ(s_0) plus the (γ−1)ΣΦ drift, which is what we reconstruct here.
    expected = (g * phiT - phi0) + 1.0 * (g * DT - D0) + 0.5 * (g * sepT - sep0)
    drift = rewards.sum(axis=0) - expected
    # Only the intermediate (γ−1)Φ terms remain; with γ≈1 they are tiny.
    assert np.all(np.abs(drift) < 0.05), drift


def rollout_no_reset(env, steps):
    rng = np.random.default_rng(0)
    rewards = []
    for _ in range(steps):
        _, r, term, trunc, _ = env.step(rng.integers(0, 27, size=env.n_agents))
        rewards.append(np.asarray(r, dtype=float).copy())
        if np.logical_or(term, trunc).all():
            break
    return np.array(rewards), None

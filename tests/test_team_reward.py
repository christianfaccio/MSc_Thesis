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


def test_comm_neighbor_blocks_are_sorted_nearest_first():
    '''The actor is parameter-shared, so neighbour slots must carry the same
    meaning for every agent. Index order does not: slot 0 is "agent 1" for agent
    0 but "agent 0" for agent 1. Distance order does.'''
    env = make_env(n_agents=4, communication=True, k=0)
    env.reset(seed=23)
    # Agent 0 at the origin; the others at known, deliberately out-of-index-order
    # ranges so index order and distance order disagree.
    for a, p in zip(env.sim.agents, [(0., 0., -10.), (300., 0., -10.),
                                     (100., 0., -10.), (200., 0., -10.)]):
        a.pos[:] = p
        a.psi = 0.0
    obs, _, _, _ = env._build_state(0, env.sim.agents[0], None)
    # frame = 9 scalars, then 3 blocks of 5. rel_x is offset 1 within each block.
    rel_x = [float(obs[9 + 5 * b + 1]) for b in range(3)]
    assert rel_x == pytest.approx([100.0, 200.0, 300.0]), \
        f"neighbour blocks not nearest-first: {rel_x}"

    # Every agent's slots must be in ascending range. The body-frame rotation is
    # orthogonal, so ‖(rel_x, rel_y, rel_z)‖ is the true distance regardless of
    # heading (rel_x alone is signed and would be negative for a neighbour astern).
    for i, agent in enumerate(env.sim.agents):
        o, _, _, _ = env._build_state(i, agent, None)
        slots = [float(np.linalg.norm(o[9 + 5 * b + 1: 9 + 5 * b + 4]))
                 for b in range(env.n_agents - 1)]
        truth = sorted(float(np.linalg.norm(other.pos - agent.pos))
                       for j, other in enumerate(env.sim.agents) if j != i)
        assert slots == pytest.approx(truth), f"agent {i} slots not nearest-first"


def test_sorting_is_a_noop_for_two_agents():
    '''With one neighbour there is nothing to sort, so the 2-agent runs recorded
    before this change stay exactly comparable.'''
    env = make_env(n_agents=2, communication=True, k=0)
    env.reset(seed=29)
    for i, agent in enumerate(env.sim.agents):
        obs, _, _, _ = env._build_state(i, agent, None)
        other = env.sim.agents[1 - i]
        rel = other.pos - agent.pos
        psi = np.deg2rad(agent.psi)
        c, s = np.cos(psi), np.sin(psi)
        assert obs[9] == pytest.approx(1.0)
        assert float(obs[9 + 1]) == pytest.approx(rel[0] * c + rel[1] * s)
        assert float(obs[9 + 2]) == pytest.approx(-rel[0] * s + rel[1] * c)
        assert float(obs[9 + 3]) == pytest.approx(rel[2])


def test_spawn_mode_origin_puts_every_agent_at_the_corner():
    env = make_env(n_agents=3, spawn_mode="origin")
    for seed in (1, 2):
        env.reset(seed=seed)
        for a in env.sim.agents:
            np.testing.assert_allclose(a.pos, (0.0, 0.0, 0.0), atol=1e-9)


def test_spawn_mode_max_dist_matches_the_specified_geometry():
    '''N=2 on the 1 km domain must be exactly (500,0,0) and (0,500,0) — the
    maximum-separation placement along the L-shaped coastline.'''
    env = make_env(n_agents=2, spawn_mode="max_dist")
    env.reset(seed=1)
    got = sorted(tuple(np.round(a.pos, 6)) for a in env.sim.agents)
    assert got == [(0.0, 500.0, 0.0), (500.0, 0.0, 0.0)], got

    # N=4: still on the walls, still at z=0, and evenly spaced along the path.
    env = make_env(n_agents=4, spawn_mode="max_dist")
    env.reset(seed=1)
    pos = np.array([a.pos for a in env.sim.agents], float)
    assert np.allclose(pos[:, 2], 0.0), "fixed spawns must be at the surface"
    on_wall = np.isclose(pos[:, 0], 0.0) | np.isclose(pos[:, 1], 0.0)
    assert on_wall.all(), f"agents off the land walls: {pos}"
    # Arclength along (Lx,0)->(0,0)->(0,Ly) must be the segment centres.
    Lx = env.domain[0]
    s = np.sort([Lx - p[0] if np.isclose(p[1], 0.0) else Lx + p[1] for p in pos])
    np.testing.assert_allclose(s, [250.0, 750.0, 1250.0, 1750.0])


def test_fixed_spawns_are_deterministic_but_heading_still_varies():
    '''The point of the fixed modes is a reproducible deployment geometry; the
    heading stays random so episodes are not fully degenerate.'''
    env = make_env(n_agents=2, spawn_mode="max_dist")
    env.reset(seed=5)
    p1 = np.array([a.pos.copy() for a in env.sim.agents])
    h1 = [a.psi for a in env.sim.agents]
    env.reset(seed=6)
    p2 = np.array([a.pos.copy() for a in env.sim.agents])
    h2 = [a.psi for a in env.sim.agents]
    np.testing.assert_allclose(p1, p2)
    assert h1 != h2, "heading should still be redrawn each episode"


def test_min_spawn_distance_rejected_with_a_fixed_spawn_mode():
    with pytest.raises(ValueError):
        make_env(spawn_mode="origin", min_spawn_distance=100.0)
    with pytest.raises(ValueError):
        make_env(spawn_mode="bogus")


def test_timeout_flag_distinguishes_team_terminal_from_time_limit():
    '''Regression: under end_on_any_success the non-succeeding agents are flagged
    `truncated`, but a teammate's success is TERMINAL for the team — bootstrapping
    gamma*V(final) there while also paying the shared bonus double-counts it and
    makes free-riding pay more than finding the target.'''
    # (a) episode ends on a teammate's success -> NOT a timeout
    env = make_env(max_steps=10_000, end_on_any_success=True,
                   shared_success_bonus=True)
    env.reset(seed=11)
    env.epsilon_salinity = 1e9      # everything is in-zone -> immediate success
    env.epsilon_turbidity = 1e9
    _, _, term, trunc, info = env.step(np.zeros(env.n_agents, dtype=np.int64))
    assert term.any(), "expected a success"
    assert info["timeout"] is False, "team success must not be reported as a timeout"

    # (b) episode ends at max_steps with nobody succeeding -> IS a timeout
    env = make_env(max_steps=3, epsilon_salinity=1e-9)  # unreachable zone
    env.reset(seed=11)
    for _ in range(3):
        _, _, term, trunc, info = env.step(np.zeros(env.n_agents, dtype=np.int64))
    assert not term.any() and trunc.all()
    assert info["timeout"] is True, "a genuine time limit must be bootstrappable"

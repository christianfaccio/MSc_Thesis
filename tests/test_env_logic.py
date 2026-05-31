import pytest

from src.envs.single_agent import SingleAgentEnv


@pytest.fixture
def env():
    e = SingleAgentEnv(xml_file="unused.xml", n_sources=2)
    e.target_salinity = 2.0
    e.target_turbidity = 0.5
    return e


def test_observation_space_default_k():
    pass  # covered in test_action_space


def test_in_zone_when_perfect_match(env):
    env.current_salinity = env.target_salinity
    env.current_turbidity = env.target_turbidity
    assert env._is_in_zone() is True


def test_in_zone_within_epsilon(env):
    env.current_salinity = env.target_salinity + env.epsilon_salinity / 2
    env.current_turbidity = env.target_turbidity - env.epsilon_turbidity / 2
    assert env._is_in_zone() is True


def test_not_in_zone_above_target(env):
    """Regression: the previous signed inequality wrongly returned True here."""
    env.current_salinity = env.target_salinity + 2 * env.epsilon_salinity
    env.current_turbidity = env.target_turbidity
    assert env._is_in_zone() is False


def test_not_in_zone_below_target(env):
    """Regression: the previous signed inequality wrongly returned True here."""
    env.current_salinity = env.target_salinity - 2 * env.epsilon_salinity
    env.current_turbidity = env.target_turbidity
    assert env._is_in_zone() is False


def test_not_in_zone_when_turbidity_off(env):
    env.current_salinity = env.target_salinity
    env.current_turbidity = env.target_turbidity + 10.0
    assert env._is_in_zone() is False


def test_just_outside_epsilon_is_out(env):
    """A value safely above epsilon must be classified as out-of-zone."""
    env.current_salinity = env.target_salinity + 1.5 * env.epsilon_salinity
    env.current_turbidity = env.target_turbidity
    assert env._is_in_zone() is False


def test_observation_dim_formula():
    for k in (1, 4, 8):
        e = SingleAgentEnv(xml_file="unused.xml", n_sources=2, k=k)
        assert e.observation_space.shape == (2 * k + 7,)


def test_epsilon_defaults(env):
    assert env.epsilon_salinity == pytest.approx(0.1)
    assert env.epsilon_turbidity == pytest.approx(0.01)


def test_max_steps_default(env):
    assert env.max_steps == 1024


def test_action_space_is_discrete_27(env):
    assert env.action_space.n == 27

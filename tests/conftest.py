import json

import pytest


@pytest.fixture
def single_source():
    return [
        {"name": "s0", "x": 50.0, "y": 50.0, "depth": 20.0, "Q": 1.0},
    ]


@pytest.fixture
def two_sources():
    return [
        {"name": "s0", "x": 50.0, "y": 50.0, "depth": 20.0, "Q": 1.0},
        {"name": "s1", "x": 80.0, "y": 80.0, "depth": 30.0, "Q": 0.5},
    ]


@pytest.fixture
def tmp_sources_json(tmp_path, two_sources):
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(two_sources))
    return str(path)

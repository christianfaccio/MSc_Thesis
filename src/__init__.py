import sys

# The ./SwarmSwIM/ submodule clone has no top-level __init__.py (the real
# package lives at ./SwarmSwIM/SwarmSwIM/), so whenever the repo root is on
# sys.path (cwd, pytest's pythonpath, `python -m`, etc.) Python's PathFinder
# treats ./SwarmSwIM/ as an empty namespace package and shadows the editable
# install. Promote the editable-install finder so it answers `import SwarmSwIM`
# before PathFinder gets to it. Mirrored in /conftest.py for pytest startup.
for _finder in list(sys.meta_path):
    _mod = getattr(_finder, "__module__", "") or ""
    if "editable" in _mod and "swarmswim" in _mod.lower():
        sys.meta_path.remove(_finder)
        sys.meta_path.insert(0, _finder)
        break

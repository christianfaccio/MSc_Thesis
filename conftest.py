import sys

# The ./SwarmSwIM/ submodule directory has no top-level __init__.py (the
# real package lives at ./SwarmSwIM/SwarmSwIM/), so once pyproject's
# pythonpath=["."] adds the repo root to sys.path, Python's PathFinder
# treats ./SwarmSwIM/ as an empty namespace package and shadows the
# editable install. Promote the editable-install finder to the front of
# sys.meta_path so it resolves `SwarmSwIM` before PathFinder gets to it.
for _finder in list(sys.meta_path):
    _mod = getattr(_finder, "__module__", "") or ""
    if "editable" in _mod and "swarmswim" in _mod.lower():
        sys.meta_path.remove(_finder)
        sys.meta_path.insert(0, _finder)
        break

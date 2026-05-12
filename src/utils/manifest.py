"""
Read/write helpers for the precomputed-environments manifest.

The manifest is a small JSON file inside `data/envs/` that indexes every
bundle (env_*.nc), records the global parameters used to generate them,
and the source catalog. It's the entry point used by the Gym env at
reset() to pick a bundle without cracking open NetCDFs.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


def git_commit() -> str | None:
    """Best-effort short git commit, or None if not in a repo."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def write_manifest(
    out_dir: Path,
    environments: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    global_params: dict[str, Any],
) -> Path:
    """Write `manifest.json` describing the precomputed bundles."""
    if not environments:
        raise ValueError("Refusing to write an empty manifest.")

    manifest_path = Path(out_dir) / "manifest.json"
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "n_environments": len(environments),
        "global_params": global_params,
        "sources": [
            {
                "name": s["name"],
                "y": s["y"],
                "x": s["x"],
                "depth": s["depth"],
                "Q": s["Q"],
            }
            for s in sources
        ],
        "environments": environments,
    }
    with manifest_path.open("w") as f:
        json.dump(payload, f, indent=2)
    return manifest_path


def read_manifest(path: str | Path) -> dict[str, Any]:
    """Load `manifest.json`. Used by the Gym env at reset()."""
    with Path(path).open() as f:
        return json.load(f)

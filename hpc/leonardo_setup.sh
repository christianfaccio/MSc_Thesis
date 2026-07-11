#!/bin/bash
# One-time setup on a Leonardo LOGIN node (compute nodes have NO internet:
# everything that downloads — uv, python, pip packages — must run here).
#
# Usage:
#   ssh <user>@login.leonardo.cineca.it
#   git clone --recursive git@github.com:christianfaccio/MSc_Thesis.git $WORK/MSc_Thesis
#   cd $WORK/MSc_Thesis && bash hpc/leonardo_setup.sh
#
# $WORK (project space, big quota, not purged) is the right home for the repo,
# the venv and the datasets. $SCRATCH is purged after ~40 days — fine for runs/,
# risky for anything you can't regenerate.
set -euo pipefail

cd "$(dirname "$0")/.."

# uv (installs to ~/.local/bin)
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Python 3.10 venv (uv downloads a standalone build; no module load needed,
# so batch jobs don't depend on the site's module tree).
uv venv .venv --python 3.10
source .venv/bin/activate

# CPU-only torch: the training is CPU-bound (tiny MLPs, env stepping dominates)
# and the CPU wheel saves ~2 GB and any CUDA-module coupling. Install it FIRST
# so requirements.txt doesn't pull the default CUDA build as a dependency.
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt
(cd SwarmSwIM && uv pip install -e .)

echo
echo "Setup done. Next:"
echo "  1. Put the datasets under data/oceananigans/ (rsync from your Mac:"
echo "     rsync -avP data/oceananigans/ <user>@login.leonardo.cineca.it:$PWD/data/oceananigans/"
echo "     — or symlink the copies already generated on Leonardo)."
echo "  2. Edit hpc/train.sbatch: set #SBATCH --account (see: saldo -b)."
echo "  3. Submit:  sbatch hpc/train.sbatch src.multi_agent.ippo_oceananigans [flags...]"

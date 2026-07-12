#!/bin/bash
# One-time setup on a Leonardo LOGIN node (compute nodes have NO internet:
# everything that downloads — uv, python, pip packages — must run here).
#
# Usage:
#   ssh <user>@login.leonardo.cineca.it
#   git clone --recursive git@github.com:christianfaccio/MSc_Thesis.git ~/MSc_Thesis
#   cd ~/MSc_Thesis && bash hpc/leonardo_setup.sh
#
# Layout: repo + venv + runs/ live in $HOME (private, 50 GB quota — plenty for
# code, the ~2 GB venv and checkpoints). The big NetCDF datasets live in
# $CINECA_SCRATCH (private, no quota) and are symlinked into data/oceananigans.
# CAVEAT: scratch files untouched for ~40 days are PURGED — datasets are
# regenerable / re-rsyncable, but never keep runs/ there.
set -euo pipefail

cd "$(dirname "$0")/.."

# SwarmSwIM submodule (empty if the clone wasn't --recursive)
git submodule update --init

# Datasets on scratch, symlinked into the repo
mkdir -p "$CINECA_SCRATCH/oceananigans" data
ln -sfn "$CINECA_SCRATCH/oceananigans" data/oceananigans

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
echo "  1. Put the datasets in \$CINECA_SCRATCH/oceananigans (already symlinked as"
echo "     data/oceananigans). From your Mac:"
echo "     rsync -avP data/oceananigans/ <user>@login.leonardo.cineca.it:$CINECA_SCRATCH/oceananigans/"
echo "     — or move the copies already generated on Leonardo there."
echo "     NB: scratch is purged after ~40 days of no access; re-rsync if needed."
echo "  2. Edit hpc/train.sbatch: set #SBATCH --account (see: saldo -b)."
echo "  3. Submit:  sbatch hpc/train.sbatch src.multi_agent.ippo_oceananigans [flags...]"

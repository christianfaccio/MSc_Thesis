#!/bin/bash
# One training run on Leonardo Booster.
#
#   sbatch hpc/train.sbatch src.multi_agent.ippo_oceananigans [trainer flags...]
#   sbatch hpc/train.sbatch src.single_agent.ppo_oceananigans --target-mode tail
#
# Sizing (see hpc/README.md): 16 cores fit one run at the default num_envs=12
# (12 rollout workers + 4 eval workers + main; rollout and eval never run at
# the same time, so peak concurrency is ~13). Booster allocates cores in
# GPU-share bundles of 8, and accounting bills the DOMINANT resource share —
# 16 of 32 cores = half a node = 2 A100-hours per hour even though the GPU
# idles. If your budget also covers the DCGP (CPU) partition, run there
# instead: swap the partition/gres lines as commented below.
#
#SBATCH --job-name=marl-train
#SBATCH --account=uTS26_Tornator       # your project budget: `saldo -b`
#SBATCH --partition=boost_usr_prod     # DCGP alternative: dcgp_usr_prod
#SBATCH --qos=normal                   # boost_qos_dbg = 30 min, fast queue, for smoke tests
#SBATCH --time=4:00:00                 # 10M steps ≈ a few hours at 12 parallel envs; 24h is the partition max
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1                   # Booster minimum; REMOVE this line on DCGP
#SBATCH --mem=64G                      # ~12 GB loader caches + slack
#SBATCH --output=hpc/logs/%x-%j.out
set -euo pipefail

MODULE="${1:?usage: sbatch hpc/train.sbatch <python -m module> [flags...]}"
shift

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

# One BLAS/OpenMP thread per process: parallelism comes from the env worker
# processes, and letting numpy/torch each spin up a thread pool on top of
# 13+ processes oversubscribes the allocation.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# --no-cuda: the nets are tiny and per-step host<->device transfers cost more
# than they save; the GPU in this allocation is a billing artifact, not a resource.
python -m "$MODULE" --no-cuda "$@"

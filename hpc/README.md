# Running the trainers on Leonardo (CINECA)

The trainers are CPU-bound (env stepping in worker processes; the MLPs are
tiny), so Leonardo is used as a many-core CPU machine. The `parallelization`
branch's async stepping is what makes the cores count: one run keeps
`num_envs` worker processes busy (plus `eval_workers` during the periodic
greedy eval).

## Storage layout

- `$HOME` (private, 50 GB quota): repo, venv, `runs/` — everything you can't
  afford to lose.
- `$CINECA_SCRATCH` (private, no quota, **purged after ~40 days of no access**):
  the NetCDF datasets, symlinked into the repo as `data/oceananigans` by the
  setup script. Regenerable/re-rsyncable, so purge is an inconvenience, not a
  loss — but never point `runs/` there.
- `$WORK` is per-project and group-readable by project collaborators; not used
  here.

## One-time setup (login node — compute nodes have no internet)

```bash
ssh <user>@login.leonardo.cineca.it
git clone --recursive git@github.com:christianfaccio/MSc_Thesis.git ~/MSc_Thesis
cd ~/MSc_Thesis
bash hpc/leonardo_setup.sh          # uv + python 3.10 venv + CPU torch + deps + SwarmSwIM + scratch symlink
# datasets (~20 GB each), one-time:
#   from the Mac:  rsync -avP data/oceananigans/ <user>@login.leonardo.cineca.it:'$CINECA_SCRATCH'/oceananigans/
#   or move the files already generated on Leonardo into $CINECA_SCRATCH/oceananigans/
```

Then set `#SBATCH --account=...` in `hpc/train.sbatch` (budgets: `saldo -b`).

## Submitting

```bash
cd ~/MSc_Thesis && mkdir -p hpc/logs
sbatch hpc/train.sbatch src.multi_agent.ippo_oceananigans --target-mode tail --seed 1
sbatch hpc/train.sbatch src.single_agent.ppo_oceananigans --target-mode tail --seed 1
# smoke test first (30-min debug QoS, fast queue):
sbatch --qos=boost_qos_dbg --time=00:30:00 hpc/train.sbatch \
    src.multi_agent.ippo_oceananigans --total-timesteps 200000
```

Monitor: `squeue --me`, `tail -f hpc/logs/marl-train-<jobid>.out`.
Checkpoints resume across walltime kills:
`sbatch hpc/train.sbatch src.multi_agent.ippo_oceananigans --resume runs/<run>/checkpoints/latest.pt`
(mind that a resumed run logs to a NEW run dir).

## How many cores?

Booster nodes: 32-core Ice Lake + 4×A100. Cores are bundled as **8 per GPU
share**, and accounting bills the dominant fraction of the node — cores or
GPUs, whichever is larger. The GPU itself is useless here (`--no-cuda`).

| allocation | fits | notes |
|---|---|---|
| **16 cores, 1 GPU (default in train.sbatch)** | one run at the standard `num_envs=12` | 12 rollout + 4 eval workers + main; rollout and eval alternate, so ~13 concurrently active. Billed as half a node (= 2 GPU-h/h). **Recommended: keeps runs hyperparameter-identical to the Mac ones.** |
| 8 cores, 1 GPU | one run at `--num-envs 8` | cheapest Booster slot, but batch 8·512 ≠ 12·512 — a different experiment, don't mix with existing curves |
| 32 cores (full node) | two standard runs | launch two `sbatch` jobs or two `python` lines backgrounded in one script; billed as the full node (4 GPU-h/h) |
| DCGP partition (`dcgp_usr_prod`) | same layouts, no GPU billing | pure-CPU nodes; use if `saldo -b` shows a DCGP budget — swap the partition line and drop `--gres` |

Rules of thumb: cores ≈ `num_envs + max(eval_workers, 1) − eval overlap ≈ num_envs + 1..4`; RAM ≈ `num_envs · max_cached_loaders · 90 MB` + ~2 GB (64 GB is generous for the defaults). Scaling `num_envs` beyond 12 buys more speed but changes the batch size — treat it as a deliberate hyperparameter change (scale `num_minibatches` with it).

## Getting results back

```bash
# from the Mac — TensorBoard runs + checkpoints:
rsync -avP <user>@login.leonardo.cineca.it:MSc_Thesis/runs/ runs/
# or live TensorBoard through a tunnel:
ssh -L 6006:localhost:6006 <user>@login.leonardo.cineca.it \
    'cd ~/MSc_Thesis && source .venv/bin/activate && tensorboard --logdir runs --port 6006'
```

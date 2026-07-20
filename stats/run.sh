for P in random gradient checkpoint; do
  python stats/eval_oceananigans_stats.py \
      --checkpoint runs/mappo_buoyancy_history/checkpoints/latest.pt \
      --policy $P --netcdf-file data/oceananigans/buoyancy_active/train \
      --episodes 100 --seed 0
done

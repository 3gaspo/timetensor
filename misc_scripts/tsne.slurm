#!/bin/bash
#SBATCH --job-name=tsne
#SBATCH --gres=gpu:1
#SBATCH --output=new_scripts/tsne.out
#SBATCH --error=new_scripts/tsne.err
#SBATCH --time=02:00:00
#SBATCH --partition=a100
source .venv/bin/activate

output_dir="../outputs/tsne/"
# rm -rf "$output_dir"
# mkdir -p "$output_dir"

for data in electricity solar traffic synthetic
do
    srun python3 tsne.py "data=$data" "task.lags=336" "task.horizon=48" "training.bs=1" "data.splits.idx_mode='random'" "misc.output_dir=$output_dir"
done
# nohup bash scripts/read.sh > read.log 2>&1 &
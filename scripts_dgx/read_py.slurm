#!/bin/bash
#SBATCH --job-name=read
#SBATCH --gres=gpu:1
#SBATCH --output=scripts/read_py.out
#SBATCH --error=scripts/read_py.err
#SBATCH --time=01:00:00
#SBATCH --partition=a100
source .venv/bin/activate

srun python3 load_dataset.py "data=saturation" "task.lags=10" "task.horizon=3" "training.bs=10" "load.rebuild=True" "load.example=True" "load.replot=True" "load.heterogeneity=False" "load.clusters=False" "load.logs=True" "data.splits.remove_train_cte=False" "data.splits.remove_eval_cte=False"

# nohup bash scripts/read.sh > read.log 2>&1 &
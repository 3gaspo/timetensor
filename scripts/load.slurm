#!/bin/bash
#SBATCH --job-name=load
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/load.out
#SBATCH --error=script_outputs/load.err
#SBATCH --time=02:00:00
#SBATCH --partition=a100
source .venv/bin/activate

for data in electricity #solar traffic synthetic
do
    srun python3 load_dataset.py "data.sampling.use_context=False" "data=$data" "task.lags=168" "task.horizon=24" "load.rebuild=True" "load.example=True" "load.shapes=True"
done
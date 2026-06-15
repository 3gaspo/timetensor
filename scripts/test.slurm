#!/bin/bash
#SBATCH --job-name=test
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/test.out
#SBATCH --error=script_outputs/test.err
#SBATCH --time=02:00:00
#SBATCH --partition=a100
source .venv/bin/activate

data=electricity
model=chronos

srun python3 test_chronos.py "data=$data" "model=$model"
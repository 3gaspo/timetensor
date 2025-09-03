#!/bin/bash
#SBATCH --job-name=read
#SBATCH --gres=gpu:1
#SBATCH --output=scripts/read_py.out
#SBATCH --error=scripts/read_py.err
#SBATCH --time=01:00:00
#SBATCH --mem=20G
source .venv/bin/activate

srun python3 load_dataset.py "data=solar"

# nohup bash scripts/read.sh > read.log 2>&1 &
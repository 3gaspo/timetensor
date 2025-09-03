#!/bin/bash
#SBATCH --job-name=fed
#SBATCH --gres=gpu:1
#SBATCH --output=scripts/fed.out
#SBATCH --error=scripts/fed.err
#SBATCH --time=23:00:00
#SBATCH --mem=40G
source .venv/bin/activate

main_output_dir="../outputs/fed/"

rm -rf "$main_output_dir"
mkdir -p "$main_output_dir"

data=synthetic

model_name=PatchTST
loss=NMSE
lr=0.00001
epochs=10
bs=100
eval_freq=100

seed=1
output_dir="${main_output_dir}seed_$seed/"
    
srun python3 load_dataset.py "data=$data" "misc.seed=$seed" "misc.verbose=0"

normalization=revin

srun python3 train_fedavg.py \
    "misc.seed=$seed" \
    "data=$data" \
    "data.reshuffle=True" \
    "misc.output_dir=$output_dir" \
    "misc.benchmark=True" \
    "model=$model_name" \
    "normalization=$normalization" \
    "training.loss=$loss" \
    "training.bs=$bs" \
    "training.lr=$lr" \
    "training.epochs=$epochs" \
    "training.eval_freq=$eval_freq" \
    "training.print_freq=100" \
    "data.clusters='clusters/'" \
    "misc.save_name=$normalization"

normalization=mIN

srun python3 train_fedavg.py \
    "misc.seed=$seed" \
    "data=$data" \
    "data.reshuffle=True" \
    "misc.output_dir=$output_dir" \
    "misc.benchmark=True" \
    "model=$model_name" \
    "normalization=$normalization" \
    "training.loss=$loss" \
    "training.bs=$bs" \
    "training.lr=$lr" \
    "training.epochs=$epochs" \
    "training.eval_freq=$eval_freq" \
    "training.print_freq=100" \
    "data.clusters='clusters/'" \
    "misc.save_name=$normalization"
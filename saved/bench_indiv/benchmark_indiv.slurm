#!/bin/bash
#SBATCH --job-name=bench_indiv -p h100
#SBATCH --gres=gpu:1 -p h100
#SBATCH --output=scripts/bench_indiv.out -p h100
#SBATCH --error=scripts/bench_indiv.err -p h100
#SBATCH --time=23:00:00 -p h100
#SBATCH --mem=40G -p h100
source .venv/bin/activate

output_dir="../outputs/bench_indiv/"
data=electricity
seed=1

rm -rf "$output_dir"
mkdir -p "$output_dir"

model_name=PatchTST
lr=0.0001
eval_freq=100

loss=NMSE
normalization=revin

by_idx=dates
epochs=1
bs=5
srun python3 train_model.py \
    "model=${model_name}" \
    "misc.seed=$seed" \
    "misc.output_dir=$output_dir" \
    "normalization=$normalization" \
    "training.loss=$loss" \
    "misc.benchmark=True" \
    "training.lr=$lr" \
    "training.epochs=$epochs" \
    "training.bs=$bs" \
    "training.eval_freq=$eval_freq" \
    "training.print_freq=100" \
    "data=$data" \
    "data.by_idx=$by_idx" \
    "data.reshuffle=True" \
    "misc.save_name=TST_bs${bs}_$by_idx"


by_idx=individuals
epochs=300
bs=28
srun python3 train_model.py \
    "model=${model_name}" \
    "misc.seed=$seed" \
    "misc.output_dir=$output_dir" \
    "normalization=$normalization" \
    "training.loss=$loss" \
    "misc.benchmark=True" \
    "training.lr=$lr" \
    "training.epochs=$epochs" \
    "training.bs=$bs" \
    "training.eval_freq=$eval_freq" \
    "training.print_freq=100" \
    "data=$data" \
    "data.by_idx=$by_idx" \
    "data.reshuffle=True" \
    "misc.save_name=TST_bs${bs}_$by_idx"

by_idx=individuals
epochs=300
bs=56
srun python3 train_model.py \
    "model=${model_name}" \
    "misc.seed=$seed" \
    "misc.output_dir=$output_dir" \
    "normalization=$normalization" \
    "training.loss=$loss" \
    "misc.benchmark=True" \
    "training.lr=$lr" \
    "training.epochs=$epochs" \
    "training.bs=$bs" \
    "training.eval_freq=$eval_freq" \
    "training.print_freq=100" \
    "data=$data" \
    "data.by_idx=$by_idx" \
    "data.reshuffle=True" \
    "misc.save_name=TST_bs${bs}_$by_idx"

by_idx=individuals
epochs=300
bs=28
lr=1e-5
srun python3 train_model.py \
    "model=${model_name}" \
    "misc.seed=$seed" \
    "misc.output_dir=$output_dir" \
    "normalization=$normalization" \
    "training.loss=$loss" \
    "misc.benchmark=True" \
    "training.lr=$lr" \
    "training.epochs=$epochs" \
    "training.bs=$bs" \
    "training.eval_freq=$eval_freq" \
    "training.print_freq=100" \
    "data=$data" \
    "data.by_idx=$by_idx" \
    "data.reshuffle=True" \
    "misc.save_name=TST_bs${bs}_${by_idx}_lr$lr"

srun python3 -c "from src.timetensor.visu import plot_expe;plot_expe('$output_dir', $eval_freq)"

multipliers="-7 0"
srun python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}test1_mean_results.json', multipliers='$multipliers')"
srun python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}test2_mean_results.json', multipliers='$multipliers')"
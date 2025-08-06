#!/bin/bash
#SBATCH --job-name=bench_sk -p h100
#SBATCH --gres=gpu:1 -p h100
#SBATCH --output=scripts/bench_sk.out -p h100
#SBATCH --error=scripts/bench_sk.err -p h100
#SBATCH --time=23:00:00 -p h100
#SBATCH --mem=40G -p h100
source .venv/bin/activate

output_dir="../outputs/bench_sk/"
data=electricity
seed=None

rm -rf "$output_dir"
mkdir -p "$output_dir"


lr=0.0001
epochs=300
loss=MSE
eval_freq=100

model_name=sklinear
seed=1
by_idx=dates
for normalization in None instance relative
do
    srun python3 train_model.py \
        "model=${model_name}" \
        "normalization=$normalization" \
        "misc.output_dir=$output_dir" \
        "misc.benchmark=True" \
        "data=$data" \
        "data.by_idx=$by_idx" \
        "data.reshuffle=True" \
        "misc.seed=$seed"
done


by_idx=individuals
normalization=instance
srun python3 train_model.py \
    "model=${model_name}" \
    "normalization=$normalization" \
    "misc.output_dir=$output_dir" \
    "misc.benchmark=True" \
    "data=$data" \
    "data.by_idx=$by_idx" \
    "data.reshuffle=True" \
    "misc.seed=$seed" \
    "misc.save_name=sklinear_instance_indiv"


multipliers="-7 0"
srun python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}test1_mean_results.json', multipliers='$multipliers')"

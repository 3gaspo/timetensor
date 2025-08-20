#!/bin/bash
#SBATCH --job-name=bench_linear -p h100
#SBATCH --gres=gpu:1 -p h100
#SBATCH --output=scripts/benchmark_linear.out -p h100
#SBATCH --error=scripts/benchmark_linear.err -p h100
#SBATCH --time=23:00:00 -p h100
#SBATCH --mem=40G -p h100
source .venv/bin/activate

output_dir="../outputs/benchmark_linear/"
data=electricity
seed=1

rm -rf "$output_dir"
mkdir -p "$output_dir"

#repeat
srun python3 train_model.py \
    "model=repeat" \
    "misc.output_dir=$output_dir" \
    "misc.benchmark=True" \
    "data=$data" \
    "data.reshuffle=True" \
    "misc.seed=$seed"

lr=0.0001
epochs=300
eval_freq=100

for loss in MSE NMSE RMSE
do
    for model_name in linear DLinear
    do
        for normalization in None instance
        do
            srun python3 train_model.py \
                "model=${model_name}" \
                "normalization=$normalization" \
                "training.loss=$loss" \
                "training.lr=$lr" \
                "training.epochs=$epochs" \
                "training.eval_freq=$eval_freq" \
                "training.print_freq=100" \
                "misc.output_dir=$output_dir" \
                "misc.benchmark=True" \
                "data=$data" \
                "data.reshuffle=True" \
                "misc.seed=$seed"
        done
    done
done


model_name=sklinear
by_idx=dates
for normalization in None instance relative
do
    srun python3 train_model.py \
        "model=$model_name" \
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
    "model=$model_name" \
    "normalization=$normalization" \
    "misc.output_dir=$output_dir" \
    "misc.benchmark=True" \
    "data=$data" \
    "data.by_idx=$by_idx" \
    "data.reshuffle=True" \
    "misc.seed=$seed" \
    "misc.save_name=sklinear_instance_indiv"

srun python3 -c "from src.timetensor.visu import plot_expe;plot_expe('$output_dir', $eval_freq)"

multipliers="-7 0"
srun python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}test1_mean_results.json', multipliers='$multipliers')"
srun python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}test2_mean_results.json', multipliers='$multipliers')"
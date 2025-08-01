#!/bin/bash
#SBATCH --job-name=bench_linear
#SBATCH --gres=gpu:1
#SBATCH --output=scripts/benchmark_linear.out
#SBATCH --error=scripts/benchmark_linear.err
#SBATCH --time=23:00:00
#SBATCH --mem=40G
source .venv/bin/activate

output_dir="../outputs/benchmark_linear/"

# rm -rf "$output_dir"
# mkdir -p "$output_dir"

# #lookback
# model_name=repeat
# srun python3 train_model.py \
#     "model=${model_name}" \
#     "misc.output_dir=$output_dir" \
#     "misc.benchmark=True"

data=electricity
lr=0.0001
epochs=200
latent=False
loss=MSE
seed=1

# for loss in MSE NMSE
# do
#     for model_name in linear DLinear
#     do
#         for normalization in None instance
#         do
#             srun python3 train_model.py \
#                 "misc.seed=$seed" \
#                 "model=${model_name}" \
#                 "misc.output_dir=$output_dir" \
#                 "normalization=$normalization" \
#                 "training.loss=$loss" \
#                 "misc.benchmark=True" \
#                 "training.lr=$lr" \
#                 "training.epochs=$epochs" \
#                 "training.eval_freq=100" \
#                 "training.print_freq=100" \
#                 "data=$data" \
#                 "data.reshuffle=True"
#         done
#     done
# done

srun python3 train_model.py \
    "misc.seed=$seed" \
    "model=DLinear" \
    "misc.output_dir=$output_dir" \
    "normalization=instance" \
    "training.loss=NMSE" \
    "misc.benchmark=True" \
    "training.lr=$lr" \
    "training.epochs=$epochs" \
    "training.eval_freq=100" \
    "training.print_freq=100" \
    "data=$data" \
    "data.reshuffle=True"

# model_name=sklinear
# for normalization in None instance relative
# do
#     srun python3 train_model.py \
#         "model=${model_name}" \
#         "misc.seed=$seed" \
#         "misc.output_dir=$output_dir" \
#         "normalization=$normalization" \
#         "training.loss=$loss" \
#         "misc.benchmark=True" \
#         "data=$data" \
#         "data.reshuffle=True"
# done


srun python3 -c "from src.timetensor.visu import plot_expe;plot_expe('$output_dir', $eval_freq)"

multipliers="-8 -3 0 0"
srun python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}test1_mean_results.json', multipliers='$multipliers')"
srun python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}test2_mean_results.json', multipliers='$multipliers')"
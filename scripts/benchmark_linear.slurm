#!/bin/bash
#SBATCH --job-name=bench_linear
#SBATCH --gres=gpu:1
#SBATCH --output=scripts/benchmark_linear.out
#SBATCH --error=scripts/benchmark_linear.err
#SBATCH --time=23:00:00
#SBATCH --mem=40G
source .venv/bin/activate

output_dir="../outputs/benchmark_tst/"
rm -rf "$output_dir"
mkdir -p "$output_dir"

#lookback
model_name=repeat
srun python3 train_model.py \
    "model=${model_name}" \
    "misc.output_dir=$output_dir" \
    "misc.benchmark=True"

data=electricity
model_name=PatchTST
lr=0.0001
epochs=200
latent=False
loss=MSE

for loss in MSE NMSE
do
    for model_name in linear DLinear
    do
        for normalization in None instance
        do
            python3 train_model.py \
                "model=${model_name}" \
                "misc.output_dir=$output_dir" \
                "normalization=$normalization" \
                "training.loss=$loss" \
                "misc.benchmark=True" \
                "training.lr=$lr" \
                "training.epochs=$epochs" \
                "training.eval_freq=100" \
                "training.print_freq=100" \
                "data=$data" \
                "data.reshuffle=False"
        done
    done
done

python3 -c "from src.timetensor.visu import plot_expe;plot_expe('$output_dir', $eval_freq)"

multipliers="-8 -3 0 0"
python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}test1_mean_results.json', multipliers='$multipliers')"
python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}test2_mean_results.json', multipliers='$multipliers')"
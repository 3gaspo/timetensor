#!/bin/bash
#SBATCH --job-name=bench_tst
#SBATCH --gres=gpu:1
#SBATCH --output=scripts/benchmark_tst.out
#SBATCH --error=scripts/benchmark_tst.err
#SBATCH --time=23:00:00
#SBATCH --mem=40G
source .venv/bin/activate

output_dir="../outputs/benchmark_tst/"
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

model_name=PatchTST
lr=0.0001
epochs=300
eval_freq=100

for loss in MSE NMSE
do
    for normalization in instance revin
    do
        srun python3 train_model.py \
            "model=${model_name}" \
            "misc.seed=$seed" \
            "misc.output_dir=$output_dir" \
            "normalization=$normalization" \
            "training.loss=$loss" \
            "misc.benchmark=True" \
            "training.lr=$lr" \
            "training.epochs=$epochs" \
            "training.eval_freq=$eval_freq" \
            "training.print_freq=100" \
            "data=$data" \
            "data.reshuffle=True"
    done
done

srun python3 -c "from src.timetensor.visu import plot_expe;plot_expe('$output_dir', $eval_freq)"

multipliers="-7 0"
srun python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}test1_mean_results.json', multipliers='$multipliers')"
srun python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}test2_mean_results.json', multipliers='$multipliers')"
#!/bin/bash
#SBATCH --job-name=bench_base -p h100
#SBATCH --gres=gpu:1 -p h100
#SBATCH --output=scripts/benchmark_baselines.out -p h100
#SBATCH --error=scripts/benchmark_baselines.err -p h100
#SBATCH --time=10:00:00 -p h100
#SBATCH --mem=30G -p h100
source .venv/bin/activate

output_dir="../outputs/benchmark_baselines/"

rm -rf "$output_dir"
mkdir -p "$output_dir"

data=electricity
seed=1

for model_name in persistence repeat expected
do
    srun python3 train_model.py \
        "model=${model_name}" \
        "normalization=None" \
        "misc.output_dir=$output_dir" \
        "misc.benchmark=True" \
        "misc.complete_evaluation=True" \
        "data=$data" \
        "data.reshuffle=True" \
        "misc.seed=$seed"
done

model_name=lookback
for lookback_idx in 0 168
do
    srun python3 train_model.py \
        "model=${model_name}" \
        "normalization=None" \
        "model.configs.lookback_idx=$lookback_idx" \
        "misc.benchmark=True" \
        "misc.complete_evaluation=True" \
        "misc.output_dir=$output_dir" \
        "data=$data" \
        "data.reshuffle=True" \
        "misc.seed=$seed" \
        "misc.save_name=lookback_$lookback_idx"
done

multipliers="-8 -3 0 0"
srun python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}test1_mean_results.json', multipliers='$multipliers')"
srun python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}test2_mean_results.json', multipliers='$multipliers')"
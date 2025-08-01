#!/bin/bash
#SBATCH --job-name=bench_repeat
#SBATCH --gres=gpu:1
#SBATCH --output=scripts/bench_repeat.out
#SBATCH --error=scripts/bench_repeat.err
#SBATCH --time=1:00:00
#SBATCH --mem=10G
source .venv/bin/activate

output_dir="../outputs/bench_repeat/"

rm -rf "$output_dir"
mkdir -p "$output_dir"

#lookback

data=electricity
for seed in 1 2 3
do
    for data in sim1 electricity
    do
        model_name=repeat
        srun python3 train_model.py \
            "model=$model_name" \
            "misc.seed=$seed" \
            "data.reshuffle=True" \
            "misc.output_dir=$output_dir" \
            "misc.benchmark=True" \
            "normalization=None" \
            "misc.save_name=${data}_repeat_${seed}"
    done
done

multipliers="-8 -3 0 0"
srun python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}test1_mean_results.json', multipliers='$multipliers')"
srun python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}test2_mean_results.json', multipliers='$multipliers')"
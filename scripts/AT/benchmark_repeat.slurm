#!/bin/bash
#SBATCH --job-name=bench_repeat -p h100
#SBATCH --gres=gpu:1 -p h100
#SBATCH --output=scripts/bench_repeat.out -p h100
#SBATCH --error=scripts/bench_repeat.err -p h100
#SBATCH --time=23:00:00 -p h100
#SBATCH --mem=40G -p h100
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
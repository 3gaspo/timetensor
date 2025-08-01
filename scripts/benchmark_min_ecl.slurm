#!/bin/bash
#SBATCH --job-name=min_ecl
#SBATCH --gres=gpu:1
#SBATCH --output=scripts/benchmark_min_ecl.out
#SBATCH --error=scripts/benchmark_min_ecl.err
#SBATCH --time=23:00:00
#SBATCH --mem=40G
source .venv/bin/activate

main_output_dir="../outputs/benchmark_min_ecl/"

rm -rf "$main_output_dir"
mkdir -p "$main_output_dir"

data=electricity

model_name=PatchTST
loss=MSE
lr=0.0001
epochs=300

for seed in 1 2 3 4 5
do
    output_dir="${main_output_dir}seed_$seed/"
    
    normalization=revin
    srun python3 train_model.py \
        "model=${model_name}" \
        "misc.seed=$seed" \
        "misc.output_dir=$output_dir" \
        "normalization=$normalization" \
        "training.loss=$loss" \
        "misc.benchmark=True" \
        "training.bs=10" \
        "training.lr=$lr" \
        "training.epochs=$epochs" \
        "training.eval_freq=100" \
        "training.print_freq=100" \
        "data=$data" \
        "data.reshuffle=True" \
        "task.lags=100" \
        "task.horizon=20"

    normalization=mIN
    srun python3 train_model.py \
        "model=${model_name}" \
        "misc.seed=$seed" \
        "misc.output_dir=$output_dir" \
        "normalization=$normalization" \
        "training.loss=$loss" \
        "misc.benchmark=True" \
        "training.bs=10" \
        "training.lr=$lr" \
        "training.epochs=$epochs" \
        "training.eval_freq=100" \
        "training.print_freq=100" \
        "data=$data" \
        "data.reshuffle=True" \
        "task.lags=100" \
        "task.horizon=20" \
        "normalization.configs.use_gamma=True" \
        "normalization.configs.inverse_gamma=True" \
        "misc.save_name=min_gamma"

    srun python3 -c "from src.timetensor.visu import plot_expe;plot_expe('$output_dir', $eval_freq)"

done

multipliers="3 3 4 6"
srun python3 -c "from src.timetensor.visu import print_nice_tables;print_nice_tables('$main_output_dir', 'test1_mean_results.json', 5, multipliers='$multipliers', baseline='PatchTST_revin')"
srun python3 -c "from src.timetensor.visu import print_nice_tables;print_nice_tables('$main_output_dir', 'test2_mean_results.json', 5, multipliers='$multipliers', baseline='PatchTST_revin')"
srun python3 -c "from src.timetensor.visu import get_boxplots;get_boxplots('$main_output_dir', 'test1_mean_results.json', 5, 'Test MSE', '$main_output_dir', 'test1_boxplot.pdf')"
srun python3 -c "from src.timetensor.visu import get_boxplots;get_boxplots('$main_output_dir', 'test2_mean_results.json', 5, 'Test MSE', '$main_output_dir', 'test2_boxplot.pdf')"
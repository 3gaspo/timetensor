#!/bin/bash
#SBATCH --job-name=min_1
#SBATCH --gres=gpu:1
#SBATCH --output=scripts/benchmark_min_1.out
#SBATCH --error=scripts/benchmark_min_1.err
#SBATCH --time=23:00:00
#SBATCH --mem=40G
source .venv/bin/activate

main_output_dir="../outputs/benchmark_min_1_2/"

rm -rf "$main_output_dir"
mkdir -p "$main_output_dir"

data=sim1

model_name=PatchTST
loss=NMSE
lr=0.0001
epochs=300

for seed in 1 2 3 4 5
do
    output_dir="${main_output_dir}seed_$seed/"
    
    for normalization in instance revin
    do
        srun python3 train_model.py \
            "misc.seed=$seed" \
            "model=$model_name" \
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
    done

    normalization=mIN
    alpha=0.948
    beta=0.781

    for use_gamma in False True
    do
        if [ "$use_gamma" = "True" ]; then
            suffix="_gamma"
        else
            suffix=""
        fi
        #default
        srun python3 train_model.py \
            "misc.seed=$seed" \
            "model=$model_name" \
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
            "normalization.configs.use_gamma=$use_gamma" \
            "normalization.configs.inverse_gamma=$use_gamma" \
            "misc.save_name=default$suffix"

        #init
        srun python3 train_model.py \
            "misc.seed=$seed" \
            "model=$model_name" \
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
            "normalization.configs.init_beta=$beta" \
            "normalization.configs.init_alpha=$alpha" \
            "normalization.configs.use_gamma=$use_gamma" \
            "normalization.configs.inverse_gamma=$use_gamma" \
            "misc.save_name=init$suffix"

        #fixed
        srun python3 train_model.py \
            "misc.seed=$seed" \
            "model=$model_name" \
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
            "normalization.configs.init_beta=$beta" \
            "normalization.configs.init_alpha=$alpha" \
            "normalization.configs.fixed_beta=True" \
            "normalization.configs.fixed_alpha=True " \
            "normalization.configs.use_gamma=$use_gamma" \
            "normalization.configs.inverse_gamma=$use_gamma" \
            "misc.save_name=fixed$suffix"
    done
    
    srun python3 -c "from src.timetensor.visu import plot_expe;plot_expe('$output_dir', $eval_freq)"

done

multipliers="3 3 4 6"
srun python3 -c "from src.timetensor.visu import print_nice_tables;print_nice_tables('$main_output_dir', 'test1_mean_results.json', 5, multipliers='$multipliers', baseline='PatchTST_revin')"
srun python3 -c "from src.timetensor.visu import print_nice_tables;print_nice_tables('$main_output_dir', 'test2_mean_results.json', 5, multipliers='$multipliers', baseline='PatchTST_revin')"
srun python3 -c "from src.timetensor.visu import get_boxplots;get_boxplots('$main_output_dir', 'test1_mean_results.json', 5, 'Test MSE', '$main_output_dir', 'test1_boxplot.pdf')"
srun python3 -c "from src.timetensor.visu import get_boxplots;get_boxplots('$main_output_dir', 'test2_mean_results.json', 5, 'Test MSE', '$main_output_dir', 'test2_boxplot.pdf')"
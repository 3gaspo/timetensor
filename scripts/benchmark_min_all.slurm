#!/bin/bash
#SBATCH --job-name=min_all
#SBATCH --gres=gpu:1
#SBATCH --output=scripts/benchmark_min_all.out
#SBATCH --error=scripts/benchmark_min_all.err
#SBATCH --time=23:00:00
#SBATCH --mem=40G

source .venv/bin/activate

main_output_dir="../outputs/benchmark_min_all/"

rm -rf "$main_output_dir"
mkdir -p "$main_output_dir"

data=sim

model_name=PatchTST
loss=NMSE
lr=0.0001
epochs=300
seed=1

    
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

    #default
    normalization=mIN
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
        "misc.save_name=min"

    #cmin
    normalization=cmIN
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
        "normalization.configs.init_alphas=2" \
        "normalization.configs.init_betas=2" \
        "misc.save_name=cmin_default"

    #init
    normalization=cmIN
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
        "normalization.configs.init_alphas='0.948;0.948'" \
        "normalization.configs.init_betas='0.781;-0.781'" \
        "misc.save_name=cmin_init"

    #init + gamma
    normalization=cmIN
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
        "normalization.configs.init_alphas='0.948;0.948'" \
        "normalization.configs.init_betas='0.781;-0.781'" \
        "normalization.configs.use_gamma=True" \
        "normalization.configs.inverse_gamma=True" \
        "misc.save_name=cmin_init_gamma"

    srun python3 -c "from src.timetensor.visu import plot_expe;plot_expe('$output_dir', $eval_freq)"


multipliers="3 3 4 6"
srun python3 -c "from src.timetensor.visu import print_nice_tables;print_nice_tables('$main_output_dir', 'test1_mean_results.json', 5, multipliers='$multipliers', baseline='PatchTST_revin')"
srun python3 -c "from src.timetensor.visu import print_nice_tables;print_nice_tables('$main_output_dir', 'test2_mean_results.json', 5, multipliers='$multipliers', baseline='PatchTST_revin')"
srun python3 -c "from src.timetensor.visu import get_boxplots;get_boxplots('$main_output_dir', 'test1_mean_results.json', 5, 'Test MSE', '$main_output_dir', 'test1_boxplot.pdf')"
srun python3 -c "from src.timetensor.visu import get_boxplots;get_boxplots('$main_output_dir', 'test2_mean_results.json', 5, 'Test MSE', '$main_output_dir', 'test2_boxplot.pdf')"
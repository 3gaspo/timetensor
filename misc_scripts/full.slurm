#!/bin/bash
#SBATCH --job-name=d_syn_100_100
#SBATCH --gres=gpu:1
#SBATCH --output=new_scripts/synthetic/d_syn_100_100.out
#SBATCH --error=new_scripts/synthetic/d_syn_100_100.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100

source .venv/bin/activate

data=synthetic
lags=100
horizon=100
bs=256
model_name=DLinear

main_output_dir="../outputs/synthetic/${model_name}/"
eval_freq=100
lr=1e-5
epochs=1000

big_output_dir="${main_output_dir}lags${lags}_horizon${horizon}/"
rm -rf "$big_output_dir"
mkdir -p "$big_output_dir"
for seed in 1 2 3 4 5
do
    output_dir="${big_output_dir}seed_$seed/"

    for loss in MSE NMSE
    do
        for normalization in None instance revin
        do
            srun python3 train_model.py \
                "misc.seed=$seed" \
                "data=$data" \
                "misc.output_dir=$output_dir" \
                "model=$model_name" \
                "normalization=$normalization" \
                "task.lags=$lags" \
                "task.horizon=$horizon" \
                "training.loss=$loss" \
                "training.bs=$bs" \
                "training.lr=$lr" \
                "training.epochs=$epochs" \
                "training.eval_freq=$eval_freq" \
                "training.print_freq=100" \
                "misc.save_name=${normalization}_${loss}"
        done
    done

    normalization=instance
    for loss in MSE NMSE
    do
        srun python3 train_model.py \
            "misc.seed=$seed" \
            "data=$data" \
            "misc.output_dir=$output_dir" \
            "model=$model_name" \
            "normalization=$normalization" \
            "normalization.configs.latent=True" \
            "task.lags=$lags" \
            "task.horizon=$horizon" \
            "training.loss=$loss" \
            "training.bs=$bs" \
            "training.lr=$lr" \
            "training.epochs=$epochs" \
            "training.eval_freq=$eval_freq" \
            "training.print_freq=100" \
            "misc.save_name=og_instance_${loss}"
    done

    #softmin
    loss=NMSE
    normalization=softmin
    srun python3 train_model.py \
        "misc.seed=$seed" \
        "data=$data" \
        "misc.output_dir=$output_dir" \
        "model=$model_name" \
        "normalization=$normalization" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "training.loss=$loss" \
        "training.bs=$bs" \
        "training.lr=$lr" \
        "training.epochs=$epochs" \
        "training.eval_freq=$eval_freq" \
        "training.print_freq=100" \
        "misc.save_name=${normalization}_${loss}"

    #cmin
    loss=NMSE
    normalization=cmIN
    for init in True False
    do
        srun python3 train_model.py \
            "misc.seed=$seed" \
            "data=$data" \
            "misc.output_dir=$output_dir" \
            "model=$model_name" \
            "normalization=$normalization" \
            "normalization.configs.init_alpha=$init" \
            "normalization.configs.init_beta=$init" \
            "task.lags=$lags" \
            "task.horizon=$horizon" \
            "training.loss=$loss" \
            "training.bs=$bs" \
            "training.lr=$lr" \
            "training.epochs=$epochs" \
            "training.eval_freq=$eval_freq" \
            "training.print_freq=100" \
            "misc.save_name=cmIn_init$init"
    done

    #residuals
    loss=NMSE
    normalization=instance
    srun python3 train_model.py \
        "misc.seed=$seed" \
        "data=$data" \
        "misc.output_dir=$output_dir" \
        "model=$model_name" \
        "model.residuals=True" \
        "normalization=$normalization" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "training.loss=$loss" \
        "training.bs=$bs" \
        "training.lr=$lr" \
        "training.epochs=$epochs" \
        "training.eval_freq=$eval_freq" \
        "training.print_freq=100" \
        "misc.save_name=instance_residuals"

    srun python3 -c "from src.timetensor.visu import plot_expe;plot_expe('$output_dir', $eval_freq)"
done

multipliers="5 4"
baseline="instance_NMSE"
names="instance_NMSE;revin_NMSE;cmIn_initTrue;cmIn_initFalse;instance_residuals"
srun python3 -c "from src.timetensor.visu import get_boxplots;get_boxplots('$big_output_dir', 'test1_mean_results.json', 5, 'Test MSE', save_name='test1_mse_boxplot.pdf', names='$names')"
srun python3 -c "from src.timetensor.visu import get_boxplots;get_boxplots('$big_output_dir', 'test1_mean_results.json', 5, 'Test NMSE', save_name='test1_nmse_boxplot.pdf', names='$names')"
srun python3 -c "from src.timetensor.visu import get_multiple_expe_results;get_multiple_expe_results('${big_output_dir}', 'test1_mean_results.json', 5, multipliers='$multipliers', save_name='errors1.pdf', baseline='$baseline', names='$names')"
srun python3 -c "from src.timetensor.visu import get_multiple_expe_results;get_multiple_expe_results('${big_output_dir}', 'test2_mean_results.json', 5, multipliers='$multipliers', save_name='errors2.pdf', baseline='$baseline', names='$names')"

#!/bin/bash
#SBATCH --wckey=P12CU:DATASCIENCE
#SBATCH --partition=an
#SBATCH --qos=an_all_long_rd
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --job-name=solar_504_24
#SBATCH --output=scripts/solar/solar_504_24.out
#SBATCH --error=scripts/solar/solar_504_24.err

source .venv/bin/activate

data=solar
lags=504
horizon=24
bs=256

main_output_dir="../outputs/norm/${data}/"
model_name=PatchTST
eval_freq=100
lr=1e-5
epochs=1200

big_output_dir="${main_output_dir}lags${lags}_horizon${horizon}/"
rm -rf "$big_output_dir"
mkdir -p "$big_output_dir"
for seed in 1 2 3 4 5
do
    output_dir="${big_output_dir}seed_$seed/"

    for loss in NMSE
    do
        for normalization in instance revin
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

    loss=NMSE
    normalization=flexrevin
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

    #min init
    loss=NMSE
    normalization=mIN
    srun python3 train_model.py \
        "misc.seed=$seed" \
        "data=$data" \
        "misc.output_dir=$output_dir" \
        "model=$model_name" \
        "normalization=$normalization" \
        "normalization.configs.use_gamma=False" \
        "normalization.configs.inverse_gamma=False" \
        "normalization.configs.init_alpha=True" \
        "normalization.configs.init_beta=True" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "training.loss=$loss" \
        "training.bs=$bs" \
        "training.lr=$lr" \
        "training.epochs=$epochs" \
        "training.eval_freq=$eval_freq" \
        "training.print_freq=100" \
        "misc.save_name=min_init"
    
    #min + revin
    loss=NMSE
    normalization=mIN
    srun python3 train_model.py \
        "misc.seed=$seed" \
        "data=$data" \
        "misc.output_dir=$output_dir" \
        "model=$model_name" \
        "normalization=$normalization" \
        "normalization.configs.use_gamma=True" \
        "normalization.configs.inverse_gamma=True" \
        "normalization.configs.init_alpha=False" \
        "normalization.configs.init_beta=False" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "training.loss=$loss" \
        "training.bs=$bs" \
        "training.lr=$lr" \
        "training.epochs=$epochs" \
        "training.eval_freq=$eval_freq" \
        "training.print_freq=100" \
        "misc.save_name=min_revin"

    srun python3 -c "from src.timetensor.visu import plot_expe;plot_expe('$output_dir', $eval_freq)"
done

multipliers="1 0"
srun python3 -c "from src.timetensor.visu import get_boxplots;get_boxplots('$big_output_dir', 'test1_mean_results.json', 5, 'Test MSE', save_name='test1_mse_boxplot.pdf')"
srun python3 -c "from src.timetensor.visu import get_boxplots;get_boxplots('$big_output_dir', 'test1_mean_results.json', 5, 'Test NMSE', save_name='test1_nmse_boxplot.pdf')"
srun python3 -c "from src.timetensor.visu import get_multiple_expe_results;get_multiple_expe_results('${big_output_dir}', 'test1_mean_results.json', 5, multipliers='$multipliers', save_name='errors1.pdf')"
srun python3 -c "from src.timetensor.visu import get_multiple_expe_results;get_multiple_expe_results('${big_output_dir}', 'test2_mean_results.json', 5, multipliers='$multipliers', save_name='errors2.pdf')"

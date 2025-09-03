#!/bin/bash
#SBATCH --job-name=sets -p h100
#SBATCH --gres=gpu:1 -p h100
#SBATCH --output=scripts/bench_sets.out -p h100
#SBATCH --error=scripts/bench_sets.err -p h100
#SBATCH --time=1-23:00:00 -p h100
#SBATCH --mem=40G -p h100
source .venv/bin/activate

#19h30
model_name=PatchTST
loss=NMSE
lr=0.00001
epochs=2000
bs=256
eval_freq=100

for data in electricity traffic solar
do
    big_output_dir="../outputs/bench_sets/${data}/"

    #336 24
    lags=336
    horizon=24
    main_output_dir="${big_output_dir}${lags}_${horizon}/"
    rm -rf "$main_output_dir"
    mkdir -p "$main_output_dir"
    for seed in 1 2 3 4 5
    do
        output_dir="${main_output_dir}seed_$seed/"
        srun python3 load_dataset.py "data=$data" "misc.seed=$seed" "misc.verbose=0" "task.lags=$lags" "task.horizon=$horizon"

        #revin
        normalization=revin
        srun python3 train_model.py \
            "misc.seed=$seed" \
            "data=$data" \
            "data.reshuffle=True" \
            "misc.output_dir=$output_dir" \
            "misc.benchmark=True" \
            "model=${model_name}" \
            "normalization=$normalization" \
            "training.loss=$loss" \
            "training.bs=$bs" \
            "training.lr=$lr" \
            "training.epochs=$epochs" \
            "training.eval_freq=$eval_freq" \
            "training.print_freq=100" \
            "task.lags=$lags" \
            "task.horizon=$horizon" \
            "misc.save_name=revin"

        #revin+min+init
        normalization=mIN
        srun python3 train_model.py \
            "misc.seed=$seed" \
            "data=$data" \
            "data.reshuffle=True" \
            "misc.output_dir=$output_dir" \
            "misc.benchmark=True" \
            "model=${model_name}" \
            "normalization=$normalization" \
            "training.loss=$loss" \
            "training.bs=$bs" \
            "training.lr=$lr" \
            "training.epochs=$epochs" \
            "training.eval_freq=$eval_freq" \
            "training.print_freq=100" \
            "task.lags=$lags" \
            "task.horizon=$horizon" \
            "normalization.configs.init_beta=True" \
            "normalization.configs.init_alpha=True" \
            "normalization.configs.use_gamma=True" \
            "normalization.configs.inverse_gamma=True" \
            "misc.save_name=revin_min_init"

        srun python3 -c "from src.timetensor.visu import plot_expe;plot_expe('$output_dir', $eval_freq)"
    done

    multipliers="-7 0"
    srun python3 -c "from src.timetensor.visu import get_boxplots;get_boxplots('$main_output_dir', 'test1_mean_results.json', 5, 'Test MSE', save_name='test1_boxplot.pdf')"
    srun python3 -c "from src.timetensor.visu import get_boxplots;get_boxplots('$main_output_dir', 'test2_mean_results.json', 5, 'Test MSE', save_name='test2_boxplot.pdf')"
    srun python3 -c "from src.timetensor.visu import get_multiple_expe_results;get_multiple_expe_results('${main_output_dir}', 'test1_mean_results.json', 5, multipliers='$multipliers', save_name='errors1.pdf')"
    srun python3 -c "from src.timetensor.visu import get_multiple_expe_results;get_multiple_expe_results('${main_output_dir}', 'test2_mean_results.json', 5, multipliers='$multipliers', save_name='errors2.pdf')"

    #168 24
    lags=168
    horizon=24
    main_output_dir="${big_output_dir}${lags}_${horizon}/"
    rm -rf "$main_output_dir"
    mkdir -p "$main_output_dir"
    for seed in 1 2 3 4 5
    do
        output_dir="${main_output_dir}seed_$seed/"
        srun python3 load_dataset.py "data=$data" "misc.seed=$seed" "misc.verbose=0" "task.lags=$lags" "task.horizon=$horizon"

        #revin
        normalization=revin
        srun python3 train_model.py \
            "misc.seed=$seed" \
            "data=$data" \
            "data.reshuffle=True" \
            "misc.output_dir=$output_dir" \
            "misc.benchmark=True" \
            "model=${model_name}" \
            "normalization=$normalization" \
            "training.loss=$loss" \
            "training.bs=$bs" \
            "training.lr=$lr" \
            "training.epochs=$epochs" \
            "training.eval_freq=$eval_freq" \
            "training.print_freq=100" \
            "task.lags=$lags" \
            "task.horizon=$horizon" \
            "misc.save_name=revin"

        #revin+mIN+init
        normalization=mIN
        srun python3 train_model.py \
            "misc.seed=$seed" \
            "data=$data" \
            "data.reshuffle=True" \
            "misc.output_dir=$output_dir" \
            "misc.benchmark=True" \
            "model=${model_name}" \
            "normalization=$normalization" \
            "training.loss=$loss" \
            "training.bs=$bs" \
            "training.lr=$lr" \
            "training.epochs=$epochs" \
            "training.eval_freq=$eval_freq" \
            "training.print_freq=100" \
            "task.lags=$lags" \
            "task.horizon=$horizon" \
            "normalization.configs.init_beta=True" \
            "normalization.configs.init_alpha=True" \
            "normalization.configs.use_gamma=True" \
            "normalization.configs.inverse_gamma=True" \
            "misc.save_name=revin_min_init"

        srun python3 -c "from src.timetensor.visu import plot_expe;plot_expe('$output_dir', $eval_freq)"
    done

    multipliers="-7 0"
    srun python3 -c "from src.timetensor.visu import get_boxplots;get_boxplots('$main_output_dir', 'test1_mean_results.json', 5, 'Test MSE', save_name='test1_boxplot.pdf')"
    srun python3 -c "from src.timetensor.visu import get_boxplots;get_boxplots('$main_output_dir', 'test2_mean_results.json', 5, 'Test MSE', save_name='test2_boxplot.pdf')"
    srun python3 -c "from src.timetensor.visu import get_multiple_expe_results;get_multiple_expe_results('${main_output_dir}', 'test1_mean_results.json', 5, multipliers='$multipliers', save_name='errors1.pdf')"
    srun python3 -c "from src.timetensor.visu import get_multiple_expe_results;get_multiple_expe_results('${main_output_dir}', 'test2_mean_results.json', 5, multipliers='$multipliers', save_name='errors2.pdf')"
 
    #336 168
    lags=336
    horizon=168
    bs=128
    main_output_dir="${big_output_dir}${lags}_${horizon}/"
    rm -rf "$main_output_dir"
    mkdir -p "$main_output_dir"
    for seed in 1 2 3 4 5
    do
        output_dir="${main_output_dir}seed_$seed/"
        srun python3 load_dataset.py "data=$data" "misc.seed=$seed" "misc.verbose=0" "task.lags=$lags" "task.horizon=$horizon"

        #revin
        normalization=revin
        srun python3 train_model.py \
            "misc.seed=$seed" \
            "data=$data" \
            "data.reshuffle=True" \
            "misc.output_dir=$output_dir" \
            "misc.benchmark=True" \
            "model=${model_name}" \
            "normalization=$normalization" \
            "training.loss=$loss" \
            "training.bs=$bs" \
            "training.lr=$lr" \
            "training.epochs=$epochs" \
            "training.eval_freq=$eval_freq" \
            "training.print_freq=100" \
            "task.lags=$lags" \
            "task.horizon=$horizon" \
            "misc.save_name=revin"

        #revin+mIN+init
        normalization=mIN
        srun python3 train_model.py \
            "misc.seed=$seed" \
            "data=$data" \
            "data.reshuffle=True" \
            "misc.output_dir=$output_dir" \
            "misc.benchmark=True" \
            "model=${model_name}" \
            "normalization=$normalization" \
            "training.loss=$loss" \
            "training.bs=$bs" \
            "training.lr=$lr" \
            "training.epochs=$epochs" \
            "training.eval_freq=$eval_freq" \
            "training.print_freq=100" \
            "task.lags=$lags" \
            "task.horizon=$horizon" \
            "normalization.configs.init_beta=True" \
            "normalization.configs.init_alpha=True" \
            "normalization.configs.use_gamma=True" \
            "normalization.configs.inverse_gamma=True" \
            "misc.save_name=revin_min_init"

        srun python3 -c "from src.timetensor.visu import plot_expe;plot_expe('$output_dir', $eval_freq)"
    done

    multipliers="-7 0"
    srun python3 -c "from src.timetensor.visu import get_boxplots;get_boxplots('$main_output_dir', 'test1_mean_results.json', 5, 'Test MSE', save_name='test1_boxplot.pdf')"
    srun python3 -c "from src.timetensor.visu import get_boxplots;get_boxplots('$main_output_dir', 'test2_mean_results.json', 5, 'Test MSE', save_name='test2_boxplot.pdf')"
    srun python3 -c "from src.timetensor.visu import get_multiple_expe_results;get_multiple_expe_results('${main_output_dir}', 'test1_mean_results.json', 5, multipliers='$multipliers', save_name='errors1.pdf')"
    srun python3 -c "from src.timetensor.visu import get_multiple_expe_results;get_multiple_expe_results('${main_output_dir}', 'test2_mean_results.json', 5, multipliers='$multipliers', save_name='errors2.pdf')"

done
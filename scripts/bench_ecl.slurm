#!/bin/bash
#SBATCH --job-name=ecl -p h100
#SBATCH --gres=gpu:1 -p h100
#SBATCH --output=scripts/bench_ecl.out -p h100
#SBATCH --error=scripts/bench_ecl.err -p h100
#SBATCH --time=23:00:00 -p h100
#SBATCH --mem=40G -p h100
source .venv/bin/activate

main_output_dir="../outputs/bench_ecl/"

rm -rf "$main_output_dir"
mkdir -p "$main_output_dir"

data=electricity

model_name=PatchTST
loss=NMSE
lr=0.00001
epochs=1000
bs=230
eval_freq=100

srun python3 load_dataset.py "task.rebuild_pt=True" "task.new_example=True" "task.replot=True" "task.recluster=True" "data=$data" "misc.verbose=0"

for seed in 1 2 3 4 5
do
    output_dir="${main_output_dir}seed_$seed/"
    
    srun python3 load_dataset.py "data=$data" "misc.seed=$seed" "misc.verbose=0"

    for normalization in instance revin
    do
        srun python3 train_model.py \
            "misc.seed=$seed" \
            "data=$data" \
            "data.reshuffle=True" \
            "misc.output_dir=$output_dir" \
            "misc.benchmark=True" \
            "model=$model_name" \
            "normalization=$normalization" \
            "training.loss=$loss" \
            "training.bs=$bs" \
            "training.lr=$lr" \
            "training.epochs=$epochs" \
            "training.eval_freq=$eval_freq" \
            "training.print_freq=100" \
            "misc.save_name=$normalization"
    done

    normalization=mIN    
    #revin
    init_alpha=False
    init_beta=False
    use_gamma=True
    inverse_gamma=True
    fixed_beta=False
    fixed_alpha=False
    srun python3 train_model.py \
        "misc.seed=$seed" \
        "data=$data" \
        "data.reshuffle=True" \
        "misc.output_dir=$output_dir" \
        "misc.benchmark=True" \
        "model=$model_name" \
        "normalization=$normalization" \
        "training.loss=$loss" \
        "training.bs=$bs" \
        "training.lr=$lr" \
        "training.epochs=$epochs" \
        "training.eval_freq=$eval_freq" \
        "training.print_freq=100" \
        "normalization.configs.init_beta=$init_beta" \
        "normalization.configs.init_alpha=$init_alpha" \
        "normalization.configs.fixed_beta=$fixed_beta" \
        "normalization.configs.fixed_alpha=$fixed_alpha " \
        "normalization.configs.use_gamma=$use_gamma" \
        "normalization.configs.inverse_gamma=$inverse_gamma" \
        "misc.save_name=fake_revin"

    #min
    normalization=mIN
    
    #default
    init_alpha=False
    init_beta=False
    use_gamma=False
    inverse_gamma=False
    fixed_beta=False
    fixed_alpha=False
    srun python3 train_model.py \
        "misc.seed=$seed" \
        "data=$data" \
        "data.reshuffle=True" \
        "misc.output_dir=$output_dir" \
        "misc.benchmark=True" \
        "model=$model_name" \
        "normalization=$normalization" \
        "training.loss=$loss" \
        "training.bs=$bs" \
        "training.lr=$lr" \
        "training.epochs=$epochs" \
        "training.eval_freq=$eval_freq" \
        "training.print_freq=100" \
        "normalization.configs.init_beta=$init_beta" \
        "normalization.configs.init_alpha=$init_alpha" \
        "normalization.configs.fixed_beta=$fixed_beta" \
        "normalization.configs.fixed_alpha=$fixed_alpha " \
        "normalization.configs.use_gamma=$use_gamma" \
        "normalization.configs.inverse_gamma=$inverse_gamma" \
        "misc.save_name=min"

    #default+gamma
    init_alpha=False
    init_beta=False
    use_gamma=True
    inverse_gamma=False
    fixed_beta=False
    fixed_alpha=False
    srun python3 train_model.py \
        "misc.seed=$seed" \
        "data=$data" \
        "data.reshuffle=True" \
        "misc.output_dir=$output_dir" \
        "misc.benchmark=True" \
        "model=$model_name" \
        "normalization=$normalization" \
        "training.loss=$loss" \
        "training.bs=$bs" \
        "training.lr=$lr" \
        "training.epochs=$epochs" \
        "training.eval_freq=$eval_freq" \
        "training.print_freq=100" \
        "normalization.configs.init_beta=$init_beta" \
        "normalization.configs.init_alpha=$init_alpha" \
        "normalization.configs.fixed_beta=$fixed_beta" \
        "normalization.configs.fixed_alpha=$fixed_alpha " \
        "normalization.configs.use_gamma=$use_gamma" \
        "normalization.configs.inverse_gamma=$inverse_gamma" \
        "misc.save_name=min_gamma"

    #default+init
    init_alpha=True
    init_beta=True
    use_gamma=False
    inverse_gamma=False
    fixed_beta=False
    fixed_alpha=False
    srun python3 train_model.py \
        "misc.seed=$seed" \
        "data=$data" \
        "data.reshuffle=True" \
        "misc.output_dir=$output_dir" \
        "misc.benchmark=True" \
        "model=$model_name" \
        "normalization=$normalization" \
        "training.loss=$loss" \
        "training.bs=$bs" \
        "training.lr=$lr" \
        "training.epochs=$epochs" \
        "training.eval_freq=$eval_freq" \
        "training.print_freq=100" \
        "normalization.configs.init_beta=$init_beta" \
        "normalization.configs.init_alpha=$init_alpha" \
        "normalization.configs.fixed_beta=$fixed_beta" \
        "normalization.configs.fixed_alpha=$fixed_alpha " \
        "normalization.configs.use_gamma=$use_gamma" \
        "normalization.configs.inverse_gamma=$inverse_gamma" \
        "misc.save_name=min_init"
    
    #default+init+gamma
    init_alpha=True
    init_beta=True
    use_gamma=True
    inverse_gamma=False
    fixed_beta=False
    fixed_alpha=False
    srun python3 train_model.py \
        "misc.seed=$seed" \
        "data=$data" \
        "data.reshuffle=True" \
        "misc.output_dir=$output_dir" \
        "misc.benchmark=True" \
        "model=$model_name" \
        "normalization=$normalization" \
        "training.loss=$loss" \
        "training.bs=$bs" \
        "training.lr=$lr" \
        "training.epochs=$epochs" \
        "training.eval_freq=$eval_freq" \
        "training.print_freq=100" \
        "normalization.configs.init_beta=$init_beta" \
        "normalization.configs.init_alpha=$init_alpha" \
        "normalization.configs.fixed_beta=$fixed_beta" \
        "normalization.configs.fixed_alpha=$fixed_alpha " \
        "normalization.configs.use_gamma=$use_gamma" \
        "normalization.configs.inverse_gamma=$inverse_gamma" \
        "misc.save_name=min_init_gamma"

    #min + revin + init
    init_alpha=True
    init_beta=True
    use_gamma=True
    inverse_gamma=True
    fixed_beta=False
    fixed_alpha=False
    srun python3 train_model.py \
        "misc.seed=$seed" \
        "data=$data" \
        "data.reshuffle=True" \
        "misc.output_dir=$output_dir" \
        "misc.benchmark=True" \
        "model=$model_name" \
        "normalization=$normalization" \
        "training.loss=$loss" \
        "training.bs=$bs" \
        "training.lr=$lr" \
        "training.epochs=$epochs" \
        "training.eval_freq=$eval_freq" \
        "training.print_freq=100" \
        "normalization.configs.init_beta=$init_beta" \
        "normalization.configs.init_alpha=$init_alpha" \
        "normalization.configs.fixed_beta=$fixed_beta" \
        "normalization.configs.fixed_alpha=$fixed_alpha " \
        "normalization.configs.use_gamma=$use_gamma" \
        "normalization.configs.inverse_gamma=$inverse_gamma" \
        "misc.save_name=revin_min_init"

    srun python3 -c "from src.timetensor.visu import plot_expe;plot_expe('$output_dir', $eval_freq)"

done

multipliers="3 0"
srun python3 -c "from src.timetensor.visu import get_boxplots;get_boxplots('$main_output_dir', 'test1_mean_results.json', 5, 'Test MSE', save_name='test1_boxplot.pdf')"
srun python3 -c "from src.timetensor.visu import get_boxplots;get_boxplots('$main_output_dir', 'test2_mean_results.json', 5, 'Test MSE', save_name='test2_boxplot.pdf')"
srun python3 -c "from src.timetensor.visu import get_multiple_expe_results;get_multiple_expe_results('${main_output_dir}', 'test1_mean_results.json', 5, multipliers='$multipliers', save_name='errors1.pdf')"
srun python3 -c "from src.timetensor.visu import get_multiple_expe_results;get_multiple_expe_results('${main_output_dir}', 'test2_mean_results.json', 5, multipliers='$multipliers', save_name='errors2.pdf')"

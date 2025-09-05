#!/bin/bash
#SBATCH --job-name=tune -p h100
#SBATCH --gres=gpu:1 -p h100
#SBATCH --output=scripts/bench_tune.out -p h100
#SBATCH --error=scripts/bench_tune.err -p h100
#SBATCH --time=23:00:00 -p h100
#SBATCH --mem=40G -p h100
source .venv/bin/activate

#min tune
data=synthetic

model_name=PatchTST
loss=NMSE
lr=0.00001
bs=100
eval_freq=100
epochs=1000

#size
srun python3 load_dataset.py "task.rebuild_pt=True" "data=$data" "seed=1" "misc.verbose=0" "task.lags=100" "task.horizon=20"

for subset in 1 0.2 0.5
do
    main_output_dir="../outputs/bench_size/s${subset}/"
    sizes="${subset};1;1;1;1;1"

    rm -rf "$main_output_dir"
    mkdir -p "$main_output_dir"

    for seed in 1 2 3 4 5
    do
        output_dir="${main_output_dir}seed_$seed/"
        srun python3 load_dataset.py "data=$data" "data.subsets.sizes='$sizes'" "misc.seed=$seed" "misc.verbose=0" "task.lags=100" "task.horizon=20"
        
        #mIN
        for node in node0 node1
        do
            normalization=mIN
            srun python3 train_model.py \
                "misc.seed=$seed" \
                "data=$data" \
                "data.subsets.cluster='$node'" \
                "data.reshuffle=True" \
                "data.subsets.sizes='$sizes'" \
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
                "task.lags=100" \
                "task.horizon=20" \
                "normalization.configs.init_beta=True" \
                "normalization.configs.init_alpha=True" \
                "normalization.configs.fixed_beta=False" \
                "normalization.configs.fixed_alpha=False " \
                "normalization.configs.use_gamma=True" \
                "normalization.configs.inverse_gamma=True" \
                "misc.save_name=min_$node"
        done

    done
    multipliers="4 2"
    srun python3 -c "from src.timetensor.visu import get_boxplots;get_boxplots('$main_output_dir', 'test1_mean_results.json', 5, 'Test MSE', save_name='test1_boxplot.pdf')"
    srun python3 -c "from src.timetensor.visu import get_boxplots;get_boxplots('$main_output_dir', 'test2_mean_results.json', 5, 'Test MSE', save_name='test2_boxplot.pdf')"
    srun python3 -c "from src.timetensor.visu import get_multiple_expe_results;get_multiple_expe_results('${main_output_dir}', 'test1_mean_results.json', 5, multipliers='$multipliers', save_name='errors1.pdf')"
    srun python3 -c "from src.timetensor.visu import get_multiple_expe_results;get_multiple_expe_results('${main_output_dir}', 'test2_mean_results.json', 5, multipliers='$multipliers', save_name='errors2.pdf')"

done


#mix
srun python3 load_dataset.py "task.rebuild_pt=True" "data=$data" "seed=1" "misc.verbose=0" "task.lags=100" "task.horizon=20"

for r in 0.1 0
do

    main_output_dir="../outputs/bench_mix/r${r}/"

    rm -rf "$main_output_dir"
    mkdir -p "$main_output_dir"

    for seed in 1 2 3 4 5
    do
        output_dir="${main_output_dir}seed_$seed/"
        srun python3 load_dataset.py "task.rebuild_pt=True" "data=$data" "data.r1=$r" "data.r2=$r" "misc.seed=$seed" "misc.verbose=0" "task.lags=100" "task.horizon=20"
        
        #mIN
        normalization=mIN
        srun python3 train_model.py \
            "misc.seed=$seed" \
            "data=$data" \
            "data.reshuffle=True" \
            "data.r1=$r" \
            "data.r2=$r" \
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
            "task.lags=100" \
            "task.horizon=20" \
            "normalization.configs.init_beta=True" \
            "normalization.configs.init_alpha=True" \
            "normalization.configs.fixed_beta=False" \
            "normalization.configs.fixed_alpha=False " \
            "normalization.configs.use_gamma=True" \
            "normalization.configs.inverse_gamma=True" \
            "misc.save_name=min"
        
        init_path="${output_dir}min/trained_model.pt"
        for node in node0 node1
        do
            normalization=mIN
            srun python3 train_model.py \
                "misc.seed=$seed" \
                "data=$data" \
                "data.subsets.cluster='$node'" \
                "data.reshuffle=True" \
                "data.r1=$r" \
                "data.r2=$r" \
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
                "training.retrain=False" \
                "training.init='$init_path'" \
                "task.lags=100" \
                "task.horizon=20" \
                "normalization.configs.init_beta=True" \
                "normalization.configs.init_alpha=True" \
                "normalization.configs.fixed_beta=False" \
                "normalization.configs.fixed_alpha=False " \
                "normalization.configs.use_gamma=True" \
                "normalization.configs.inverse_gamma=True" \
                "misc.save_name=min_$node"
        done
    multipliers="4 2"
    srun python3 -c "from src.timetensor.visu import get_boxplots;get_boxplots('$main_output_dir', 'test1_mean_results.json', 5, 'Test MSE', save_name='test1_boxplot.pdf')"
    srun python3 -c "from src.timetensor.visu import get_boxplots;get_boxplots('$main_output_dir', 'test2_mean_results.json', 5, 'Test MSE', save_name='test2_boxplot.pdf')"
    srun python3 -c "from src.timetensor.visu import get_multiple_expe_results;get_multiple_expe_results('${main_output_dir}', 'test1_mean_results.json', 5, multipliers='$multipliers', save_name='errors1.pdf')"
    srun python3 -c "from src.timetensor.visu import get_multiple_expe_results;get_multiple_expe_results('${main_output_dir}', 'test2_mean_results.json', 5, multipliers='$multipliers', save_name='errors2.pdf')"

done




#tune

previous_output_dir="../outputs/bench_mix/r${r}/seed"
main_output_dir="../outputs/bench_tune/r${r}/"

rm -rf "$main_output_dir"
mkdir -p "$main_output_dir"

srun python3 load_dataset.py "task.rebuild_pt=True" "data=$data" "seed=1" "misc.verbose=0" "task.lags=100" "task.horizon=20"

for seed in 1 2 3 4 5
do
    output_dir="${main_output_dir}seed_$seed/"
    srun python3 load_dataset.py "task.rebuild_pt=True" "data=$data" "misc.seed=$seed" "misc.verbose=0" "task.lags=100" "task.horizon=20"
    
    #global
    #mIN
    epochs=1000
    normalization=mIN
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
        "task.lags=100" \
        "task.horizon=20" \
        "normalization.configs.init_beta=True" \
        "normalization.configs.init_alpha=True" \
        "normalization.configs.fixed_beta=False" \
        "normalization.configs.fixed_alpha=False " \
        "normalization.configs.use_gamma=True" \
        "normalization.configs.inverse_gamma=True" \
        "misc.save_name=global_min" \

    init_path="${output_dir}global_min/trained_model.pt"
    epochs=100
    for node in node0 node1
    do
        normalization=mIN
        srun python3 train_model.py \
            "misc.seed=$seed" \
            "data=$data" \
            "data.subsets.cluster='$node'" \
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
            "training.retrain=False" \
            "training.init='$init_path'" \
            "task.lags=100" \
            "task.horizon=20" \
            "normalization.configs.init_beta=True" \
            "normalization.configs.init_alpha=True" \
            "normalization.configs.fixed_beta=False" \
            "normalization.configs.fixed_alpha=False " \
            "normalization.configs.use_gamma=True" \
            "normalization.configs.inverse_gamma=True" \
            "misc.save_name=min_$node"
    done

    #global
    #mIN
    epochs=1000
    normalization=mIN
    for node in node0 node1
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
            "task.lags=100" \
            "task.horizon=20" \
            "normalization.configs.init_beta=True" \
            "normalization.configs.init_alpha=True" \
            "normalization.configs.fixed_beta=False" \
            "normalization.configs.fixed_alpha=False " \
            "normalization.configs.use_gamma=True" \
            "normalization.configs.inverse_gamma=True" \
            "misc.save_name=global_$node"

    done

    init_path="${output_dir}global_0/trained_model.pt"
    node="node1"
    epochs=100
    normalization=mIN
    srun python3 train_model.py \
        "misc.seed=$seed" \
        "data=$data" \
        "data.subsets.cluster='$node'" \
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
        "training.retrain=False" \
        "training.init='$init_path'" \
        "task.lags=100" \
        "task.horizon=20" \
        "normalization.configs.init_beta=True" \
        "normalization.configs.init_alpha=True" \
        "normalization.configs.fixed_beta=False" \
        "normalization.configs.fixed_alpha=False " \
        "normalization.configs.use_gamma=True" \
        "normalization.configs.inverse_gamma=True" \
        "misc.save_name=tune_$node"

    init_path="${output_dir}global_1/trained_model.pt"
    node="node0"
    epochs=100
    normalization=mIN
    srun python3 train_model.py \
        "misc.seed=$seed" \
        "data=$data" \
        "data.subsets.cluster='$node'" \
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
        "training.retrain=False" \
        "training.init='$init_path'" \
        "task.lags=100" \
        "task.horizon=20" \
        "normalization.configs.init_beta=True" \
        "normalization.configs.init_alpha=True" \
        "normalization.configs.fixed_beta=False" \
        "normalization.configs.fixed_alpha=False " \
        "normalization.configs.use_gamma=True" \
        "normalization.configs.inverse_gamma=True" \
        "misc.save_name=tune_$node"
done

multipliers="4 2"
srun python3 -c "from src.timetensor.visu import get_boxplots;get_boxplots('$main_output_dir', 'test1_mean_results.json', 5, 'Test MSE', save_name='test1_boxplot.pdf')"
srun python3 -c "from src.timetensor.visu import get_boxplots;get_boxplots('$main_output_dir', 'test2_mean_results.json', 5, 'Test MSE', save_name='test2_boxplot.pdf')"
srun python3 -c "from src.timetensor.visu import get_multiple_expe_results;get_multiple_expe_results('${main_output_dir}', 'test1_mean_results.json', 5, multipliers='$multipliers', save_name='errors1.pdf')"
srun python3 -c "from src.timetensor.visu import get_multiple_expe_results;get_multiple_expe_results('${main_output_dir}', 'test2_mean_results.json', 5, multipliers='$multipliers', save_name='errors2.pdf')"
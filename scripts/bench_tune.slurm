#!/bin/bash
#SBATCH --job-name=tune -p h100
#SBATCH --gres=gpu:1 -p h100
#SBATCH --output=scripts/bench_tune.out -p h100
#SBATCH --error=scripts/bench_tune.err -p h100
#SBATCH --time=23:00:00 -p h100
#SBATCH --mem=40G -p h100
source .venv/bin/activate

data=synthetic

model_name=PatchTST
loss=NMSE
lr=0.00001
bs=100
eval_freq=100

for r in 0 0.1 0.25
do
    for subset in 1 0.5 0.2
    do
        main_output_dir="../outputs/bench_tune/r${r}_s${subset}/"
        sizes="${subset};1;1;1;1;1"

        rm -rf "$main_output_dir"
        mkdir -p "$main_output_dir"

        for seed in 1 2 3 4 5
        do
            output_dir="${main_output_dir}seed_$seed/"
            srun python3 load_dataset.py "data=$data" "data.r1=$r" "data.r2=$r" "data.subsets.sizes='$sizes'" "misc.seed=$seed" "misc.verbose=0" "task.lags=100" "task.horizon=20"
            
            #mIN
            epochs=500
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
                    "normalization.configs.fixed_beta=True" \
                    "normalization.configs.fixed_alpha=True " \
                    "normalization.configs.use_gamma=True" \
                    "normalization.configs.inverse_gamma=True" \
                    "misc.save_name=min_$node"
            done

            epochs=100
            node=node0
            init_path=${output_dir}min_node1/trained_model.pt
            normalization=mIN
            srun python3 train_model.py \
                "misc.seed=$seed" \
                "data=$data" \
                "data.subsets.cluster='$node'" \
                "data.reshuffle=True" \
                "data.r1=$r" \
                "data.r2=$r" \
                "data.subsets.sizes='$sizes'" \
                "misc.output_dir=$output_dir" \
                "misc.benchmark=True" \
                "model=$model_name" \
                "normalization=$normalization" \
                "training.init=$init_path" \
                "training.freeze_core=$freeze_core" \
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
                "normalization.configs.fixed_beta=True" \
                "normalization.configs.fixed_alpha=True " \
                "normalization.configs.use_gamma=True" \
                "normalization.configs.inverse_gamma=True" \
                "misc.save_name=min_node0_tune1"

            epochs=100
            node=node1
            init_path=${output_dir}min_node0/trained_model.pt
            normalization=mIN
            srun python3 train_model.py \
                "misc.seed=$seed" \
                "data=$data" \
                "data.subsets.cluster='$node'" \
                "data.reshuffle=True" \
                "data.r1=$r" \
                "data.r2=$r" \
                "data.subsets.sizes='$sizes'" \
                "misc.output_dir=$output_dir" \
                "misc.benchmark=True" \
                "model=$model_name" \
                "normalization=$normalization" \
                "training.init=$init_path" \
                "training.freeze_core=$freeze_core" \
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
                "normalization.configs.fixed_beta=True" \
                "normalization.configs.fixed_alpha=True " \
                "normalization.configs.use_gamma=True" \
                "normalization.configs.inverse_gamma=True" \
                "misc.save_name=min_node1_tune0"

        done
        multipliers="4 2"
        srun python3 -c "from src.timetensor.visu import get_boxplots;get_boxplots('$main_output_dir', 'test1_mean_results.json', 5, 'Test MSE', save_name='test1_boxplot.pdf')"
        srun python3 -c "from src.timetensor.visu import get_boxplots;get_boxplots('$main_output_dir', 'test2_mean_results.json', 5, 'Test MSE', save_name='test2_boxplot.pdf')"
        srun python3 -c "from src.timetensor.visu import get_multiple_expe_results;get_multiple_expe_results('${main_output_dir}', 'test1_mean_results.json', 5, multipliers='$multipliers', save_name='errors1.pdf')"
        srun python3 -c "from src.timetensor.visu import get_multiple_expe_results;get_multiple_expe_results('${main_output_dir}', 'test2_mean_results.json', 5, multipliers='$multipliers', save_name='errors2.pdf')"

        names="min_node0;min_node1_tune0"
        srun python3 -c "from src.timetensor.visu import get_multiple_expe_results;get_multiple_expe_results('${main_output_dir}', 'test1_mean_results.json', 5, names='$names', save_name='errors_node0.pdf', print_table=False)"
        names="min_node1;min_node0_tune1"
        srun python3 -c "from src.timetensor.visu import get_multiple_expe_results;get_multiple_expe_results('${main_output_dir}', 'test1_mean_results.json', 5, names='$names', save_name='erorrs_node1.pdf', print_table=False)"

    done
done
#!/bin/bash
#SBATCH --job-name=cte -p h100
#SBATCH --gres=gpu:1 -p h100
#SBATCH --output=scripts/bench_cte.out -p h100
#SBATCH --error=scripts/bench_cte.err -p h100
#SBATCH --time=23:00:00 -p h100
#SBATCH --mem=40G -p h100
source .venv/bin/activate

output_dir="../outputs/bench_cte/"

rm -rf "$output_dir"
mkdir -p "$output_dir"

data=electricity
data_path="../datasets/electricity/"

for remove_cte in False True
do
    srun python3 -c "from src.timetensor.electricity import fetch_data;from src.timetensor.dataset import build_dataset;fetcher = lambda path: fetch_data(path, raw_format='csv', drop=$remove_cte);build_dataset(fetcher, '$data_path')"

    for seed in 2
    do
        model_name=repeat
        srun python3 train_model.py \
            "model=${model_name}" \
            "normalization=None" \
            "misc.output_dir=$output_dir" \
            "misc.benchmark=True" \
            "misc.complete_evaluation=True" \
            "data=$data" \
            "data.reshuffle=True" \
            "misc.seed=$seed" \
            "misc.save_name=${model_name}_${seed}_remove${remove_cte}" \
            "data.remove_cte=$remove_cte"

        model_name=PatchTST
        normalization=instance
        lr=0.00001
        bs=223
        epochs=300
        eval_freq=100

        for loss in MSE NMSE
        do
            srun python3 train_model.py \
                "model=$model_name" \
                "misc.seed=$seed" \
                "misc.output_dir=$output_dir" \
                "normalization=$normalization" \
                "training.loss=$loss" \
                "misc.benchmark=True" \
                "misc.complete_evaluation=True" \
                "training.lr=$lr" \
                "training.bs=$bs" \
                "training.epochs=$epochs" \
                "training.eval_freq=$eval_freq" \
                "training.print_freq=100" \
                "data=$data" \
                "data.reshuffle=True" \
                "misc.save_name=${model_name}_${loss}_${seed}_remove${remove_cte}" \
                "data.remove_cte=$remove_cte"
        done
    done
done

multipliers="-8 -3 0 0"
srun python3 -c "from src.timetensor.visu import get_expe_results;get_expe_results('${output_dir}', 'test1_mean_results.json', multipliers='$multipliers', save_name='errors1.pdf')"
srun python3 -c "from src.timetensor.visu import get_expe_results;get_expe_results('${output_dir}', 'test2_mean_results.json', multipliers='$multipliers', save_name='errors2.pdf')"

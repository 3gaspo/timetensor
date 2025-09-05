#!/bin/bash
#SBATCH --job-name=fed_big -p h100
#SBATCH --gres=gpu:1 -p h100
#SBATCH --output=scripts/fed_big.out -p h100
#SBATCH --error=scripts/fed_big.err -p h100
#SBATCH --time=23:00:00 -p h100
#SBATCH --mem=40G -p h100
source .venv/bin/activate

main_output_dir="../outputs/fed_big/"

rm -rf "$main_output_dir"
mkdir -p "$main_output_dir"

data=electricity
model_name=PatchTST
loss=NMSE
lr=0.00001
bs=1
eval_freq=100
seed=1
output_dir="${main_output_dir}seed_$seed/"

# srun python3 load_dataset.py "data=$data" "misc.seed=$seed" "misc.verbose=0"

rounds=10
sampled_clients=0.1
epochs=10

for fedmin in True False
do
    
    normalization=mIN
    srun python3 train_fedavg.py \
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
        "task.rounds=$rounds" \
        "task.sampled_clients=$sampled_clients" \
        "training.eval_freq=$eval_freq" \
        "training.print_freq=100" \
        "task.splits='None'" \
        "task.fedmin=$fedmin" \
        "misc.save_name=${normalization}_joint${fedmin}"

done

multipliers="0 0 0 0 0 0 0"
srun python3 -c "from src.timetensor.visu import get_expe_results;get_expe_results('${output_dir}', 'MSE_mean_results.json', multipliers='$multipliers', save_name='errors1.pdf')"
srun python3 -c "from src.timetensor.visu import get_expe_results;get_expe_results('${output_dir}', 'NMSE_mean_results.json', multipliers='$multipliers', save_name='errors2.pdf')"

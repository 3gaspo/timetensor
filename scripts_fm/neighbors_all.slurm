#!/bin/bash
#SBATCH --job-name=all
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/all.out
#SBATCH --error=script_outputs/all.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100
#SBATCH --mem=4000

source .venv/bin/activate

seed=1
experiment_dir="../outputs/fm/all_neighbors/"
model=chronos
date_splits='0.6;0.4'

max_windows=10_000

lags=1344
horizon=336
eval_stride=512
train_stride=24
for data in electricity traffic solar #weather exchange_rate ETTh1
do
    # for online in True False
    # do
    #     for oracle in True False
    #     do
    #         for bs in 2 4
    #         do
    #             for distance in "raw_euclidean" #"chronos_euclidean" # "raw_cosine"
    #             do
    #                 save_name="${model}_${distance}_${bs}"
    #                 if [ "$online" = "True" ]; then
    #                     save_name="${save_name}_online"
    #                 fi
    #                 if [ "$oracle" = "True" ]; then
    #                     save_name="${save_name}_oracle"
    #                 fi

    #                 srun python3 neighbors_all.py \
    #                 "extra.distance='$distance'" \
    #                 "extra.max_windows=$max_windows" \
    #                 "extra.online=$online" \
    #                 "extra.oracle=$oracle" \
    #                 "model='$model'" \
    #                 "model.configs.context_mode=future" \
    #                 "task.lags=$lags" \
    #                 "task.horizon=$horizon" \
    #                 "normalization=instance" \
    #                 "data=$data" \
    #                 "data.sampling.train_stride=$train_stride" \
    #                 "data.sampling.eval_stride=$eval_stride" \
    #                 "training.bs=$bs" \
    #                 "training.eval_runs=1" \
    #                 "misc.seed=$seed" \
    #                 "misc.device='gpu'" \
    #                 "misc.save_name='$save_name'" \
    #                 "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
    #             done
    #         done
    #     done
    # done

    #bs1
    srun python3 neighbors_all.py \
    "model='$model'" \
    "task.lags=$lags" \
    "task.horizon=$horizon" \
    "normalization=instance" \
    "data=$data" \
    "data.sampling.train_stride=$train_stride" \
    "data.sampling.eval_stride=$eval_stride" \
    "training.bs=1" \
    "training.eval_runs=1" \
    "misc.seed=$seed" \
    "misc.device='cpu'" \
    "misc.save_name='${model}_baseline'" \
    "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
done


lags=672
horizon=168
eval_stride=512
train_stride=24
for data in electricity traffic solar #weather exchange_rate ETTh1
do
    # for online in False True
    # do
    #     for oracle in False True
    #     do
    #         for bs in 2 4
    #         do
    #             for distance in "raw_euclidean" #"chronos_euclidean" # "raw_cosine"
    #             do
    #                 save_name="${model}_${distance}_${bs}"
    #                 if [ "$online" = "True" ]; then
    #                     save_name="${save_name}_online"
    #                 fi
    #                 if [ "$oracle" = "True" ]; then
    #                     save_name="${save_name}_oracle"
    #                 fi

    #                 srun python3 neighbors_all.py \
    #                 "extra.distance='$distance'" \
    #                 "extra.max_windows=$max_windows" \
    #                 "extra.online=$online" \
    #                 "extra.oracle=$oracle" \
    #                 "model='$model'" \
    #                 "model.configs.context_mode=future" \
    #                 "task.lags=$lags" \
    #                 "task.horizon=$horizon" \
    #                 "normalization=instance" \
    #                 "data=$data" \
    #                 "data.sampling.train_stride=$train_stride" \
    #                 "data.sampling.eval_stride=$eval_stride" \
    #                 "training.bs=$bs" \
    #                 "training.eval_runs=1" \
    #                 "misc.seed=$seed" \
    #                 "misc.device='gpu'" \
    #                 "misc.save_name='$save_name'" \
    #                 "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
    #             done
    #         done
    #     done
    # done

    #bs1
    srun python3 neighbors_all.py \
    "model='$model'" \
    "task.lags=$lags" \
    "task.horizon=$horizon" \
    "normalization=instance" \
    "data=$data" \
    "data.sampling.train_stride=$train_stride" \
    "data.sampling.eval_stride=$eval_stride" \
    "training.bs=1" \
    "training.eval_runs=1" \
    "misc.seed=$seed" \
    "misc.device='cpu'" \
    "misc.save_name='${model}_baseline'" \
    "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
done

lags=168
horizon=168
eval_stride=512
train_stride=24
for data in electricity traffic solar #weather exchange_rate ETTh1
do
    # for online in False True
    # do
    #     for oracle in False True
    #     do
    #         for bs in 2 4
    #         do
    #             for distance in "raw_euclidean" #"chronos_euclidean" # "raw_cosine"
    #             do
    #                 save_name="${model}_${distance}_${bs}"
    #                 if [ "$online" = "True" ]; then
    #                     save_name="${save_name}_online"
    #                 fi
    #                 if [ "$oracle" = "True" ]; then
    #                     save_name="${save_name}_oracle"
    #                 fi

    #                 srun python3 neighbors_all.py \
    #                 "extra.distance='$distance'" \
    #                 "extra.max_windows=$max_windows" \
    #                 "extra.online=$online" \
    #                 "extra.oracle=$oracle" \
    #                 "model='$model'" \
    #                 "model.configs.context_mode=future" \
    #                 "task.lags=$lags" \
    #                 "task.horizon=$horizon" \
    #                 "normalization=instance" \
    #                 "data=$data" \
    #                 "data.sampling.train_stride=$train_stride" \
    #                 "data.sampling.eval_stride=$eval_stride" \
    #                 "training.bs=$bs" \
    #                 "training.eval_runs=1" \
    #                 "misc.seed=$seed" \
    #                 "misc.device='gpu'" \
    #                 "misc.save_name='$save_name'" \
    #                 "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
    #             done
    #         done
    #     done
    # done

    #bs1
    srun python3 neighbors_all.py \
    "model='$model'" \
    "task.lags=$lags" \
    "task.horizon=$horizon" \
    "normalization=instance" \
    "data=$data" \
    "data.sampling.train_stride=$train_stride" \
    "data.sampling.eval_stride=$eval_stride" \
    "training.bs=1" \
    "training.eval_runs=1" \
    "misc.seed=$seed" \
    "misc.device='cpu'" \
    "misc.save_name='${model}_baseline'" \
    "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
done

lags=168
horizon=24
eval_stride=512
train_stride=24
for data in electricity traffic solar #weather exchange_rate ETTh1
do
    # for online in False True
    # do
    #     for oracle in False True
    #     do
    #         for bs in 2 4
    #         do
    #             for distance in "raw_euclidean" #"chronos_euclidean" # "raw_cosine"
    #             do
    #                 save_name="${model}_${distance}_${bs}"
    #                 if [ "$online" = "True" ]; then
    #                     save_name="${save_name}_online"
    #                 fi
    #                 if [ "$oracle" = "True" ]; then
    #                     save_name="${save_name}_oracle"
    #                 fi

    #                 srun python3 neighbors_all.py \
    #                 "extra.distance='$distance'" \
    #                 "extra.max_windows=$max_windows" \
    #                 "extra.online=$online" \
    #                 "extra.oracle=$oracle" \
    #                 "model='$model'" \
    #                 "model.configs.context_mode=future" \
    #                 "task.lags=$lags" \
    #                 "task.horizon=$horizon" \
    #                 "normalization=instance" \
    #                 "data=$data" \
    #                 "data.sampling.train_stride=$train_stride" \
    #                 "data.sampling.eval_stride=$eval_stride" \
    #                 "training.bs=$bs" \
    #                 "training.eval_runs=1" \
    #                 "misc.seed=$seed" \
    #                 "misc.device='gpu'" \
    #                 "misc.save_name='$save_name'" \
    #                 "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
    #             done
    #         done
    #     done
    # done

    #bs1
    srun python3 neighbors_all.py \
    "model='$model'" \
    "task.lags=$lags" \
    "task.horizon=$horizon" \
    "normalization=instance" \
    "data=$data" \
    "data.sampling.train_stride=$train_stride" \
    "data.sampling.eval_stride=$eval_stride" \
    "training.bs=1" \
    "training.eval_runs=1" \
    "misc.seed=$seed" \
    "misc.device='cpu'" \
    "misc.save_name='${model}_baseline'" \
    "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
done

settings="1344_336;672_168;168_168;168_24"
file_name=mean_results.json
model=chronos

model_names="chronos_baseline;chronos_raw_euclidean_2;chronos_raw_euclidean_2_online;chronos_raw_euclidean_4;chronos_raw_euclidean_4_online;chronos_raw_euclidean_2_oracle;chronos_raw_euclidean_2_online_oracle;chronos_raw_euclidean_4_oracle;chronos_raw_euclidean_4_online_oracle"

for metric in nMSE 'w10 nMSE' improvements
do
    output_path="${experiment_dir}neighbors_chronos_${metric}_results.tex"
    srun python3 -c "from src.timetensor.visu import generate_results_table;generate_results_table('$experiment_dir', model_names='$model_names', json_filename='$file_name', metric_key='$metric', output_tex_path='$output_path', settings='$settings', decimals=4)"
done
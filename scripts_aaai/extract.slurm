#!/bin/bash
#SBATCH --job-name=extract
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/extract.out
#SBATCH --error=script_outputs/extract.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100
#SBATCH --mem=40000

source .venv/bin/activate

seed=1
experiment_dir="../outputs/aaai/extract/"
model=chronos
date_splits='0.3;0.5;0.2'

# max_windows=10_000

lags=672
horizon=168
eval_stride=512
train_stride=24
for data in weather #electricity #traffic solar #weather exchange_rate ETTh1
do
    period=24
    
    for online in True #False
    do
        for space in "raw" "chronos"
        do
            metric="euclidean"
            distance="${space}_${metric}"            
            if [ "$space" = "raw" ]; then
                max_windows=10_000
            else
                max_windows=5000
            fi
            
            for neighbors in 1 3 10
            do
                save_name="${model}_${distance}_${neighbors}"
                if [ "$online" = "True" ]; then
                    save_name="${save_name}_online"
                fi

                srun python3 extract_neighbors.py \
                "extra.distance='$distance'" \
                "extra.max_windows=$max_windows" \
                "extra.online=$online" \
                "extra.period=$period" \
                "extra.neighbors=$neighbors" \
                "model='$model'" \
                "model.configs.context_mode=future" \
                "task.lags=$lags" \
                "task.horizon=$horizon" \
                "normalization=instance" \
                "data=$data" \
                "data.splits.date_splits='$date_splits'" \
                "data.sampling.train_stride=$train_stride" \
                "data.sampling.eval_stride=$eval_stride" \
                "misc.seed=$seed" \
                "misc.device='gpu'" \
                "misc.save_name='$save_name'" \
                "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
            done
        done
    done
done


# lags=672
# horizon=168
# eval_stride=512
# train_stride=24
# for data in electricity traffic solar #weather exchange_rate ETTh1
# do
#     # for online in False True
#     # do
#     #     for oracle in False True
#     #     do
#     #         for bs in 2 4
#     #         do
#     #             for distance in "raw_euclidean" #"chronos_euclidean" # "raw_cosine"
#     #             do
#     #                 save_name="${model}_${distance}_${bs}"
#     #                 if [ "$online" = "True" ]; then
#     #                     save_name="${save_name}_online"
#     #                 fi
#     #                 if [ "$oracle" = "True" ]; then
#     #                     save_name="${save_name}_oracle"
#     #                 fi

#     #                 srun python3 neighbors_all.py \
#     #                 "extra.distance='$distance'" \
#     #                 "extra.max_windows=$max_windows" \
#     #                 "extra.online=$online" \
#     #                 "extra.oracle=$oracle" \
#     #                 "model='$model'" \
#     #                 "model.configs.context_mode=future" \
#     #                 "task.lags=$lags" \
#     #                 "task.horizon=$horizon" \
#     #                 "normalization=instance" \
#     #                 "data=$data" \
#     #                 "data.sampling.train_stride=$train_stride" \
#     #                 "data.sampling.eval_stride=$eval_stride" \
#     #                 "training.bs=$bs" \
#     #                 "training.eval_runs=1" \
#     #                 "misc.seed=$seed" \
#     #                 "misc.device='gpu'" \
#     #                 "misc.save_name='$save_name'" \
#     #                 "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
#     #             done
#     #         done
#     #     done
#     # done

#     #bs1
#     srun python3 neighbors_all.py \
#     "model='$model'" \
#     "task.lags=$lags" \
#     "task.horizon=$horizon" \
#     "normalization=instance" \
#     "data=$data" \
#     "data.sampling.train_stride=$train_stride" \
#     "data.sampling.eval_stride=$eval_stride" \
#     "training.bs=1" \
#     "training.eval_runs=1" \
#     "misc.seed=$seed" \
#     "misc.device='cpu'" \
#     "misc.save_name='${model}_baseline'" \
#     "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
# done

# lags=168
# horizon=168
# eval_stride=512
# train_stride=24
# for data in electricity traffic solar #weather exchange_rate ETTh1
# do
#     # for online in False True
#     # do
#     #     for oracle in False True
#     #     do
#     #         for bs in 2 4
#     #         do
#     #             for distance in "raw_euclidean" #"chronos_euclidean" # "raw_cosine"
#     #             do
#     #                 save_name="${model}_${distance}_${bs}"
#     #                 if [ "$online" = "True" ]; then
#     #                     save_name="${save_name}_online"
#     #                 fi
#     #                 if [ "$oracle" = "True" ]; then
#     #                     save_name="${save_name}_oracle"
#     #                 fi

#     #                 srun python3 neighbors_all.py \
#     #                 "extra.distance='$distance'" \
#     #                 "extra.max_windows=$max_windows" \
#     #                 "extra.online=$online" \
#     #                 "extra.oracle=$oracle" \
#     #                 "model='$model'" \
#     #                 "model.configs.context_mode=future" \
#     #                 "task.lags=$lags" \
#     #                 "task.horizon=$horizon" \
#     #                 "normalization=instance" \
#     #                 "data=$data" \
#     #                 "data.sampling.train_stride=$train_stride" \
#     #                 "data.sampling.eval_stride=$eval_stride" \
#     #                 "training.bs=$bs" \
#     #                 "training.eval_runs=1" \
#     #                 "misc.seed=$seed" \
#     #                 "misc.device='gpu'" \
#     #                 "misc.save_name='$save_name'" \
#     #                 "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
#     #             done
#     #         done
#     #     done
#     # done

#     #bs1
#     srun python3 neighbors_all.py \
#     "model='$model'" \
#     "task.lags=$lags" \
#     "task.horizon=$horizon" \
#     "normalization=instance" \
#     "data=$data" \
#     "data.sampling.train_stride=$train_stride" \
#     "data.sampling.eval_stride=$eval_stride" \
#     "training.bs=1" \
#     "training.eval_runs=1" \
#     "misc.seed=$seed" \
#     "misc.device='cpu'" \
#     "misc.save_name='${model}_baseline'" \
#     "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
# done

# lags=168
# horizon=24
# eval_stride=512
# train_stride=24
# for data in electricity traffic solar #weather exchange_rate ETTh1
# do
#     # for online in False True
#     # do
#     #     for oracle in False True
#     #     do
#     #         for bs in 2 4
#     #         do
#     #             for distance in "raw_euclidean" #"chronos_euclidean" # "raw_cosine"
#     #             do
#     #                 save_name="${model}_${distance}_${bs}"
#     #                 if [ "$online" = "True" ]; then
#     #                     save_name="${save_name}_online"
#     #                 fi
#     #                 if [ "$oracle" = "True" ]; then
#     #                     save_name="${save_name}_oracle"
#     #                 fi

#     #                 srun python3 neighbors_all.py \
#     #                 "extra.distance='$distance'" \
#     #                 "extra.max_windows=$max_windows" \
#     #                 "extra.online=$online" \
#     #                 "extra.oracle=$oracle" \
#     #                 "model='$model'" \
#     #                 "model.configs.context_mode=future" \
#     #                 "task.lags=$lags" \
#     #                 "task.horizon=$horizon" \
#     #                 "normalization=instance" \
#     #                 "data=$data" \
#     #                 "data.sampling.train_stride=$train_stride" \
#     #                 "data.sampling.eval_stride=$eval_stride" \
#     #                 "training.bs=$bs" \
#     #                 "training.eval_runs=1" \
#     #                 "misc.seed=$seed" \
#     #                 "misc.device='gpu'" \
#     #                 "misc.save_name='$save_name'" \
#     #                 "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
#     #             done
#     #         done
#     #     done
#     # done

#     #bs1
#     srun python3 neighbors_all.py \
#     "model='$model'" \
#     "task.lags=$lags" \
#     "task.horizon=$horizon" \
#     "normalization=instance" \
#     "data=$data" \
#     "data.sampling.train_stride=$train_stride" \
#     "data.sampling.eval_stride=$eval_stride" \
#     "training.bs=1" \
#     "training.eval_runs=1" \
#     "misc.seed=$seed" \
#     "misc.device='cpu'" \
#     "misc.save_name='${model}_baseline'" \
#     "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
# done

# settings="1344_336;672_168;168_168;168_24"
# file_name=mean_results.json
# model=chronos

# model_names="chronos_baseline;chronos_raw_euclidean_2;chronos_raw_euclidean_2_online;chronos_raw_euclidean_4;chronos_raw_euclidean_4_online;chronos_raw_euclidean_2_oracle;chronos_raw_euclidean_2_online_oracle;chronos_raw_euclidean_4_oracle;chronos_raw_euclidean_4_online_oracle"

# for metric in nMSE 'w10 nMSE' improvements
# do
#     output_path="${experiment_dir}neighbors_chronos_${metric}_results.tex"
#     srun python3 -c "from src.timetensor.visu import generate_results_table;generate_results_table('$experiment_dir', model_names='$model_names', json_filename='$file_name', metric_key='$metric', output_tex_path='$output_path', settings='$settings', decimals=4)"
# done
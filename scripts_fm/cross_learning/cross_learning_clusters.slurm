#!/bin/bash
#SBATCH --job-name=cl_clusters
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/cl_clusters.out
#SBATCH --error=script_outputs/cl_clusters.err
#SBATCH --time=23:00:00
#SBATCH --partition=a100
source .venv/bin/activate

seed=1
experiment_dir="../outputs/fm/cross_learning/clusters/"

lags=672
horizon=168
stride=512
for data in solar #electricity traffic solar exchange_rate
do
    for bs in 1 2 #1 2 3 4 5 10
    do
        use_context=True
        cross_learning=False
        srun python3 cross_learning_clusters.py \
        "model=chronos" \
        "model.configs.cross_learning=$cross_learning" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "normalization=instance" \
        "data=$data" \
        "data.sampling.use_context=$use_context" \
        "data.sampling.train_stride=$stride" \
        "data.sampling.eval_stride=$stride" \
        "training.bs=$bs" \
        "training.eval_runs=1" \
        "misc.seed=$seed" \
        "misc.device='cpu'" \
        "misc.save_name='solar_context$bs'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

        use_context=True
        cross_learning=True
        srun python3 cross_learning_clusters.py \
        "model=chronos" \
        "model.configs.cross_learning=$cross_learning" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "normalization=instance" \
        "data=$data" \
        "data.sampling.use_context=$use_context" \
        "data.sampling.train_stride=$stride" \
        "data.sampling.eval_stride=$stride" \
        "training.bs=$bs" \
        "training.eval_runs=1" \
        "misc.seed=$seed" \
        "misc.device='cpu'" \
        "misc.save_name='solar_cross$bs'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

        use_context=False
        cross_learning=False
        srun python3 cross_learning_clusters.py \
        "model=chronos" \
        "model.configs.cross_learning=$cross_learning" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "normalization=instance" \
        "data=$data" \
        "data.sampling.use_context=$use_context" \
        "data.sampling.train_stride=$stride" \
        "data.sampling.eval_stride=$stride" \
        "training.bs=$bs" \
        "training.eval_runs=1" \
        "misc.seed=$seed" \
        "misc.device='cpu'" \
        "misc.save_name='solarNone$bs'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"


    done
done

# lags=168
# horizon=24
# stride=512
# for data in solar #electricity traffic solar exchange_rate
# do
#     for bs in 1 #1 2 3 4 5 10
#     do
#         use_context=False
#         cross_learning=True
#         srun python3 cross_learning_clusters.py \
#         "model=chronos" \
#         "model.configs.cross_learning=$cross_learning" \
#         "task.lags=$lags" \
#         "task.horizon=$horizon" \
#         "normalization=instance" \
#         "data=$data" \
#         "data.sampling.use_context=$use_context" \
#         "data.sampling.train_stride=$stride" \
#         "data.sampling.eval_stride=$stride" \
#         "training.bs=$bs" \
#         "training.eval_runs=1" \
#         "misc.seed=$seed" \
#         "misc.device='cpu'" \
#         "misc.save_name='chronos2_None_bs$bs'" \
#         "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
#     done
# done

# lags=24
# horizon=24
# stride=512
# for data in solar #electricity traffic solar exchange_rate
# do
#     for bs in 1 #1 2 3 4 5 10
#     do
#         use_context=False
#         cross_learning=True
#         srun python3 cross_learning_clusters.py \
#         "model=chronos" \
#         "model.configs.cross_learning=$cross_learning" \
#         "task.lags=$lags" \
#         "task.horizon=$horizon" \
#         "normalization=instance" \
#         "data=$data" \
#         "data.sampling.use_context=$use_context" \
#         "data.sampling.train_stride=$stride" \
#         "data.sampling.eval_stride=$stride" \
#         "training.bs=$bs" \
#         "training.eval_runs=1" \
#         "misc.seed=$seed" \
#         "misc.device='cpu'" \
#         "misc.save_name='chronos2_None_bs$bs'" \
#         "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
#     done
# done

file_name=mean_results.json
for metric in nMSE 'w10 nMSE' 'compute (min)'
do
    output_path="${experiment_dir}${metric}_results.tex"
    srun python3 -c "from src.timetensor.visu import generate_results_table;generate_results_table('$experiment_dir', json_filename='$file_name', metric_key='$metric', output_tex_path='$output_path', settings='$settings')"
done
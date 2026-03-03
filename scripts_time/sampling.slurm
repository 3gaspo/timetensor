#!/bin/bash
#SBATCH --job-name=sampling
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/sampling.out
#SBATCH --error=script_outputs/sampling.err
#SBATCH --time=23:00:00
#SBATCH --partition=a100
source .venv/bin/activate

seed=1
experiment_dir="../outputs/sampling/"

lags=672
horizon=168
stride=512
for data in electricity #traffic solar exchange_rate
do

    #indiv bs/2 
    srun python3 train_model.py \
    "model=PatchTST" \
    "task.lags=$lags" \
    "task.horizon=$horizon" \
    "data=$data" \
    "data.splits.date_splits='0.6;0.1;0.3'" \
    "data.sampling.train_idx_mode='individuals'" \
    "data.sampling.eval_idx_mode='all'" \
    "data.sampling.train_stride=1" \
    "data.sampling.eval_stride=$stride" \
    "training.bs=128" \
    "training.epochs=750" \
    "training.test_eval=False" \
    "misc.seed=$seed" \
    "misc.save_name='PatchTST_indiv_128'" \
    "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

    srun python3 eval_users.py \
    "model=PatchTST" \
    "task.lags=$lags" \
    "task.horizon=$horizon" \
    "data=$data" \
    "data.splits.date_splits='0.6;0.1;0.3'" \
    "data.sampling.train_idx_mode='individuals'" \
    "data.sampling.eval_idx_mode='all'" \
    "data.sampling.train_stride=1" \
    "data.sampling.eval_stride=$stride" \
    "training.bs=128" \
    "training.epochs=750" \
    "training.test_eval=False" \
    "misc.seed=$seed" \
    "misc.save_name='PatchTST_indiv_128'" \
    "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

    #indiv bs
    srun python3 train_model.py \
    "model=PatchTST" \
    "task.lags=$lags" \
    "task.horizon=$horizon" \
    "data=$data" \
    "data.splits.date_splits='0.6;0.1;0.3'" \
    "data.sampling.train_idx_mode='individuals'" \
    "data.sampling.eval_idx_mode='all'" \
    "data.sampling.train_stride=1" \
    "data.sampling.eval_stride=$stride" \
    "training.bs=256" \
    "training.epochs=1500" \
    "training.test_eval=False" \
    "misc.seed=$seed" \
    "misc.save_name='PatchTST_indiv_256'" \
    "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

    srun python3 eval_users.py \
    "model=PatchTST" \
    "task.lags=$lags" \
    "task.horizon=$horizon" \
    "data=$data" \
    "data.splits.date_splits='0.6;0.1;0.3'" \
    "data.sampling.train_idx_mode='individuals'" \
    "data.sampling.eval_idx_mode='all'" \
    "data.sampling.train_stride=1" \
    "data.sampling.eval_stride=$stride" \
    "training.bs=256" \
    "training.epochs=1500" \
    "training.test_eval=False" \
    "misc.seed=$seed" \
    "misc.save_name='PatchTST_indiv_256'" \
    "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

    #dates
    srun python3 train_model.py \
    "model=PatchTST" \
    "task.lags=$lags" \
    "task.horizon=$horizon" \
    "data=$data" \
    "data.splits.date_splits='0.6;0.1;0.3'" \
    "data.sampling.train_idx_mode='dates'" \
    "data.sampling.eval_idx_mode='all'" \
    "data.sampling.train_stride=1" \
    "data.sampling.eval_stride=$stride" \
    "training.bs=256" \
    "training.epochs=1" \
    "training.test_eval=False" \
    "misc.seed=$seed" \
    "misc.save_name='PatchTST_date'" \
    "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

    srun python3 eval_users.py \
    "model=PatchTST" \
    "task.lags=$lags" \
    "task.horizon=$horizon" \
    "data=$data" \
    "data.splits.date_splits='0.6;0.1;0.3'" \
    "data.sampling.train_idx_mode='dates'" \
    "data.sampling.eval_idx_mode='all'" \
    "data.sampling.train_stride=1" \
    "data.sampling.eval_stride=$stride" \
    "training.bs=256" \
    "training.epochs=1" \
    "training.test_eval=False" \
    "misc.seed=$seed" \
    "misc.save_name='PatchTST_date'" \
    "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

    #dates block 4 indiv
    srun python3 train_model.py \
    "model=PatchTST" \
    "task.lags=$lags" \
    "task.horizon=$horizon" \
    "normalization=instance" \
    "data=$data" \
    "data.splits.date_splits='0.6;0.1;0.3'" \
    "data.sampling.train_idx_mode='dates'" \
    "data.sampling.train_block_individuals=4" \
    "data.sampling.eval_idx_mode='all'" \
    "data.sampling.train_stride=1" \
    "data.sampling.eval_stride=$stride" \
    "training.bs=64" \
    "training.epochs=1" \
    "training.test_eval=False" \
    "misc.seed=$seed" \
    "misc.save_name='PatchTST_date_4'" \
    "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

    srun python3 eval_users.py \
    "model=PatchTST" \
    "task.lags=$lags" \
    "task.horizon=$horizon" \
    "normalization=instance" \
    "data=$data" \
    "data.splits.date_splits='0.6;0.1;0.3'" \
    "data.sampling.train_idx_mode='dates'" \
    "data.sampling.train_block_individuals=4" \
    "data.sampling.eval_idx_mode='all'" \
    "data.sampling.train_stride=1" \
    "data.sampling.eval_stride=$stride" \
    "training.bs=64" \
    "training.epochs=1" \
    "training.test_eval=False" \
    "misc.seed=$seed" \
    "misc.save_name='PatchTST_date_4'" \
    "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
done



settings="1344_336;672_168;168_24;24_24"
model_names="PatchTST_indiv128;PatchTST_indiv256;PatchTST_date;PatchTST_date_4"

for metric in nMSE 'w10 nMSE' 'train time (min)'
do
    for prefix in train test1
    do
        file_name="${prefix}_mean_results.json"
        output_path="${experiment_dir}${prefix}_${metric}_results.tex"
        srun python3 -c "from src.timetensor.visu import generate_results_table;generate_results_table('$experiment_dir', json_filename='$file_name', metric_key='$metric', output_tex_path='$output_path', model_names='$model_names', settings='$settings', decimals=4)"
    done
done
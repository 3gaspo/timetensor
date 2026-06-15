#!/bin/bash
#SBATCH --job-name=exotics
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/exotics.out
#SBATCH --error=script_outputs/exotics.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100-bis
source .venv/bin/activate

experiment_dir="../outputs/exotics/"

lags=1344
horizon=336
stride=512
for seed in 1 2 3 4 5
do
    for data in raw_electricity electricity traffic solar
    do
        #all
        srun python3 train_model.py \
        "model=PatchTST" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "data=$data" \
        "data.splits.date_splits='0.6;0.2;0.2'" \
        "data.sampling.train_idx_mode='individuals'" \
        "data.sampling.eval_idx_mode='all'" \
        "data.sampling.eval_stride=$stride" \
        "data.sampling.remove_train_cte=False" \
        "data.sampling.remove_eval_cte=False" \
        "training.bs=256" \
        "training.epochs=1000" \
        "training.test_eval=False" \
        "misc.seed=$seed" \
        "misc.save_name='PatchTST_all'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/$seed/'"

        srun python3 eval_users.py \
        "model=PatchTST" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "data=$data" \
        "data.splits.date_splits='0.6;0.2;0.2'" \
        "data.sampling.train_idx_mode='individuals'" \
        "data.sampling.eval_idx_mode='all'" \
        "data.sampling.eval_stride=$stride" \
        "data.sampling.remove_train_cte=False" \
        "data.sampling.remove_eval_cte=False" \
        "training.bs=256" \
        "training.epochs=1000" \
        "misc.seed=$seed" \
        "misc.save_name='PatchTST_all'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/$seed/'"

        # rmv train
        srun python3 train_model.py \
        "model=PatchTST" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "data=$data" \
        "data.splits.date_splits='0.6;0.2;0.2'" \
        "data.sampling.train_idx_mode='individuals'" \
        "data.sampling.eval_idx_mode='all'" \
        "data.sampling.eval_stride=$stride" \
        "data.sampling.remove_train_cte=True" \
        "data.sampling.remove_eval_cte=False" \
        "training.bs=256" \
        "training.epochs=1000" \
        "training.test_eval=False" \
        "misc.seed=$seed" \
        "misc.save_name='PatchTST_rmvtrain'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/$seed/'"

        srun python3 eval_users.py \
        "model=PatchTST" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "data=$data" \
        "data.splits.date_splits='0.6;0.2;0.2'" \
        "data.sampling.train_idx_mode='individuals'" \
        "data.sampling.eval_idx_mode='all'" \
        "data.sampling.eval_stride=$stride" \
        "data.sampling.remove_train_cte=True" \
        "data.sampling.remove_eval_cte=False" \
        "training.bs=256" \
        "training.epochs=1000" \
        "misc.seed=$seed" \
        "misc.save_name='PatchTST_rmvtrain'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/$seed/'"

        #rmv all
        srun python3 train_model.py \
        "model=PatchTST" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "data=$data" \
        "data.splits.date_splits='0.6;0.2;0.2'" \
        "data.sampling.train_idx_mode='individuals'" \
        "data.sampling.eval_idx_mode='all'" \
        "data.sampling.eval_stride=$stride" \
        "data.sampling.remove_train_cte=True" \
        "data.sampling.remove_eval_cte=True" \
        "training.bs=256" \
        "training.epochs=1000" \
        "training.test_eval=False" \
        "misc.seed=$seed" \
        "misc.save_name='PatchTST_rmvall'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/$seed/'"

        srun python3 eval_users.py \
        "model=PatchTST" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "data=$data" \
        "data.splits.date_splits='0.6;0.2;0.2'" \
        "data.sampling.train_idx_mode='individuals'" \
        "data.sampling.eval_idx_mode='all'" \
        "data.sampling.eval_stride=$stride" \
        "data.sampling.remove_train_cte=True" \
        "data.sampling.remove_eval_cte=True" \
        "training.bs=256" \
        "training.epochs=1000" \
        "misc.seed=$seed" \
        "misc.save_name='PatchTST_rmvall'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/$seed/'"

        #rmv test
        srun python3 train_model.py \
        "model=PatchTST" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "data=$data" \
        "data.splits.date_splits='0.6;0.2;0.2'" \
        "data.sampling.train_idx_mode='individuals'" \
        "data.sampling.eval_idx_mode='all'" \
        "data.sampling.eval_stride=$stride" \
        "data.sampling.remove_train_cte=False" \
        "data.sampling.remove_eval_cte=True" \
        "training.bs=256" \
        "training.epochs=1000" \
        "training.test_eval=False" \
        "misc.seed=$seed" \
        "misc.save_name='PatchTST_rmvtest'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/$seed/'"
        
        srun python3 eval_users.py \
        "model=PatchTST" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "data=$data" \
        "data.splits.date_splits='0.6;0.2;0.2'" \
        "data.sampling.train_idx_mode='individuals'" \
        "data.sampling.eval_idx_mode='all'" \
        "data.sampling.eval_stride=$stride" \
        "data.sampling.remove_train_cte=False" \
        "data.sampling.remove_eval_cte=True" \
        "training.bs=256" \
        "training.epochs=1000" \
        "misc.seed=$seed" \
        "misc.save_name='PatchTST_rmvtest'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/$seed/'"
    done
done


lags=672
horizon=168
stride=512
for seed in 1 2 3 4 5
do
    for data in raw_electricity electricity traffic solar
    do

        #all
        srun python3 train_model.py \
        "model=PatchTST" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "data=$data" \
        "data.splits.date_splits='0.6;0.2;0.2'" \
        "data.sampling.train_idx_mode='individuals'" \
        "data.sampling.eval_idx_mode='all'" \
        "data.sampling.eval_stride=$stride" \
        "data.sampling.remove_train_cte=False" \
        "data.sampling.remove_eval_cte=False" \
        "training.bs=256" \
        "training.epochs=1000" \
        "training.test_eval=False" \
        "misc.seed=$seed" \
        "misc.save_name='PatchTST_all'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/$seed/'"

        srun python3 eval_users.py \
        "model=PatchTST" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "data=$data" \
        "data.splits.date_splits='0.6;0.2;0.2'" \
        "data.sampling.train_idx_mode='individuals'" \
        "data.sampling.eval_idx_mode='all'" \
        "data.sampling.eval_stride=$stride" \
        "data.sampling.remove_train_cte=False" \
        "data.sampling.remove_eval_cte=False" \
        "training.bs=256" \
        "training.epochs=1000" \
        "misc.seed=$seed" \
        "misc.save_name='PatchTST_all'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/$seed/'"

        # rmv train
        srun python3 train_model.py \
        "model=PatchTST" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "data=$data" \
        "data.splits.date_splits='0.6;0.2;0.2'" \
        "data.sampling.train_idx_mode='individuals'" \
        "data.sampling.eval_idx_mode='all'" \
        "data.sampling.eval_stride=$stride" \
        "data.sampling.remove_train_cte=True" \
        "data.sampling.remove_eval_cte=False" \
        "training.bs=256" \
        "training.epochs=1000" \
        "training.test_eval=False" \
        "misc.seed=$seed" \
        "misc.save_name='PatchTST_rmvtrain'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/$seed/'"

        srun python3 eval_users.py \
        "model=PatchTST" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "data=$data" \
        "data.splits.date_splits='0.6;0.2;0.2'" \
        "data.sampling.train_idx_mode='individuals'" \
        "data.sampling.eval_idx_mode='all'" \
        "data.sampling.eval_stride=$stride" \
        "data.sampling.remove_train_cte=True" \
        "data.sampling.remove_eval_cte=False" \
        "training.bs=256" \
        "training.epochs=1000" \
        "misc.seed=$seed" \
        "misc.save_name='PatchTST_rmvtrain'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/$seed/'"

        #rmv all
        srun python3 train_model.py \
        "model=PatchTST" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "data=$data" \
        "data.splits.date_splits='0.6;0.2;0.2'" \
        "data.sampling.train_idx_mode='individuals'" \
        "data.sampling.eval_idx_mode='all'" \
        "data.sampling.eval_stride=$stride" \
        "data.sampling.remove_train_cte=True" \
        "data.sampling.remove_eval_cte=True" \
        "training.bs=256" \
        "training.epochs=1000" \
        "training.test_eval=False" \
        "misc.seed=$seed" \
        "misc.save_name='PatchTST_rmvall'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/$seed/'"

        srun python3 eval_users.py \
        "model=PatchTST" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "data=$data" \
        "data.splits.date_splits='0.6;0.2;0.2'" \
        "data.sampling.train_idx_mode='individuals'" \
        "data.sampling.eval_idx_mode='all'" \
        "data.sampling.eval_stride=$stride" \
        "data.sampling.remove_train_cte=True" \
        "data.sampling.remove_eval_cte=True" \
        "training.bs=256" \
        "training.epochs=1000" \
        "misc.seed=$seed" \
        "misc.save_name='PatchTST_rmvall'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/$seed/'"

        #rmv test
        srun python3 train_model.py \
        "model=PatchTST" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "data=$data" \
        "data.splits.date_splits='0.6;0.2;0.2'" \
        "data.sampling.train_idx_mode='individuals'" \
        "data.sampling.eval_idx_mode='all'" \
        "data.sampling.eval_stride=$stride" \
        "data.sampling.remove_train_cte=False" \
        "data.sampling.remove_eval_cte=True" \
        "training.bs=256" \
        "training.epochs=1000" \
        "training.test_eval=False" \
        "misc.seed=$seed" \
        "misc.save_name='PatchTST_rmvtest'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/$seed/'"
        
        srun python3 eval_users.py \
        "model=PatchTST" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "data=$data" \
        "data.splits.date_splits='0.6;0.2;0.2'" \
        "data.sampling.train_idx_mode='individuals'" \
        "data.sampling.eval_idx_mode='all'" \
        "data.sampling.eval_stride=$stride" \
        "data.sampling.remove_train_cte=False" \
        "data.sampling.remove_eval_cte=True" \
        "training.bs=256" \
        "training.epochs=1000" \
        "misc.seed=$seed" \
        "misc.save_name='PatchTST_rmvtest'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/$seed/'"
    done
done

lags=168
horizon=24
stride=512
for seed in 1 2 3 4 5
do
    for data in raw_electricity electricity traffic solar
    do

        #all
        srun python3 train_model.py \
        "model=PatchTST" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "data=$data" \
        "data.splits.date_splits='0.6;0.2;0.2'" \
        "data.sampling.train_idx_mode='individuals'" \
        "data.sampling.eval_idx_mode='all'" \
        "data.sampling.eval_stride=$stride" \
        "data.sampling.remove_train_cte=False" \
        "data.sampling.remove_eval_cte=False" \
        "training.bs=256" \
        "training.epochs=1000" \
        "training.test_eval=False" \
        "misc.seed=$seed" \
        "misc.save_name='PatchTST_all'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/$seed/'"

        srun python3 eval_users.py \
        "model=PatchTST" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "data=$data" \
        "data.splits.date_splits='0.6;0.2;0.2'" \
        "data.sampling.train_idx_mode='individuals'" \
        "data.sampling.eval_idx_mode='all'" \
        "data.sampling.eval_stride=$stride" \
        "data.sampling.remove_train_cte=False" \
        "data.sampling.remove_eval_cte=False" \
        "training.bs=256" \
        "training.epochs=1000" \
        "misc.seed=$seed" \
        "misc.save_name='PatchTST_all'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/$seed/'"

        # rmv train
        srun python3 train_model.py \
        "model=PatchTST" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "data=$data" \
        "data.splits.date_splits='0.6;0.2;0.2'" \
        "data.sampling.train_idx_mode='individuals'" \
        "data.sampling.eval_idx_mode='all'" \
        "data.sampling.eval_stride=$stride" \
        "data.sampling.remove_train_cte=True" \
        "data.sampling.remove_eval_cte=False" \
        "training.bs=256" \
        "training.epochs=1000" \
        "training.test_eval=False" \
        "misc.seed=$seed" \
        "misc.save_name='PatchTST_rmvtrain'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/$seed/'"

        srun python3 eval_users.py \
        "model=PatchTST" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "data=$data" \
        "data.splits.date_splits='0.6;0.2;0.2'" \
        "data.sampling.train_idx_mode='individuals'" \
        "data.sampling.eval_idx_mode='all'" \
        "data.sampling.eval_stride=$stride" \
        "data.sampling.remove_train_cte=True" \
        "data.sampling.remove_eval_cte=False" \
        "training.bs=256" \
        "training.epochs=1000" \
        "misc.seed=$seed" \
        "misc.save_name='PatchTST_rmvtrain'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/$seed/'"

        #rmv all
        srun python3 train_model.py \
        "model=PatchTST" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "data=$data" \
        "data.splits.date_splits='0.6;0.2;0.2'" \
        "data.sampling.train_idx_mode='individuals'" \
        "data.sampling.eval_idx_mode='all'" \
        "data.sampling.eval_stride=$stride" \
        "data.sampling.remove_train_cte=True" \
        "data.sampling.remove_eval_cte=True" \
        "training.bs=256" \
        "training.epochs=1000" \
        "training.test_eval=False" \
        "misc.seed=$seed" \
        "misc.save_name='PatchTST_rmvall'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/$seed/'"

        srun python3 eval_users.py \
        "model=PatchTST" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "data=$data" \
        "data.splits.date_splits='0.6;0.2;0.2'" \
        "data.sampling.train_idx_mode='individuals'" \
        "data.sampling.eval_idx_mode='all'" \
        "data.sampling.eval_stride=$stride" \
        "data.sampling.remove_train_cte=True" \
        "data.sampling.remove_eval_cte=True" \
        "training.bs=256" \
        "training.epochs=1000" \
        "misc.seed=$seed" \
        "misc.save_name='PatchTST_rmvall'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/$seed/'"

        #rmv test
        srun python3 train_model.py \
        "model=PatchTST" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "data=$data" \
        "data.splits.date_splits='0.6;0.2;0.2'" \
        "data.sampling.train_idx_mode='individuals'" \
        "data.sampling.eval_idx_mode='all'" \
        "data.sampling.eval_stride=$stride" \
        "data.sampling.remove_train_cte=False" \
        "data.sampling.remove_eval_cte=True" \
        "training.bs=256" \
        "training.epochs=1000" \
        "training.test_eval=False" \
        "misc.seed=$seed" \
        "misc.save_name='PatchTST_rmvtest'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/$seed/'"
        
        srun python3 eval_users.py \
        "model=PatchTST" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "data=$data" \
        "data.splits.date_splits='0.6;0.2;0.2'" \
        "data.sampling.train_idx_mode='individuals'" \
        "data.sampling.eval_idx_mode='all'" \
        "data.sampling.eval_stride=$stride" \
        "data.sampling.remove_train_cte=False" \
        "data.sampling.remove_eval_cte=True" \
        "training.bs=256" \
        "training.epochs=1000" \
        "misc.seed=$seed" \
        "misc.save_name='PatchTST_rmvtest'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/$seed/'"
    done
done


settings="1344_336;672_168;168_24"

model_names="PatchTST_all;PatchTST_rmvtrain;PatchTST_rmvall;PatchTST_rmvtest"

for metric in nMSE 'w10 nMSE' 'train time (min)'
do
    for prefix in train test1
    do
        file_name="${prefix}_mean_results.json"
        output_path="${experiment_dir}${prefix}_${metric}_results.tex"
        srun python3 -c "from src.timetensor.visu import generate_results_table;generate_results_table('$experiment_dir', json_filename='$file_name', metric_key='$metric', output_tex_path='$output_path', model_names='$model_names', settings='$settings', decimals=4, seeds=True, show_std=False)"
    done
done

#!/bin/bash
#SBATCH --job-name=results
#SBATCH --gres=gpu:1
#SBATCH --output=new_scripts/results.out
#SBATCH --error=new_scripts/results.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100

source .venv/bin/activate

for model_name in PatchTST DLinear
do
    save_dir="../outputs/syn_$model_name/"

    datasets="synthetic;synthetic;synthetic"
    settings="40-10;100-20;100-100"
    n_paths="5;5;5"
    models="None_MSE;global_MSE;instance_MSE;instance_NMSE;revin_NMSE;revin_NMSE;cmIn_initFalse;cmIn_initTrue"

    #test 1 mse
    show_row=0
    file_name=test1_mean_results.json
    save_name="test1_MSE.tex"
    title="Test 1 (1e1 MSE)"
    baseline="instance_MSE"
    multipliers="5;5;5"
    srun python3 -c "from src.timetensor.visu import build_results_table_latex;build_results_table_latex('$save_dir', '$datasets', '$settings', $show_row, '$models', '$file_name', '$n_paths', '$multipliers', '$baseline', '$title', '$save_name')"

    show_row=1
    file_name=test1_mean_results.json
    save_name="test1_nMSE.tex"
    title="Test 1 (1e1 MSE)"
    baseline="None_MSE"
    multipliers="5;5;5"
    srun python3 -c "from src.timetensor.visu import build_results_table_latex;build_results_table_latex('$save_dir', '$datasets', '$settings', $show_row, '$models', '$file_name', '$n_paths', '$multipliers', '$baseline', '$title', '$save_name')"

done

# #test 1 nmse
# show_row=1
# file_name=test1_mean_results.json
# save_name="test1_nMSE.tex"
# title="Test 1 (1e0 nMSE)"
# multipliers="1 0"
# baseline="instance_NMSE"
# srun python3 -c "from src.timetensor.visu import build_results_table_latex;build_results_table_latex('$save_dir', '$datasets', '$settings', $show_row, '$models', '$file_name', '$n_paths', '$multipliers', '$baseline', '$title', '$save_name')"

# #test 2 mse
# show_row=0
# file_name=test2_mean_results.json
# save_name="test2_MSE.tex"
# title="Test 2 (1e1 MSE)"
# multipliers="1 0"
# baseline="instance_NMSE"
# srun python3 -c "from src.timetensor.visu import build_results_table_latex;build_results_table_latex('$save_dir', '$datasets', '$settings', $show_row, '$models', '$file_name', '$n_paths', '$multipliers', '$baseline', '$title', '$save_name')"

# #test 1 nmse
# show_row=1
# file_name=test2_mean_results.json
# save_name="test2_nMSE.tex"
# title="Test 2 (1e0 nMSE)"
# multipliers="1 0"
# baseline="instance_NMSE"
# srun python3 -c "from src.timetensor.visu import build_results_table_latex;build_results_table_latex('$save_dir', '$datasets', '$settings', $show_row, '$models', '$file_name', '$n_paths', '$multipliers', '$baseline', '$title', '$save_name')"

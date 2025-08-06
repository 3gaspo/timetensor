#!/bin/bash
#SBATCH --job-name=bench_t
#SBATCH --gres=gpu:1
#SBATCH --output=scripts/bench_t.out
#SBATCH --error=scripts/bench_t.err
#SBATCH --time=01:00:00
#SBATCH --mem=10G
source .venv/bin/activate

expe_name=benchmark_tst
names=None

output_dir="../outputs/$expe_name/"
save_path="../outputs/$expe_name/plots/"
eval_freq=100

#python3 -c "from src.timetensor.visu import plot_expe;plot_expe('$output_dir', $eval_freq, $names, save_path='$save_path')"
srun python3 -c "from src.timetensor.visu import plot_expe;plot_expe('$output_dir', $eval_freq, '$names', save_path='$save_path')"

multipliers="-7 0"
srun python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}test1_mean_results.json', multipliers='$multipliers', names='$names')"
srun python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}test2_mean_results.json', multipliers='$multipliers', names='$names')"
# nohup bash scripts/read.sh > read.log 2>&1 &
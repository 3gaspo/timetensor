source .venv/bin/activate

output_dir="../outputs/benchmark_min_1/"
eval_freq=50
names=None
save_path="../outputs/benchmark_min_1_plots/"

python3 -c "from src.timetensor.visu import plot_expe;plot_expe('$output_dir', $eval_freq, $names, save_path='$save_path')"

multipliers="2 1 1 5"
python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}test1_mean_results.json', multipliers='$multipliers', names=$names)"

# nohup bash scripts/read.sh > read.log 2>&1 &
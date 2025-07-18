source .venv/bin/activate

output_dir="../outputs/benchmark_min/"
rm -rf "$output_dir"
mkdir -p "$output_dir"


model_name=PatchTST
lr=0.0001
epochs=150

latent=True
loss=normalize_y
normalization=mIN

python3 train_model.py \
    "model=${model_name}" \
    "misc.output_dir=$output_dir" \
    "normalization=$normalization" \
    "training.loss=$loss" \
    "misc.benchmark=True" \
    "training.lr=$lr" \
    "training.epochs=$epochs" \
    "training.eval_freq=50" \
    "training.print_freq=100"

python3 -c "from src.timetensor.visu import plot_expe;plot_expe('$output_dir', $eval_freq)"

multipliers="2 1 1 5"
python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}test1_mean_results.json', multipliers='$multipliers')"
python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}test2_mean_results.json', multipliers='$multipliers')"

# nohup bash scripts/benchmark_min.sh > benchmark_min.log 2>&1 &
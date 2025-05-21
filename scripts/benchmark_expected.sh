source .venv/bin/activate

model_name=expected
output_dir="../outputs/benchmark_${model_name}/"
rm -rf "$output_dir"
mkdir -p "$output_dir"


python3 train_model.py \
    "model.name=${model_name}" \
    "model_configs=${model_name}" \
    "misc.output_dir=$output_dir" \
    "misc.benchmark=True" \
    "training.bs=64" \
    "data.by_idx=individuals" \
    "misc.save_name=${model_name}_1"

python3 train_model.py \
    "model.name=${model_name}" \
    "model_configs=${model_name}" \
    "misc.output_dir=$output_dir" \
    "misc.benchmark=True" \
    "training.bs=64" \
    "data.by_idx=individuals" \
    "misc.save_name=${model_name}_2"

python3 train_model.py \
    "model.name=${model_name}" \
    "model_configs=${model_name}" \
    "misc.output_dir=$output_dir" \
    "misc.benchmark=True" \
    "training.bs=64" \
    "data.by_idx=individuals" \
    "misc.save_name=${model_name}_3"


python3 -c "from src.timetensor.visu import plot_expe;plot_expe('$output_dir')"

multipliers="-3 0 2"
python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}mean_results.json', multipliers='$multipliers')"

# nohup bash scripts/benchmark.sh > benchmark.log 2>&1 &
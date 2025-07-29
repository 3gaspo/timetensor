source .venv/bin/activate

output_dir="../outputs/benchmark_models/"
rm -rf "$output_dir"
mkdir -p "$output_dir"

#lookback
model_name=repeat
python3 train_model.py \
    "model=${model_name}" \
    "misc.output_dir=$output_dir" \
    "misc.benchmark=True"

loss=MSE
lr=0.0001
epochs=150

for model_name in DLinear PatchTST
do
    for normalization in None instance revin
    do
        python3 train_model.py \
            "model=${model_name}" \
            "misc.output_dir=$output_dir" \
            "normalization=$normalization" \
            "training.loss=$loss" \
            "misc.benchmark=True" \
            "training.lr=$lr" \
            "training.epochs=$epochs" \
            "training.eval_freq=100" \
            "training.print_freq=100"
    done
done

python3 -c "from src.timetensor.visu import plot_expe;plot_expe('$output_dir', $eval_freq)"

multipliers="-8 -3 0 0"
python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}test1_mean_results.json', multipliers='$multipliers')"
python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}test2_mean_results.json', multipliers='$multipliers')"

# nohup bash scripts/benchmark_models.sh > benchmark_models.log 2>&1 &
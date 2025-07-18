source .venv/bin/activate

output_dir="../outputs/benchmark_tst/"
rm -rf "$output_dir"
mkdir -p "$output_dir"

#lookback
model_name=repeat
python3 train_model.py \
    "model=${model_name}" \
    "misc.output_dir=$output_dir" \
    "misc.benchmark=True"
    
model_name=PatchTST
lr=0.0001
epochs=150

latent=False
loss=NMSE

for normalization in instance revin
do
    python3 train_model.py \
        "model=${model_name}" \
        "misc.output_dir=$output_dir" \
        "normalization=$normalization" \
        "normalization.configs.latent=$latent" \
        "training.loss=$loss" \
        "misc.benchmark=True" \
        "training.lr=$lr" \
        "training.epochs=$epochs" \
        "training.eval_freq=50" \
        "training.print_freq=100"
done

python3 -c "from src.timetensor.visu import plot_expe;plot_expe('$output_dir', $eval_freq)"

multipliers="-8 -3 0 0"
python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}test1_mean_results.json', multipliers='$multipliers')"
python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}test2_mean_results.json', multipliers='$multipliers')"

# nohup bash scripts/benchmark_tst.sh > benchmark_tst.log 2>&1 &
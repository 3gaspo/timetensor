source .venv/bin/activate

output_dir="../outputs/benchmark_baselines/"
rm -rf "$output_dir"
mkdir -p "$output_dir"

#baselines
for model_name in expected persistence repeat
do 
    python3 train_model.py \
        "model=${model_name}" \
        "misc.output_dir=$output_dir" \
        "misc.benchmark=True"
done

model_name=lookback
for lookback_idx in 0 168
do 
    python3 train_model.py \
        "model=${model_name}" \
        "misc.output_dir=$output_dir" \
        "misc.benchmark=True" \
        "model.configs.lookback_idx=$lookback_idx" \
        "misc.save_name=lookback_$lookback_idx"
done


multipliers="-8 -3 0 0"
python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}test1_mean_results.json', multipliers='$multipliers')"


# nohup bash scripts/benchmark_baselines.sh > benchmark_baselines.log 2>&1 &
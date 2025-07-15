source .venv/bin/activate

output_dir="../outputs/benchmark_linear/"
rm -rf "$output_dir"
mkdir -p "$output_dir"

baselines
for model_name in expected
do 
    python3 train_model.py \
        "model=${model_name}" \
        "misc.output_dir=$output_dir" \
        "misc.benchmark=True"
done

#sklinear
model_name=sklinear
for normalization in None instance relative
do
    for subset in 0.1 1
    do
        python3 train_model.py \
            "model=${model_name}" \
            "misc.save_name=${model_name}_${normalization}_${subset}" \
            "misc.output_dir=$output_dir" \
            "misc.benchmark=True" \
            "model.normalization=$normalization" \
            "data.subsets.train=$subset" \
            "data.by_idx=dates"
    done
done

#linear
model_name=linear
for normalization in None instance relative
do
    for subset in 1 0.1
    do
        python3 train_model.py \
            "model=${model_name}" \
            "misc.output_dir=$output_dir" \
            "misc.benchmark=True" \
            "model.normalization=$normalization" \
            "data.subsets.train=$subset" \
            "data.by_idx=individuals" \
            "training.loss=MSE" \
            "training.bs=28" \
            "training.epochs=10" \
            "training.eval_freq=10" \
            "training.print_freq=10"
    done
done

multipliers="-8 -3 -10 -10"
python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}test1_mean_results.json', multipliers='$multipliers')"

# nohup bash scripts/benchmark_linear.sh > benchmark_linear.log 2>&1 &
source .venv/bin/activate

output_dir="../outputs/benchmark_min/"
rm -rf "$output_dir"
mkdir -p "$output_dir"

model_name=expected
python3 train_model.py \
    "model=${model_name}" \
    "misc.output_dir=$output_dir" \
    "misc.benchmark=True" \
    "data.by_idx=individuals" \
    "misc.save_name=${model_name}"

model_name=sklinear
python3 train_model.py \
    "model=${model_name}" \
    "misc.output_dir=$output_dir" \
    "misc.benchmark=True" \
    "data.by_idx=individuals" \
    "misc.save_name=${model_name}"

by_idx=individuals
model_name=PatchTST
for normalization in None instance revin mIN
do
    for loss in MSE
    do
        for lr in 0.0001
        do
            python3 train_model.py \
                "model=${model_name}" \
                "misc.output_dir=$output_dir" \
                "model.normalization=$normalization" \
                "training.loss=$loss" \
                "misc.benchmark=True" \
                "training.bs=10" \
                "training.lr=$lr" \
                "training.epochs=100" \
                "training.eval_freq=50" \
                "training.print_freq=100" \
                "data.by_idx=$by_idx"
        done
    done
done

python3 -c "from src.timetensor.visu import plot_expe;plot_expe('$output_dir')"

multipliers="0 0"
python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}test1_mean_results.json', multipliers='$multipliers')"

# nohup bash scripts/benchmark_min.sh > benchmark_min.log 2>&1 &
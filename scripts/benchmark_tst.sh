source .venv/bin/activate

output_dir="../outputs/benchmark_tst/"
rm -rf "$output_dir"
mkdir -p "$output_dir"

by_idx=individuals
for model_name in PatchTST
do
    for normalization in instance
    do
        for loss in NMSE
        do
            for lr in 0.0001 0.00001
            do
                python3 train_model.py \
                    "model.name=${model_name}" \
                    "model_configs=${model_name}" \
                    "misc.output_dir=$output_dir" \
                    "model.normalization=$normalization" \
                    "training.loss=$loss" \
                    "misc.benchmark=True" \
                    "training.bs=28" \
                    "training.lr=$lr" \
                    "training.epochs=40" \
                    "training.eval_freq=5" \
                    "training.print_freq=10" \
                    "data.by_idx=$by_idx"
            done
        done
    done
done


python3 -c "from src.timetensor.visu import plot_expe;plot_expe('$output_dir')"

multipliers="-3 0 2"
python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}mean_results.json', multipliers='$multipliers')"

# nohup bash scripts/benchmark_tst.sh > benchmark_tst.log 2>&1 &
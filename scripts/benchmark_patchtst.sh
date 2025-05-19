source .venv/bin/activate

output_dir="../outputs/benchmark_patchtst/"
rm -rf "$output_dir"
mkdir -p "$output_dir"

for normalization in global
do
    for loss in MMSE
    do
        python3 train_model.py \
            "model.name=PatchTST" \
            "model_configs=PatchTST" \
            "misc.output_dir=$output_dir" \
            "model.normalization=$normalization" \
            "training.loss=$loss" \
            "misc.benchmark=True" \
            "training.bs=32" \
            "training.epochs=100" \
            "data.by_idx=indiv"
    done
done

python3 -c "from src.timetensor.visu import plot_expe;plot_expe('$output_dir')"

multipliers="3 1 2"
python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}mean_results.json', multipliers='$multipliers')"

# nohup bash scripts/benchmark_idx.sh > fed_benchmark.log 2>&1 &
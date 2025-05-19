source .venv/bin/activate

output_dir="../outputs/benchmark_indiv/"
rm -rf "$output_dir"
mkdir -p "$output_dir"

by_idx=individuals
for model_name in linear DLinear PatchTST
do
    for normalization in None
    do
        for loss in NMSE
        do
            for lr in 0.0001
            do
                python3 train_model.py \
                    "model.name=${model_name}" \
                    "model_configs=${model_name}" \
                    "misc.output_dir=$output_dir" \
                    "model.normalization=$normalization" \
                    "training.loss=$loss" \
                    "misc.benchmark=True" \
                    "training.bs=32" \
                    "training.lr=$lr" \
                    "training.epochs=20" \
                    "data.by_idx=$by_idx"
            done
        done
    done
done

by_idx=individuals
for model_name in linear DLinear PatchTST
do
    for normalization in revin_latent
    do
        for loss in MSE
        do
            for lr in 0.0001
            do
                python3 train_model.py \
                    "model.name=${model_name}" \
                    "model_configs=${model_name}" \
                    "misc.output_dir=$output_dir" \
                    "model.normalization=$normalization" \
                    "training.loss=$loss" \
                    "misc.benchmark=True" \
                    "training.bs=32" \
                    "training.lr=$lr" \
                    "training.epochs=20" \
                    "data.by_idx=$by_idx"
            done
        done
    done
done


python3 -c "from src.timetensor.visu import plot_expe;plot_expe('$output_dir')"

multipliers="6 2 2 2"
python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}mean_results.json', multipliers='$multipliers')"

# nohup bash scripts/benchmark_indiv.sh > benchmark_indiv.log 2>&1 &
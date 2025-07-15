source .venv/bin/activate

output_dir="../outputs/benchmark_tst/"
rm -rf "$output_dir"
mkdir -p "$output_dir"

#baselines
for model_name in expected lookback
do 
    python3 train_model.py \
        "model=${model_name}" \
        "misc.output_dir=$output_dir" \
        "misc.benchmark=True"
done

#sklinear
model_name=sklinear
normalization=None
python3 train_model.py \
    "model=${model_name}" \
    "misc.output_dir=$output_dir" \
    "misc.benchmark=True" \
    "model.normalization=$normalization"

multipliers="-8 -3 -6 0"
python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}test1_mean_results.json', multipliers='$multipliers')"

#tst

model_name=PatchTST
for normalization in None
do
    for loss in MSE
    do
        for lr in 1
        do
            python3 train_model.py \
                "model=${model_name}" \
                "misc.output_dir=$output_dir" \
                "model.normalization=$normalization" \
                "training.loss=$loss" \
                "misc.benchmark=True" \
                "training.bs=28" \
                "training.lr=$lr" \
                "training.epochs=100" \
                "training.eval_freq=50" \
                "training.print_freq=100"
        done
    done
done


model_name=PatchTST
for normalization in instance revin mIN
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
                "training.bs=28" \
                "training.lr=$lr" \
                "training.epochs=100" \
                "training.eval_freq=50" \
                "training.print_freq=100"
        done
    done
done

python3 -c "from src.timetensor.visu import plot_expe;plot_expe('$output_dir')"

multipliers="-8 -3 -6 0"
python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}test1_mean_results.json', multipliers='$multipliers')"

# nohup bash scripts/benchmark_tst.sh > benchmark_tst.log 2>&1 &
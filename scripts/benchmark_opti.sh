source .venv/bin/activate

output_dir="../outputs/benchmark_opti/"
rm -rf "$output_dir"
mkdir -p "$output_dir"

by_idx=dates
for model_name in linear
do
    for normalization in instance
    do
        for loss in RMSE
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
                    "training.bs=2" \
                    "training.lr=$lr" \
                    "training.epochs=2" \
                    "training.eval_freq=100" \
                    "training.print_freq=1000" \
                    "data.by_idx=$by_idx"
            done
        done
    done
done

by_idx=dates
for model_name in linear
do
    for normalization in instance
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
                    "training.bs=2" \
                    "training.lr=$lr" \
                    "training.epochs=2" \
                    "training.eval_freq=100" \
                    "training.print_freq=1000" \
                    "data.by_idx=$by_idx"
            done
        done
    done
done

by_idx=dates
for model_name in linear
do
    for normalization in instance
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
                    "training.bs=2" \
                    "training.lr=$lr" \
                    "training.epochs=2" \
                    "training.eval_freq=100" \
                    "training.print_freq=1000" \
                    "data.by_idx=$by_idx"
            done
        done
    done
done


by_idx=individuals
for model_name in linear
do
    for normalization in instance
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
                    "training.bs=28" \
                    "training.lr=$lr" \
                    "training.epochs=1200" \
                    "training.eval_freq=100" \
                    "training.print_freq=1000" \
                    "data.by_idx=$by_idx" \
                    "misc.save_name=linear_indiv_instance_NMSE"
            done
        done
    done
done


python3 -c "from src.timetensor.visu import plot_expe;plot_expe('$output_dir')"

multipliers="-3 0 2"
python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}mean_results.json', multipliers='$multipliers')"
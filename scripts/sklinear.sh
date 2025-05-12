source .venv/bin/activate

output_dir="../outputs/sklinear_benchmark/"
rm -rf "$output_dir"
mkdir -p "$output_dir"

model_name=sklinear
for subset in partial full
do
    for normalize_method in None instance relative
    do
        python3 train_model.py \
            "model.name=$model_name" \
            "model_configs=$model_name" \
            "misc.output_dir=$output_dir" \
            "misc.benchmark=True" \
            "training.retrain=True" \
            "model.normalization=0" \
            "misc.save_name=sk_${subset}_${normalize_method}" \
            "subset=$subset" \
            "model_configs.normalize_method=$normalize_method"
    done
done

multipliers="3 1 2"
python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}mean_results.json', multipliers='$multipliers')"
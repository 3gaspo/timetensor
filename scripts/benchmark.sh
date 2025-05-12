source .venv/bin/activate

output_dir="../outputs/benchmark/"
rm -rf "$output_dir"
mkdir -p "$output_dir"

#baselines
for model_name in persistence lookback
do
    python3 train_model.py \
        "model.name=$model_name" \
        "model_configs=$model_name" \
        "misc.output_dir=$output_dir" \
        "model.normalization=0" \
        "misc.benchmark=True" \
        "subset=partial"
done

#sklinear

python3 train_model.py \
    "model.name=sklinear" \
    "model_configs=sklinear" \
    "misc.output_dir=$output_dir" \
    "misc.benchmark=True" \
    "model_configs.normalize_method=None" \
    "misc.save_name=sklinear" \
    "subset=partial"

# python3 train_model.py \
#     "model.name=sklinear" \
#     "model_configs=sklinear" \
#     "misc.output_dir=$output_dir" \
#     "misc.benchmark=True" \
#     "model_configs.normalize_method=relative" \
#     "misc.save_name=sklinear_relative"


#Dlinear
for normalization in 1 2 3
do
    python3 train_model.py \
        "model.name=DLinear" \
        "model_configs=DLinear" \
        "misc.output_dir=$output_dir" \
        "model.normalization=$normalization" \
        "training.loss=MMSE" \
        "misc.benchmark=True" \
        "subset=partial"
done

python3 -c "from src.timetensor.visu import plot_expe;plot_expe('$output_dir')"

multipliers="3 1 2"
python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}mean_results.json', multipliers='$multipliers')"
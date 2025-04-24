source .venv/bin/activate

output_dir="../outputs/benchmark/"
rm -rf "$output_dir"
mkdir -p "$output_dir"

for model_name in persistence lookback sklinear
do
    python3 train_model.py \
        "model.name=$model_name" \
        "model_configs=$model_name" \
        "misc.output_dir=$output_dir" \
        "model.normalization=0" \
        "misc.benchmark=True" \
        "training.retrain=True"
done

# for model_name in linear
# do
#     for normalization in 1 2 3
#     do
#         if [ "$normalization" = "1"]; then
#             loss = MSE
#         else
#             loss = NMSE
#         fi
#         python3 train_model.py \
#             "model.name=$model_name" \
#             "model_configs=$model_name" \
#             "misc.output_dir=$output_dir" \
#             "model.normalization=$normalization" \
#             "misc.benchmark=True" \
#             "training.retrain=True" \
#             "training.loss=$loss"
# done
for model_name in linear
do
    for normalization in 1 2 3
    do
        for loss in MSE NMSE
        do
            python3 train_model.py \
                "model.name=$model_name" \
                "model_configs=$model_name" \
                "misc.output_dir=$output_dir" \
                "model.normalization=$normalization" \
                "misc.benchmark=True" \
                "training.retrain=True" \
                "training.loss=$loss" \
                "misc.save_name=${model_name}_normal${normalization}_crit${loss}"
        done
done

python3 losses.py "misc.output_dir=$output_dir"
python3 print_table.py "misc.output_dir=$output_dir"
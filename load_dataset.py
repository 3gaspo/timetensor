import hydra
import logging
import os
from time import perf_counter
import pandas as pd

from src.timetensor.dataset import get_train_loaders, build_dataset, get_dataset_splits, get_sizes
from src.timetensor.utils import set_random_data, load_data
from src.timetensor.visu import plot_named_example, plot_stats, plot_means

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    logger.info("=====Running data script=====")

    #configs
    data_path, dataset_name = cfg.data.path, cfg.data.dataset
    lags, horizon = cfg.task.lags, cfg.task.horizon
    indiv_split, date_splits = cfg.data.indiv_split, cfg.data.date_splits
    batch_size = cfg.training.bs
    for suffix in ["plots/", "examples/"]:
        if not os.path.exists(data_path+suffix):
            os.makedirs(data_path+suffix)
    
    rebuild_pt = False
    reshuffle = False
    new_example = False
    replot = True

    logger.info("Fetched configs")

    #dataset
    if dataset_name == "electricity":
        from src.timetensor.electricity import fetch_data  #adapt path if script in another working directory
    elif "sim" in dataset_name:
        def fetch_data(data_path, raw_format=None, output_format=None):
            values, _, _ = load_data(data_path)
            df = pd.DataFrame(values.squeeze(1).numpy().T)
            return df
    else:
            raise ValueError("Dataset name not recognized")
    if rebuild_pt:
        logger.info("Rebuilding dataset")
        t1 = perf_counter()
        fetcher = lambda path: fetch_data(path, raw_format="csv")
        build_dataset(fetcher, data_path) #saves values, context, datetimes as .pt
        t2 = perf_counter()
        logger.info(f"Build in {(t2-t1)/60:.3f} min")

    #splits
    data_dict = get_dataset_splits(data_path, indiv_split, date_splits, context_by_individuals=True, reshuffle=reshuffle)
    loaders_dict = get_train_loaders(data_dict, batch_size, lags, horizon, by_date=True)
    for k,v in data_dict.items():
        logger.info(f"{k}: {v[0].shape}")
    
    #sizes
    _, _, batch_str = get_sizes(loaders_dict["train"], str_info=True)
    logger.info(batch_str)

    #example
    if new_example:
        logger.info("Setting new example")
        ex_dir = data_path + "examples/" + f"{lags}_{horizon}/"
        set_random_data(data_path, lags, horizon, name="rand")
        plot_named_example(ex_dir, f"rand")
    
    #plots
    if replot:
        plot_dir = data_path + "plots/"
        full_df = fetch_data(data_path, raw_format="csv", output_format="pandas")
        df_dict = {key: loaders_dict[key].dataset.get_df() for key in loaders_dict if key in ["train", "test1", "test2"]}
        
        t1 = perf_counter()
        plot_stats(full_df, plot_dir, name="per_user_stats.pdf", per_user=True, title=f"{dataset_name} user statistics", remove_cte=True)
        plot_stats(df_dict, plot_dir, name="split_stats.pdf", per_user=True, title=f"{dataset_name} splits statistics", remove_cte=True)
        t2 = perf_counter()
        plot_stats(full_df, plot_dir, name="full_input_stats.pdf", per_user=False, lookback=lags, samples=2000, title=f"{dataset_name} input statistics", remove_cte=False)
        t3 = perf_counter()
        t4 = perf_counter()
        plot_stats(full_df, plot_dir, name="input_stats.pdf", per_user=False, lookback=lags, samples=1000, title=f"{dataset_name} input statistics", remove_cte=True)

        df_dict = {key: loaders_dict[key].dataset.get_df() for key in loaders_dict if key in ["train", "test1"]}
        plot_means(df_dict, plot_dir, name="split_means.pdf", per_user=True, title=f"{dataset_name} splits means")
        plot_means(df_dict, plot_dir, name="input_means.pdf", per_user=False, title=f"{dataset_name} input means")

    logger.info('End of script\n')

if __name__ == "__main__":
    run()


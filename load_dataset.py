import hydra
import logging
import os
from time import perf_counter
import pandas as pd
import torch
import numpy as np

from src.timetensor.dataset import get_train_loaders, build_dataset, get_dataset_splits, get_sizes
from src.timetensor.utils import set_random_data, load_data
from src.timetensor.visu import plot_named_example, plot_stats, plot_means, check_cte_windows
from src.timetensor.analysis import *

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
    seed = cfg.misc.seed
    n_clusters = cfg.data.n_clusters
    remove_cte = cfg.data.remove_cte

    for suffix in ["plots/", "examples/"]:
        if not os.path.exists(data_path+suffix):
            os.makedirs(data_path+suffix)
    if seed == "None":
        seed = None
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    rebuild_pt = False
    reshuffle = True
    new_example = True
    replot = True
    do_heterogeneity = True
    logger.info("Fetched configs")

    #dataset
    if dataset_name == "electricity":
        from src.timetensor.electricity import fetch_data  #adapt path if script in another working directory
    elif "sim" in dataset_name:
        def fetch_data(data_path, drop=None):
            values, _, _ = load_data(data_path)
            df = pd.DataFrame(values.squeeze(1).numpy().T)
            return df
    else:
            raise ValueError("Dataset name not recognized")
    if rebuild_pt:
        logger.info("Rebuilding dataset")
        t1 = perf_counter()
        fetcher = lambda path: fetch_data(path, drop=remove_cte)
        build_dataset(fetcher, data_path) #saves values, context, datetimes as .pt
        t2 = perf_counter()
        logger.info(f"Build in {(t2-t1)/60:.3f} min")

    #splits
    data_dict = get_dataset_splits(data_path, indiv_split, date_splits, context_by_individuals=True, reshuffle=reshuffle)
    loaders_dict = get_train_loaders(data_dict, batch_size, lags, horizon, by_date=True, remove_cte=remove_cte)
    for k,v in data_dict.items():
        logger.info(f"{k}: {list(v[0].shape)}")
    
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
    plot_dir = data_path + "plots/"
    if replot:
        full_df = fetch_data(data_path, raw_format="csv", output_format="pandas")
        
        check_cte_windows(full_df, lags, path=plot_dir)

        df_dict = {key: loaders_dict[key].dataset.get_df() for key in loaders_dict if key in ["train", "test1", "test2"]}
        
        plot_stats(full_df, plot_dir, name="per_user_stats.pdf", per_user=True, title=f"{dataset_name} user statistics", remove_cte=True)
        plot_stats(df_dict, plot_dir, name="split_stats.pdf", per_user=True, title=f"{dataset_name} splits statistics", remove_cte=True)
        plot_stats(full_df, plot_dir, name="full_input_stats.pdf", per_user=False, lookback=lags, samples=2000, title=f"{dataset_name} input statistics", remove_cte=False)
        plot_stats(full_df, plot_dir, name="input_stats.pdf", per_user=False, lookback=lags, samples=1000, title=f"{dataset_name} input statistics", remove_cte=True)

        df_dict = {key: loaders_dict[key].dataset.get_df() for key in loaders_dict if key in ["train", "test1"]}
        plot_means(df_dict, plot_dir, name="split_means.pdf", per_user=True, title=f"{dataset_name} splits means")
        plot_means(df_dict, plot_dir, name="input_means.pdf", per_user=False, title=f"{dataset_name} input means")


        #gamma plot
        plot_gamma(full_df, plot_dir, "gammas.pdf", per_user=True, lookback=lags, horizon=horizon)
        plot_gamma(full_df, plot_dir, "input_gammas.pdf", per_user=False, lookback=lags, horizon=horizon, samples=1000)

        #fourier clustering
        logger.info("Fourier clustering")
        fourier_df = fourier(full_df)
        if do_heterogeneity:
            logger.info("Computing heterogeneity plot")
            plot_heterogeneity(fourier_df, path=plot_dir, name="fourier_heterogeneity.pdf")
            logger.info(f"Computing {n_clusters} clusters")
        Z, distances_matrix = init_clusters(fourier_df)
        labels, cluster_indices = get_clusters(Z, n_clusters)
        logger.info(f"Cluster size: {[len(cluster) for cluster in cluster_indices]}")
        
        plot_dendogram(Z, path=plot_dir, name="fourier_dendogram.pdf")
        plot_distances(distances_matrix, path=plot_dir, name="fourier_distances.pdf")
        
        centroids = get_centroids(fourier_df, cluster_indices)
        plot_centroids(centroids, path=plot_dir, name="fourier_centroids.pdf")
        centroids = get_centroids(full_df, cluster_indices)
        plot_centroids(centroids, path=plot_dir, name="fourier_raw_centroids.pdf")
        
        df_dict = get_cluster_dicts(full_df, cluster_indices)
        plot_stats(df_dict, plot_dir, name="fourier_stats.pdf", per_user=True, lookback=lags, title=f"{dataset_name} input statistics", remove_cte=True)
        plot_gamma(df_dict, plot_dir, "fourier_gammas.pdf", per_user=True, lookback=lags, horizon=horizon)

        #gamma clustering
        logger.info("Gamma clustering")
        alphas_df, betas_df = get_gammas(full_df, lags, horizon)
        gamma_df =  pd.concat((alphas_df, betas_df))

        if do_heterogeneity:
            logger.info("Computing heterogeneity plot")
            plot_heterogeneity(gamma_df, path=plot_dir, name="gammas_heterogeneity.pdf")
            logger.info(f"Computing {n_clusters} clusters")
        Z, distances_matrix = init_clusters(gamma_df)
        labels, cluster_indices = get_clusters(Z, n_clusters)
        logger.info(f"Cluster size: {[len(cluster) for cluster in cluster_indices]}")

        plot_dendogram(Z, path=plot_dir, name="gammas_dendogram.pdf")
        plot_distances(distances_matrix, path=plot_dir, name="gammas_distances.pdf")
        
        centroids = get_centroids(gamma_df, cluster_indices)
        plot_centroids(centroids, path=plot_dir, name="gammas_centroids.pdf")
        centroids = get_centroids(full_df, cluster_indices)
        plot_centroids(centroids, path=plot_dir, name="gammas_raw_centroids.pdf")
        plot_gamma(full_df, plot_dir, "gammas.pdf", per_user=True, lookback=lags, horizon=horizon)

        df_dict = get_cluster_dicts(full_df, cluster_indices)
        plot_stats(df_dict, plot_dir, name="gammas_stats.pdf", per_user=True, lookback=lags, title=f"{dataset_name} input statistics", remove_cte=True)
        plot_gamma(df_dict, plot_dir, "clustered_gammas.pdf", per_user=True, lookback=lags, horizon=horizon)


    logger.info('End of script\n')

if __name__ == "__main__":
    run()


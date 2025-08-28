import hydra
import logging
import os
from time import perf_counter
import pandas as pd
import torch
import numpy as np

from src.timetensor.dataset import get_train_loaders, build_dataset, get_dataset_splits, aggregate_loaders_dict, get_sizes
from src.timetensor.synthetic import fetch_data
from src.timetensor.analysis import get_gammas

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
    data_path = "../datasets/sim/"
    mix = cfg.data.mix
    if float(mix)>0:
        data_path = data_path + "mix_" + str(mix) + "/" 
    lags, horizon, batch_size = 100, 20, 10

    indiv_split, date_splits = cfg.data.indiv_split, cfg.data.date_splits    

    seed = cfg.misc.seed
    if seed == "None":
        seed = None
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    rebuild_pt = True
    reshuffle = True
    new_example = True
    replot = True
    do_heterogeneity = True
    logger.info("Fetched configs")

    if rebuild_pt:
        logger.info("Rebuilding dataset")

        fetcher1 = lambda path: fetch_data(path, n1=100, n2=0, r1=mix)
        fetcher2 = lambda path: fetch_data(path, n1=0, n2=100, r2=mix)
       
        for suffix in ["node1/", "node2/", ""]:
            path = data_path + suffix
            for suffix in ["plots/", "examples/"]:
                if not os.path.exists(path+suffix):
                    os.makedirs(path+suffix)
        build_dataset(fetcher1, data_path + "node1/")
        build_dataset(fetcher2, data_path + "node2/")
    
    def fetch_data_(data_path):
        values, _, _ = load_data(data_path)
        df = pd.DataFrame(values.squeeze(1).numpy().T)
        return df

    #splits
    print("-------")
    data_dict1 = get_dataset_splits(data_path + "node1/", indiv_split, date_splits, context_by_individuals=True, reshuffle=reshuffle)
    loaders_dict1 = get_train_loaders(data_dict1, batch_size, lags, horizon, by_date=False)
    for k,v in loaders_dict1.items():
        if k in ["train","test2"]:
            logger.info(f"{k}: {list(v.dataset.shape)}")

    alphas, betas = get_gammas(loaders_dict1["train"].dataset.get_df(), lags, horizon, eps=1e-6)
    logger.info(f"Mean alphas: {np.mean(alphas):.4f}")
    logger.info(f"Mean betas: {np.mean(betas):.4f}")

    print("-------")
    data_dict2 = get_dataset_splits(data_path + "node2/", indiv_split, date_splits, context_by_individuals=True, reshuffle=True)
    loaders_dict2 = get_train_loaders(data_dict2, batch_size, lags, horizon, by_date=False)
    for k,v in loaders_dict2.items():
        if k in ["train","test2"]:
            logger.info(f"{k}: {list(v.dataset.shape)}")
    alphas, betas = get_gammas(loaders_dict2["train"].dataset.get_df(), lags, horizon, eps=1e-6)
    logger.info(f"Mean alphas: {np.mean(alphas):.4f}")
    logger.info(f"Mean betas: {np.mean(betas):.4f}")

    print("-------")
    loaders_dict12 = aggregate_loaders_dict([loaders_dict1, loaders_dict2])
    for k,v in loaders_dict12.items():
        if k in ["train","test2"]:
            logger.info(f"{k}: {list(v.dataset.shape)}")
    alphas, betas = get_gammas(loaders_dict12["train"].dataset.get_df(), lags, horizon, eps=1e-6)
    logger.info(f"Mean alphas: {np.mean(alphas):.4f}")
    logger.info(f"Mean betas: {np.mean(betas):.4f}")
    
    print("-------")
    loaders_dicts = [loaders_dict1, loaders_dict1, loaders_dict12]

    #example
    if new_example:
        for suffix in ["node1/", "node2/"]:
            logger.info(f"Setting new example for {suffix}")
            path = data_path + suffix
            ex_dir = path + "examples/" + f"{lags}_{horizon}/"
            set_random_data(path, lags, horizon, name="rand")
            plot_named_example(ex_dir, f"rand")
    
    #plots
    full_df1 = fetch_data_(data_path + "/node1/")
    full_df2 = fetch_data_(data_path + "/node2/")
    full_df12 = pd.concat((full_df1, full_df2), axis=1)
    full_df12.columns = range(full_df12.shape[1])
    full_dfs = [full_df1, full_df2, full_df12]
    logger.info("Dataframe shapes" + str([df.shape for df in full_dfs]))
    for ci, suffix in enumerate(["node1/", "node2/", ""]):
        plot_dir = data_path + suffix + "plots/"
        full_df = full_dfs[ci]
        loaders_dict = loaders_dicts[ci]
        dataset_name="sim/" + suffix
        n_clusters = 1 if ci in [0,1] else 2
        if replot:
            
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


    #sizes
    logger.info('End of script\n')

if __name__ == "__main__":
    run()


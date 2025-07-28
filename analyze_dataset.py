import hydra
import logging
import os
import pandas as pd

from src.timetensor.analysis import *
from src.timetensor.visu import plot_stats

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    logger.info("=====Running data script=====")

    #configs
    data_path, dataset_name = cfg.data.path, cfg.data.dataset
    lags, horizon = cfg.task.lags, cfg.task.horizon

    for suffix in ["plots/", "examples/"]:
        if not os.path.exists(data_path+suffix):
            os.makedirs(data_path+suffix)

    do_heterogeneity = False

    logger.info("Fetched configs")

    #data
    if dataset_name == "electricity":
        from src.timetensor.electricity import fetch_data  #adapt path if script in another working directory
    else:
            raise ValueError("Dataset name not recognized")
    plot_dir = data_path + "plots/"
    full_df = fetch_data(data_path, raw_format="csv", output_format="pandas")

    #gamma plot
    plot_gamma(full_df, plot_dir, "gammas.pdf", per_user=True, lookback=lags, horizon=horizon)
    plot_gamma(full_df, plot_dir, "input_gammas.pdf", per_user=False, lookback=lags, horizon=horizon, samples=1000)


    #fourier clustering
    logger.info("Fourier clustering")
    fourier_df = fourier(full_df)
    logger.info("Fourier het")
    if do_heterogeneity:
        logger.info("Computing heterogeneity plot")
        plot_heterogeneity(fourier_df, path=plot_dir, name="fourier_heterogeneity.pdf")
    n_clusters = 3
    logger.info(f"Computing {n_clusters} clusters")
    Z, distances_matrix = init_clusters(fourier_df)
    labels, cluster_indices = get_clusters(Z, n_clusters)
    
    logger.info(f"Drawing plots")
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
    n_clusters = 3
    logger.info(f"Computing {n_clusters} clusters")
    Z, distances_matrix = init_clusters(gamma_df)
    labels, cluster_indices = get_clusters(Z, n_clusters)

    logger.info(f"Drawing plots")
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


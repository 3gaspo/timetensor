## This script loads a raw dataset, turns into proper tensors and plots its statistics

import hydra
import logging
import os
from time import perf_counter
import pandas as pd
import torch
import numpy as np

from src.timetensor.dataset import get_sizes, fetch_training_data, set_random_data, fetch_csv, apply_stats
from src.timetensor.visu import plot_named_example, plot_stats, plot_means, plot_clustering
from src.timetensor.analysis import *
from src.timetensor.utils import filter_dict

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    logger.info("=====Running data script=====")

    #configs
    data_path, dataset_name, context_cols = cfg.data.path, cfg.data.dataset, cfg.data.context_cols
    lags, horizon = int(cfg.task.lags), int(cfg.task.horizon)
    seed, verbose = cfg.misc.seed, cfg.misc.verbose

    split_kwargs, subset_kwargs = cfg.data.splits, cfg.data.subsets
    clusters, n_clusters = cfg.data.clustering.clusters, cfg.data.clustering.n_clusters

    #dirs
    for suffix in ["plots/", "examples/"]:
        if not os.path.exists(data_path+suffix):
            os.makedirs(data_path+suffix)
    for suffix in ["gammas/", "stats/"]:
        if not os.path.exists(data_path+"plots/" + suffix):
            os.makedirs(data_path+"plots/" + suffix)
    if clusters is not None:
        for suffix in ["fourier_clusters/", "gamma_clusters/"]:
            if not os.path.exists(data_path+suffix):
                os.makedirs(data_path+suffix)
            if not os.path.exists(data_path+"plots/" + suffix):
                os.makedirs(data_path+"plots/" + suffix)

    rebuild = cfg.load.rebuild
    new_example = cfg.load.example
    replot = cfg.load.replot
    do_heterogeneity = cfg.load.heterogeneity
    recluster = cfg.load.clusters

    if verbose:
        logger.info("Fetched configs")

    #build pytorch dataset
    if rebuild:
        t1 = perf_counter()
        if "synthetic" in dataset_name:
            from src.timetensor.synthetic import build_dataset
            build_dataset(data_path, n1=cfg.data.n1, n2=cfg.data.n2, r1=cfg.data.r1, r2=cfg.data.r2, seed=seed)
        if "saturation" in dataset_name:
            from src.timetensor.saturation import build_dataset
            build_dataset(data_path, seed=seed)
        else:
            from src.timetensor.dataset import build_dataset
            build_dataset(data_path, dataset_name, context_cols, drop_users=split_kwargs.drop_users)
        t2 = perf_counter()
        if verbose:
            logger.info(f"Rebuilt dataset in {(t2-t1)/60:.3f} min")

    #plot and save example
    if new_example:
        ex_dir = data_path + "examples/" + f"{lags}_{horizon}/"
        set_random_data(data_path, lags, horizon, name="rand")
        plot_named_example(ex_dir, f"rand")
        if verbose:
            logger.info("Set new example")

    #constant windows
    df, _, _ = fetch_csv(data_path, dataset_name, context_cols, split_kwargs["drop_users"])
    dates, individuals = df.shape
    _, counts = identify_cte(df, lags)
    if len(counts)>0:
        logger.info("Found constant windows!")
        if len(counts)<=10:    
            logger.info(counts)

            
    #splits
    loaders_dict, stats_dict, nodes_data_dict = fetch_training_data(data_path, split_kwargs, subset_kwargs, cfg.training.bs, lags, horizon, clusters=clusters, seed=seed)
    _, shape_str, batch_str = get_sizes(loaders_dict, str_info=True)
    if verbose:
        logger.info("Fetched dataloaders")
        logger.info(shape_str)
        logger.info(batch_str)
    for key in stats_dict:
        stats_str = "\n".join(f"{k}\t{v:.4f}" for k, v in stats_dict[key].items())
        logger.info(f"{key} stats:\n{stats_str}")


    #plots
    if replot:
        main_plot_dir = data_path + "plots/"
        remove_cte, logs = split_kwargs.remove_train_cte, cfg.load.logs
        df_dict = {key: loader.dataset.get_df() for key, loader in loaders_dict.items()}
        samples = 1000

        #stats
        plot_dir = main_plot_dir + "stats/"
        if individuals>3:
            plot_stats(df, plot_dir, "user_stats.pdf", per_user=True, lookback=lags, title=f"{dataset_name} user statistics", remove_cte=remove_cte, log=logs)
        plot_stats(df, plot_dir, "input_stats.pdf", per_user=False, samples=samples, lookback=lags, title=f"{dataset_name} input statistics", remove_cte=remove_cte, log=logs)
        if "test2" in df_dict:
            plot_stats(filter_dict(df_dict, keys=["test1", "test2"]), plot_dir, "spatial_stats.pdf", per_user=True, lookback=lags, title=f"{dataset_name} spatial splits statistics", remove_cte=remove_cte, log=logs)
            plot_stats(filter_dict(df_dict, keys=["train", "test1"]), plot_dir, "temporal_stats.pdf", per_user=True, lookback=lags, title=f"{dataset_name} temporal splits statistics", remove_cte=remove_cte, log=logs)
        plot_means(filter_dict(df_dict, keys=["train", "test1"]), plot_dir, "input_temporal_means.pdf", per_user=False, samples=samples, lookback=lags, title=f"{dataset_name} input means", log=logs)

        #gamma
        plot_dir = main_plot_dir + "gammas/"
        if individuals>3:
            plot_gamma(df, plot_dir, "gammas.pdf", per_user=True, lookback=lags, horizon=horizon, log=False)
        plot_gamma(df, plot_dir, "input_gammas.pdf", per_user=False, samples=samples, lookback=lags, horizon=horizon, log=False)
        if "test2" in df_dict:
            plot_gamma(filter_dict(df_dict, keys=["test1", "test2"]), plot_dir, name="spatial_gammas.pdf", per_user=True,  lookback=lags, title=f"{dataset_name} spatial splits statistics", remove_cte=remove_cte, log=False)
            plot_gamma(filter_dict(df_dict, keys=["train", "test1"]), plot_dir, name="temporal_gammas.pdf", per_user=True,  lookback=lags, title=f"{dataset_name} temporal splits statistics", remove_cte=remove_cte, log=False)

    if recluster and individuals>3:
        #fourier clustering
        plot_dir = main_plot_dir + "fourier_clusters/"
        logger.info("==Fourier clustering==")
        fourier_df = fourier(df)
        cluster_indices = plot_clustering(df, fourier_df, n_clusters, lags, horizon, "fourier_clusters", plot_dir, do_heterogeneity, remove_cte)
        logger.info(f"Fourier cluster sizes: {[len(cluster) for cluster in cluster_indices]}")
        for k in range(n_clusters):
            torch.save(cluster_indices[k], data_path + "fourier_clusters/" + f"node{k}.pt")
        
        #gamma clustering
        plot_dir = main_plot_dir + "gamma_clusters/"
        logger.info("==Gamma clustering==")
        alphas_df, betas_df = get_gammas(df, lags, horizon)
        gamma_df =  pd.concat((alphas_df, betas_df))
        cluster_indices = plot_clustering(df, gamma_df, n_clusters, lags, horizon, "gamma_clusters", plot_dir, do_heterogeneity, remove_cte)
        logger.info(f"Gamma cluster sizes: {[len(cluster) for cluster in cluster_indices]}")
        for k in range(n_clusters):
            torch.save(cluster_indices[k], data_path + "fourier_clusters/" + f"node{k}.pt")

    logger.info('End of script\n')

if __name__ == "__main__":
    run()


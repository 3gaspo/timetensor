## This script loads a raw dataset, turns into proper tensors and plots its statistics

import hydra
import logging
import os
from time import perf_counter
import pandas as pd
import torch
import numpy as np
import json

from src.timetensor.dataset import get_sizes, fetch_training_data, fetch_dicts, set_random_data
from src.timetensor.utils import filter_dict
from src.timetensor.visu import plot_named_example, plot_stats, plot_means, plot_clustering
from src.timetensor.analysis import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#TODO: bug nanvar sur traffic

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    logger.info("=====Running data script=====")

    #configs
    data_path, dataset_name = cfg.data.path, cfg.data.dataset
    lags, horizon = cfg.task.lags, cfg.task.horizon
    seed, verbose = cfg.misc.seed, cfg.misc.verbose
    if seed == "None":
        seed = None

    remove_cte = cfg.data.remove_cte
    clusters, n_clusters = cfg.data.clusters, cfg.data.n_clusters
    context_cols, drop_users = cfg.data.context_cols, cfg.data.drop_users

    for suffix in ["plots/", "examples/", "fourier_clusters/", "gamma_clusters/"]:
        if not os.path.exists(data_path+suffix):
            os.makedirs(data_path+suffix)
    for suffix in ["fourier_clusters/", "gamma_clusters/", "gammas/", "stats/"]:
        if not os.path.exists(data_path+"plots/" + suffix):
            os.makedirs(data_path+"plots/" + suffix)

    rebuild_pt = True
    new_example = True
    replot = True
    do_heterogeneity = True
    logger.info("Fetched configs")

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    #build dataset values.pt from dataset_name.csv
    if rebuild_pt:
        t1 = perf_counter()
        # if dataset_name == "electricity":
        #     from src.timetensor.electricity import build_dataset
        #     build_dataset(data_path, raw_format=cfg.data.type, output_format="csv", drop=cfg.data.drop_users)
        if "synthetic" in dataset_name:
            from src.timetensor.synthetic import build_dataset
            build_dataset(data_path, n1=cfg.data.n1, n2=cfg.data.n2, r1=cfg.data.r1, r2=cfg.data.r2, output_format="csv")
        else:
            from src.timetensor.dataset import build_dataset
            build_dataset(data_path, dataset_name, context_cols=cfg.data.context_cols, raw_format="csv", output_format="csv")
        t2 = perf_counter()
        if verbose:
            logger.info(f"Rebuilt dataset in {(t2-t1)/60:.3f} min")

    # if dataset_name == "electricity":
    #     from src.timetensor.electricity import fetch_csv
    # else:
    #     from src.timetensor.dataset import fetch_csv
    from src.timetensor.dataset import fetch_csv

    #plot and save example
    if new_example:
        ex_dir = data_path + "examples/" + f"{lags}_{horizon}/"
        set_random_data(data_path, lags, horizon, name="rand")
        plot_named_example(ex_dir, f"rand")
        if logger is not None:
            logger.info("Set new example")

    #look for outlier windows
    df, context, datetimes = fetch_csv(data_path, dataset_name, context_cols=context_cols, drop=drop_users)
    cte_mask = identify_cte(df, lags, path=data_path, logger=logger)

    #splits
    loaders_dict = fetch_training_data(data_path, cfg.data.indiv_split, cfg.data.date_splits, cfg.data.subsets, cfg.training.bs, lags, horizon, by_date=True, context_by_individuals=cfg.data.context_by_individuals, reshuffle=cfg.data.reshuffle, remove_cte=remove_cte, clusters=clusters, seed=seed)
    _, shape_str, batch_str = get_sizes(loaders_dict, str_info=True)
    if verbose:
        logger.info("Fetched dataloaders")
        logger.info(shape_str)
        logger.info(batch_str)

    #stats
    df, context, datetimes = fetch_csv(data_path, dataset_name, context_cols=context_cols, drop=drop_users)
    loaders_dict, df_dict = fetch_dicts(data_path, cfg, remove_cte=remove_cte, clusters=clusters, seed=seed)
    stats_dict = get_dataset_stats(df, df_dict, lags, horizon, remove_cte, logger, data_path, "raw_stats.json")

    #normalized stats
    df, context, datetimes = fetch_csv(data_path, dataset_name, context_cols=context_cols, drop=drop_users)
    loaders_dict, df_dict = fetch_dicts(data_path, cfg, remove_cte=remove_cte, clusters=clusters, seed=seed, stats_dict=stats_dict)
    df = (df - stats_dict["train"]["mean"]) / (stats_dict["train"]["std"] + 1e-6)
    stats_dict = get_dataset_stats(df, df_dict, lags, horizon, remove_cte, save_path=data_path, save_name="normal_stats.json")

    #per cluster stats
    if clusters is not None:
        if not os.path.exists(data_path+clusters+"stats/"):
            os.makedirs(data_path+clusters+"stats/")
        df, context, datetimes = fetch_csv(data_path, dataset_name, context_cols=context_cols, drop=drop_users)
        loaders_dict, df_dicts = fetch_dicts(data_path, cfg, remove_cte=remove_cte, clusters=clusters, seed=seed, aggregate=False)
        for i, df_dict in enumerate(df_dicts.values()):
            cluster_df = df[list(torch.load(data_path+clusters+f"node{i}.pt", weights_only=False))]
            stats_dict = get_dataset_stats(cluster_df, df_dict, lags, horizon, remove_cte, save_path=data_path+clusters+"stats/", save_name=f"node{i}_raw_stats.json")

    #plots
    main_plot_dir = data_path + "plots/"
    df, context, datetimes = fetch_csv(data_path, dataset_name, context_cols=context_cols, drop=drop_users)
    loaders_dict, df_dict = fetch_dicts(data_path, cfg, remove_cte=remove_cte, clusters=clusters, seed=seed)
    if replot:    
        #stats
        plot_dir = main_plot_dir + "stats/"
        plot_stats(df, plot_dir, name="user_stats.pdf", per_user=True, title=f"{dataset_name} user statistics", remove_cte=remove_cte, log=True)
        plot_stats(df, plot_dir, name="input_stats.pdf", per_user=False, lookback=lags, samples=1000, title=f"{dataset_name} input statistics", remove_cte=remove_cte, log=True)
        plot_stats(filter_dict(df_dict, keys=["train", "test1"]), plot_dir, name="temporal_stats.pdf", per_user=True, title=f"{dataset_name} temporal splits statistics", remove_cte=remove_cte, log=True)
        if "test2" in df_dict:
            plot_stats(filter_dict(df_dict, keys=["test1", "test2"]), plot_dir, name="spatial_stats.pdf", per_user=True, title=f"{dataset_name} spatial splits statistics", remove_cte=remove_cte, log=True)
        plot_means(filter_dict(df_dict, keys=["train", "test1"]), plot_dir, name="input_temporal_means.pdf", per_user=False, title=f"{dataset_name} input means", log=True)

        #gamma
        plot_dir = main_plot_dir + "gammas/"
        plot_gamma(df, plot_dir, "gammas.pdf", per_user=True, lookback=lags, horizon=horizon, log=False)
        plot_gamma(df, plot_dir, "input_gammas.pdf", per_user=False, lookback=lags, horizon=horizon, samples=1000, log=False)
        plot_gamma(filter_dict(df_dict, keys=["train", "test1"]), plot_dir, name="temporal_gammas.pdf", per_user=True, title=f"{dataset_name} temporal splits statistics", remove_cte=remove_cte, log=False)
        if "test2" in df_dict:
            plot_gamma(filter_dict(df_dict, keys=["test1", "test2"]), plot_dir, name="spatial_gammas.pdf", per_user=True, title=f"{dataset_name} spatial splits statistics", remove_cte=remove_cte, log=False)

        #fourier clustering
        plot_dir = main_plot_dir + "fourier_clusters/"
        logger.info("==Fourier clustering==")
        fourier_df = fourier(df)
        plot_clustering(df, fourier_df, n_clusters, lags, horizon, "fourier_clusters", data_path, plot_dir, do_heterogeneity, logger, remove_cte)

        #gamma clustering
        plot_dir = main_plot_dir + "gamma_clusters/"
        logger.info("==Gamma clustering==")
        alphas_df, betas_df = get_gammas(df, lags, horizon)
        gamma_df =  pd.concat((alphas_df, betas_df))
        plot_clustering(df, gamma_df, n_clusters, lags, horizon, "gamma_clusters", data_path, plot_dir, do_heterogeneity, logger, remove_cte)

    #per cluster stats
    for clusters in ["fourier_clusters/", "gamma_clusters/"]:
        if not os.path.exists(data_path+clusters+"stats/"):
            os.makedirs(data_path+clusters+"stats/")       
        df, context, datetimes = fetch_csv(data_path, dataset_name, context_cols=context_cols, drop=drop_users)
        loaders_dicts, df_dicts = fetch_dicts(data_path, cfg, remove_cte=remove_cte, clusters=clusters, seed=seed, stats_dict=stats_dict, aggregate=False)
        for i, df_dict in enumerate(df_dicts.values()):
            cluster_df = df[list(torch.load(data_path+clusters+f"node{i}.pt", weights_only=False))]
            stats_dict = get_dataset_stats(cluster_df, df_dict, lags, horizon, remove_cte, save_path=data_path+clusters+"stats/", save_name=f"node{i}_raw_stats.json")

    logger.info('End of script\n')

if __name__ == "__main__":
    run()


## This script loads a raw dataset, turns into proper tensors and plots its statistics

import hydra
import logging
import os
from time import perf_counter
import pandas as pd
import torch
import copy

from src.timetensor.dataset import get_sizes, fetch_training_data, set_random_data, fetch_csv
from src.timetensor.visu import plot_named_example, plot_stats, plot_means, plot_clustering, plot_points
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

    split_kwargs = cfg.data.splits
    clusters, n_clusters = cfg.data.clustering.clusters, cfg.data.clustering.n_clusters
    remove_cte = split_kwargs.remove_train_cte

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
    if cfg.load.clusters:        
        for suffix in ["fourier_clusters/", "gamma_clusters/"]:
            if not os.path.exists(data_path+"plots/" + suffix):
                os.makedirs(data_path+"plots/" + suffix)

    rebuild = cfg.load.rebuild
    new_example = cfg.load.example
    replot = cfg.load.replot
    do_heterogeneity = cfg.load.heterogeneity
    recluster = cfg.load.clusters
    do_shapes = cfg.load.shapes
    do_windows = cfg.load.windows
    do_distances = cfg.load.distances

    do_shapes = False
    do_windows = False
    do_distances = False

    if verbose:
        logger.info("Fetched configs")
        logger.info(f"Loading {dataset_name}")
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

    df, _, _ = fetch_csv(data_path, dataset_name, context_cols, split_kwargs["drop_users"])
    dates, individuals = df.shape
    
    #constant windows
    if do_windows:
        _, counts = identify_cte(df, lags)
        if len(counts)>0:
            logger.info(f"Found {len(counts)} constant windows!")
            if len(counts)<=10:
                logger.info(counts)

            
    #splits
    if do_shapes:
        loaders_dict, stats_dict, nodes_data_dict = fetch_training_data(data_path, split_kwargs, cfg.data.subsets, cfg.training.bs, lags, horizon, clusters=clusters, seed=seed)
        _, shape_str, batch_str = get_sizes(loaders_dict, str_info=True)
        if verbose:
            logger.info("Fetched dataloaders")
            logger.info(shape_str)
            logger.info(batch_str)
        for key in stats_dict:
            stats_str = "\n".join(f"{k}\t{v:.4f}" for k, v in stats_dict[key].items())
            logger.info(f"{key} stats:\n{stats_str}")

    #distances
    split_kwargs_ = copy.deepcopy(split_kwargs)
    split_kwargs_["idx_mode"]='random'
    batch_size_ = 1
    point_loaders_dict, stats_dict, nodes_data_dict = fetch_training_data(data_path, split_kwargs_, cfg.data.subsets, batch_size_, lags, horizon, clusters=clusters, seed=seed)
    if do_distances:
        d_space, d_temps, d_modulations = get_spatial_distance(df), get_temporal_distance(df, int(len(df) * 0.6), int(len(df) * 0.8)), get_modulation_distance(df, lags=lags, horizon=horizon)
        logger.info(f"Spatial distance: {d_space:.3f}, Temporal distance: {d_temps:.3f}, Modulations distance: {d_modulations:.3f}")
    temp_df, spatial_df = analyze_discrepency(point_loaders_dict, split_kwargs, stats_dict, samples=3000, seed=seed)
    logger.info(temp_df.applymap('{:,.2f}'.format))
    logger.info(spatial_df.applymap('{:,.2f}'.format))


    #plots
    main_plot_dir = data_path + "plots/"
    if replot:
        logs = cfg.load.logs
        df_dict = {key: loader.dataset.get_df() for key, loader in point_loaders_dict.items()}
        samples = 1000

        #stats
        plot_dir = main_plot_dir + "stats/"
        if individuals>10:
            plot_stats(df, plot_dir, "user_stats.pdf", per_user=True, lookback=lags, horizon=horizon, title=f"{dataset_name} user statistics", remove_cte=remove_cte, log=logs)
        plot_stats(df, plot_dir, "input_stats.pdf", per_user=False, samples=samples, lookback=lags, horizon=horizon, title=f"{dataset_name} input statistics", remove_cte=remove_cte, log=logs)
        if "test2" in df_dict:
            plot_stats(filter_dict(df_dict, keys=["train", "valid2"]), plot_dir, "input_spatial_stats.pdf", per_user=False, lookback=lags, horizon=horizon, title=f"{dataset_name} spatial splits statistics", remove_cte=remove_cte, log=logs)
        plot_stats(filter_dict(df_dict, keys=["train", "test1"]), plot_dir, "input_temporal_stats.pdf", per_user=False, lookback=lags, horizon=horizon, title=f"{dataset_name} temporal splits statistics", remove_cte=remove_cte, log=logs)
        
        plot_points(filter_dict(point_loaders_dict, keys=["train", "test1"]), plot_dir, "points_temporal.pdf", samples=samples, title=f"{dataset_name} input means", log=True)
        plot_points(filter_dict(point_loaders_dict, keys=["train", "valid2"]), plot_dir, "points_spatial.pdf", samples=samples, title=f"{dataset_name} input means", log=True)
        plot_points(filter_dict(point_loaders_dict, keys=["train", "test1"]), plot_dir, "norm_points_temporal.pdf", samples=samples, title=f"{dataset_name} input means", log=True, normal=True)
        plot_points(filter_dict(point_loaders_dict, keys=["train", "valid2"]), plot_dir, "norm_points_spatial.pdf", samples=samples, title=f"{dataset_name} input means", log=True, normal=True)

        #gamma
        plot_dir = main_plot_dir + "gammas/"
        if individuals>10:
            plot_gamma(df, plot_dir, "gammas.pdf", per_user=True, lookback=lags, horizon=horizon, log=False)
        plot_gamma(df, plot_dir, "input_gammas.pdf", per_user=False, samples=samples, lookback=lags, horizon=horizon, log=False)
        if "test2" in df_dict:
            plot_gamma(filter_dict(df_dict, keys=["train", "test2"]), plot_dir, name="input_spatial_gammas.pdf", per_user=False,  lookback=lags, title=f"{dataset_name} spatial splits statistics", remove_cte=remove_cte, log=False)
        plot_gamma(filter_dict(df_dict, keys=["train", "test1"]), plot_dir, name="input_temporal_gammas.pdf", per_user=False,  lookback=lags, title=f"{dataset_name} temporal splits statistics", remove_cte=remove_cte, log=False)
        
        plot_betas(filter_dict(df_dict, keys=["train", "test1"]), plot_dir, "input_deltas_temporal.pdf", per_user=False, samples=samples, lookback=lags, horizon=horizon, title=f"{dataset_name} input deltas", log=logs)
        plot_betas(filter_dict(df_dict, keys=["train", "valid1"]), plot_dir, "input_deltas_indiv.pdf", per_user=False, samples=samples, lookback=lags, horizon=horizon, title=f"{dataset_name} input deltas", log=logs)


    if recluster and individuals>10:
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


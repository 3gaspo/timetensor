## Adding neighboring past windows of each user as context

import hydra
import logging
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from time import perf_counter

from sklearn.neighbors import NearestNeighbors

from src.timetensor.dataset import fetch_csv
from src.timetensor.models import load_model
from src.timetensor.pipeline import Loss
from src.timetensor.utils import get_dirs, set_seed, get_normal_stats, save_results

from src.timetensor.analysis import calculate_distances
from src.timetensor.utils import symlog

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    logger.info("=====Running cross learning clusters script=====")

    #configs
    data_path = cfg.data.path
    lags, horizon = int(cfg.task.lags), int(cfg.task.horizon)

    criterion = Loss(nn.MSELoss(reduction="none"), mode="instance")

    model_name, norm_name = cfg.model.name, cfg.normalization.name
    if norm_name == "None":
        norm_name = None
    kwargs = {**(cfg.normalization.configs or {}), **(cfg.model.configs or {})}

    verbose, seed = cfg.misc.verbose, cfg.misc.seed

    output_dir, save_name = cfg.misc.output_dir, cfg.misc.save_name 
    save_name, save_dir = get_dirs(output_dir, save_name, model_name, norm_name)

    if verbose:
        logger.info(f"Fetched main configs, save directory : {save_dir}")
        logger.info(f"Model {model_name}, norm {norm_name}, kwargs {kwargs}")

    device = torch.device("cuda" if cfg.misc.device=="gpu" and torch.cuda.is_available() else "cpu")
    set_seed(seed)

    #data
    data, _, _ = fetch_csv(data_path, cfg.data.dataset, drop_users=cfg.data.splits.drop_users, aggr=cfg.data.aggregation)
    data = data.reset_index(drop=True)
    if verbose:
        logger.info("Fetched data csv")
        logger.info(f"Shape: {data.shape}")
            
    #model
    model = load_model(model_name, (lags, 1, horizon), norm_name, cfg.training.init, device.type=="cpu", **kwargs)
    model.eval()

    #evals
    all_indiv = list(range(data.shape[1]))
    individuals = len(all_indiv)
    bs = cfg.training.bs 
    is_context = (bs > 1)

    dates = data.shape[0]
    date_split = 1.0  #train / total dates ratio
    if cfg.data.splits.date_splits:
        date_split = float(cfg.data.splits.date_splits.split(";")[0])
    split_date_idx = int(date_split * dates)
    
    eval_stride = int(cfg.data.sampling.eval_stride)
    max_start = dates - (lags + horizon)
    train_strides_dates = np.array(range(0, min(split_date_idx, max_start+1), eval_stride))
    eval_strided_dates = np.array(range(split_date_idx, max_start + 1, eval_stride))
    
    logger.info(f"Stride dates: {len(train_strides_dates)} (train) {len(eval_strided_dates)} (eval)")
    # logger.info(f"Total eval loops: {len(eval_strided_dates) * individuals}")

    indiv_losses = {indiv: [] for indiv in range(individuals)}
    per_user_losses, stds_per_user_losses = [], []

    t1 = perf_counter()
    for indiv in all_indiv:
        indiv_data = data.iloc[:, indiv].values

        if is_context and (bs <= len(train_strides_dates)):
            # eval_windows = [indiv_data[t:t+lags] for t in eval_strided_dates]
            # train_windows = [indiv_data[t:t+lags] for t in train_strides_dates]
            train_windows = indiv_data[train_strides_dates[:, None] + np.arange(lags)]
            eval_windows = indiv_data[eval_strided_dates[:, None] + np.arange(lags)]

            nn_model = NearestNeighbors(n_neighbors=bs-1, metric=cfg.extra.distance)
            nn_model.fit(train_windows)
            distances, indices = nn_model.kneighbors(eval_windows)
            # indiv_matrix = np.array([indiv_data[t:t+lags] for t in train_strides_dates]) # (eval_strided_dates, lags)
            # D = calculate_distances(indiv_matrix.T, metric="euclidean", matrix=True) # (eval_strided_dates, eval_strided_dates)

        for i, stride_date_idx in enumerate(range(len(eval_strided_dates))):
            t = eval_strided_dates[stride_date_idx]
            x = indiv_data[t: t+lags]
            y = indiv_data[t+lags: t+lags+horizon]
            x = torch.tensor(x).unsqueeze(0).unsqueeze(0)
            y = torch.tensor(y).unsqueeze(0).unsqueeze(0)

            xc = []
            if is_context:
                if bs > len(train_strides_dates):
                    context_rows = list(range(len(train_strides_dates)))
                else:
                    # sorted_rows = np.argsort(D[stride_date_idx])
                    # context_rows = list(sorted_rows[1:bs])
                    context_rows = indices[i]

                for s in context_rows:
                    xc.append(torch.tensor(indiv_data[train_strides_dates[s]:train_strides_dates[s]+lags+horizon]))
                if xc:
                    xc = torch.stack(xc).unsqueeze(1)

            c = xc if (is_context and len(xc) > 0) else None
            mean, std = get_normal_stats(x)
            pred = model(x, c)
            loss = criterion(pred, y, mean, std)
            indiv_losses[indiv].append(loss[0].mean().item())
                
    for indiv in all_indiv:
        indiv_loss = indiv_losses[indiv]
        mean = symlog(np.mean(indiv_loss))
        std = symlog(np.std(indiv_loss))
        per_user_losses.append(mean)
        stds_per_user_losses.append(std)

    total_means = np.mean(per_user_losses)
    w10_means = np.mean(np.partition(per_user_losses, int(len(per_user_losses)*0.9))[int(len(per_user_losses)*0.9):])

    t2 = perf_counter()
    delta_t = (t2-t1)/60

    save_results(total_means, output_dir, f"mean_results.json", save_name, f"nMSE")
    save_results(w10_means, output_dir, f"mean_results.json", save_name, f"w10 nMSE")
    save_results(delta_t, output_dir, f"mean_results.json", save_name, f"eval time (min)")

    logger.info('End of script\n')

if __name__ == "__main__":
    run()



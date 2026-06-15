## Adding both neighboring past windows of user and neighbors, for each user as context

import hydra
import logging
import torch
import numpy as np
import torch.nn as nn
from time import perf_counter

from sklearn.neighbors import NearestNeighbors

from src.timetensor.dataset import fetch_csv
from src.timetensor.models import load_model
from src.timetensor.pipeline import Loss
from src.timetensor.utils import get_dirs, set_seed, get_normal_stats, save_results

from src.timetensor.analysis import get_fourier_df
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
    date_split = float(cfg.data.splits.date_splits.split(";")[0])
    split_date_idx = int(date_split * dates)
    eval_stride = int(cfg.data.sampling.eval_stride)
    max_start = dates - (lags + horizon)
    eval_strided_dates = np.array(range(split_date_idx, max_start + 1, eval_stride))
    train_strides_dates = np.array(range(0, min(split_date_idx, max_start+1), eval_stride))

    logger.info(f"Stride dates: {len(train_strides_dates)} (train) {len(eval_strided_dates)} (eval)")

    indiv_losses = {indiv: [] for indiv in range(individuals)}
    per_user_losses, stds_per_user_losses = [], []

    t1 = perf_counter()

    if is_context: # data (dates, individuals)
        distance_kws = cfg.extra.distance.split("_")
        if len(distance_kws) == 2:
            distance_space, distance_metric = distance_kws[0], distance_kws[1]
        else:
            distance_space, distance_metric = "raw", distance_kws[0]

        train_feature_idxs = train_strides_dates[:, None] + np.arange(lags) # (len(train_strides_dates), lags)
        train_value_idxs = train_strides_dates[:, None] + np.arange(lags + horizon)
        X_values = data.values[train_value_idxs].transpose(2, 0, 1).reshape(-1, lags + horizon) # (len(train_strides_dates), lags+horizon)
        if distance_space == "fourier":
            fourier_data = get_fourier_df(data)
            X_features = fourier_data.values[train_feature_idxs].transpose(2, 0, 1).reshape(-1, lags) # (len(train_strides_dates), lags)
        elif distance_space == "chronos":
            X_ = data.values[train_feature_idxs].transpose(2, 0, 1).reshape(-1, lags)
            X_features = model.representation(torch.tensor(X_).unsqueeze(1)).flatten(start_dim=1).numpy()
        else:
            X_features = data.values[train_feature_idxs].transpose(2, 0, 1).reshape(-1, lags)
        
        nn_model = NearestNeighbors(n_neighbors=min(bs-1, individuals), metric=distance_metric)    
        nn_model.fit(X_features)

    # head = nn.Linear(3*horizon, horizon)
    tau = 1e-3
    alpha = 1e-3
    for indiv in all_indiv:
        indiv_values = data.iloc[:, indiv].values
        if is_context:
            if distance_space == "fourier":
                indiv_features = fourier_data.values[:, indiv][eval_strided_dates[:, None] + np.arange(lags)]
            elif distance_space == "chronos":
                indiv_ = indiv_values[eval_strided_dates[:, None] + np.arange(lags)]
                indiv_features = model.representation(torch.tensor(indiv_).unsqueeze(1)).flatten(start_dim=1).numpy()
            else:
                indiv_features = indiv_values[eval_strided_dates[:, None] + np.arange(lags)]
            distances, indices = nn_model.kneighbors(indiv_features)
        
        for i, t in enumerate(eval_strided_dates):
            x, y = indiv_values[t : t+lags], indiv_values[t+lags : t+lags+horizon]
            x, y = torch.tensor(x).unsqueeze(0).unsqueeze(0), torch.tensor(y).unsqueeze(0).unsqueeze(0) # x: (1, 1, L)
            
            xc = None
            if is_context:
                xc = X_values[indices[i]]
                xc = torch.tensor(xc, dtype=torch.float32).unsqueeze(1)

            mean, std = get_normal_stats(x)
            pred = model(x, None)
            if xc is not None:
                yc = xc[:, :, lags: lags+horizon]
                pred_c = model(xc[:, :, :lags], None)
                rc = pred_c - yc
                w_c = torch.exp(-torch.tensor(distances[i], dtype=yc.dtype) / tau)
                w_c = w_c / (w_c.sum() + 1e-8)
                corr = (w_c[:, None, None] * rc).sum(dim=0, keepdim=True)   # (1, 1, H)
                pred = pred + alpha*corr
                
                #feat = torch.cat([pred, yc, pred_c], dim=-1)
                #pred = head(feat)
            loss = criterion(pred, y, mean, std) # (bs, dim, H)
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



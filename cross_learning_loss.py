import hydra
import logging
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from time import perf_counter

from src.timetensor.dataset import fetch_csv
from src.timetensor.models import load_model
from src.timetensor.pipeline import Loss
from src.timetensor.utils import get_dirs, set_seed, get_normal_stats, save_results

from src.timetensor.utils import symlog

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    logger.info("=====Running eval script=====")

    #configs
    data_path = cfg.data.path
    lags, horizon = int(cfg.task.lags), int(cfg.task.horizon)

    criterion = Loss(nn.MSELoss(reduction="none"), mode="instance")
    bs = cfg.training.bs

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
    data, _, _ = fetch_csv(data_path, cfg.data.dataset, drop_users=cfg.data.splits.drop_users)
    data = data.reset_index(drop=True)
    if verbose:
        logger.info("Fetched data csv")
        logger.info(f"Shape: {data.shape}")

    #model
    model = load_model(model_name, (lags, 1, horizon), norm_name, cfg.training.init, device.type=="cpu", **kwargs)
    model.eval()
    
    #user eval
    all_indiv = list(range(data.shape[1]))
    max_dates = data.shape[0] - (lags+horizon)
    stride = cfg.data.sampling.train_stride
    strided_dates = (max_dates - 1) // stride + 1
    num_full_batches = len(all_indiv) // bs
    num_runs = cfg.training.eval_runs
    assert bs > 1
    alpha = 0.5
    
    logger.info(f"Stride dates: {strided_dates}")
    logger.info(f"Total loops: {strided_dates * num_runs * num_full_batches}")

    indiv_losses = {indiv: [] for indiv in range(len(all_indiv))}
    per_user_losses, stds_per_user_losses = [], []

    errors = np.zeros((len(all_indiv), len(all_indiv)))
    scores = np.zeros((len(all_indiv), len(all_indiv)))

    t1 = perf_counter()
    for t in range(strided_dates):
        date_idx = t * stride
        for _ in range(num_runs):

            indivs = np.random.choice(all_indiv, bs)
            x, y = data.iloc[date_idx : date_idx+lags, indivs], data.iloc[date_idx+lags : date_idx+lags+horizon, indivs]

            x_batch, y_batch = torch.tensor(x.values).transpose(1,0).unsqueeze(1), torch.tensor(y.values).transpose(1,0).unsqueeze(1) # x: (bs, 1, L)
            c_batch = None
            
            mean, std = get_normal_stats(x_batch)
            pred_batch = model(x_batch, c_batch)
            loss = criterion(pred_batch, y_batch, mean, std).mean(dim=(1,2)) # (bs)
            
            for i, indiv in enumerate(indivs):
                for j, neighbor in enumerate(indivs):
                    if j!=i:
                        errors[indiv][neighbor] = (1-alpha) * errors[indiv][neighbor] + alpha * loss[i].item()


    for indiv in range(len(all_indiv)):
        for neighbor in range(len(all_indiv)):
            if indiv == neighbor:
                scores[indiv][neighbor] = np.inf
            else:
                scores[indiv][neighbor] = np.exp(- errors[indiv][neighbor] / np.sum(errors[indiv]))


    for t in range(strided_dates):
        date_idx = t * stride

        for indiv in range(len(all_indiv)):

            sorted_indices = np.argsort(scores[indiv])
            context_indivs = list(sorted_indices[1:bs])

            x, y = data.iloc[date_idx : date_idx+lags, indiv], data.iloc[date_idx+lags : date_idx+lags+horizon, indiv]  # x: (L)
            xc, yc = data.iloc[date_idx : date_idx+lags, context_indivs], data.iloc[date_idx+lags : date_idx+lags+horizon, context_indivs]
            
            x_batch, y_batch = torch.tensor(x.values).unsqueeze(0).unsqueeze(0), torch.tensor(y.values).unsqueeze(0).unsqueeze(0) # x: (1, 1, L)
            c_batch, _ = torch.tensor(xc.values).transpose(1,0).unsqueeze(1), torch.tensor(yc.values).transpose(1,0).unsqueeze(1) # c: (bs-1, 1, L)
            
            mean, std = get_normal_stats(x_batch)
            pred_batch = model(x_batch, c_batch)
            loss = criterion(pred_batch, y_batch, mean, std) # (1, dim, H)
            
            indiv_losses[indiv].append(loss[0].mean().item())     

    for indiv in range(len(all_indiv)):
        indiv_loss = indiv_losses[indiv]
        if len(indiv_loss) > 0:
            per_user_losses.append(symlog(np.mean(indiv_loss)))
            stds_per_user_losses.append(symlog(np.std(indiv_loss)))
        else:
            per_user_losses.append(np.nan) 
            stds_per_user_losses.append(np.nan)

    total_means = np.mean(per_user_losses)
    w10_means = np.mean(np.partition(per_user_losses, int(len(per_user_losses)*0.9))[int(len(per_user_losses)*0.9):])
    
    t2 = perf_counter()
    delta_t = (t2-t1)/60
    
    save_results(total_means, save_dir, f"mean_results.json", save_name, f"nMSE")
    save_results(w10_means, save_dir, f"mean_results.json", save_name, f"w10 nMSE")
    save_results(delta_t, save_dir, f"mean_results.json", save_name, f"compute (min)")

    stats_df = pd.DataFrame({
        "log(mean_error)": per_user_losses,
        "log(std_error)": stds_per_user_losses}).dropna()

    plt.figure(figsize=(10, 7))
    g = sns.jointplot(
        data=stats_df,
        x="log(mean_error)",
        y="log(std_error)",
        kind='scatter',
        palette='Set1',
    )
    plt.suptitle(f"Per-user nMSE of {save_name} (mean:{total_means:.4f}, W10:{w10_means:.4f})")
    plt.tight_layout()
    plt.savefig(save_dir+ "plots/" + f"{bs}_user_errors.pdf")
    plt.close()
    
    logger.info('End of script\n')

if __name__ == "__main__":
    run()



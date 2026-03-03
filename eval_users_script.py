import hydra
import logging
import torch
from time import perf_counter

from src.timetensor.dataset import fetch_csv
from src.timetensor.models import load_model
from src.timetensor.pipeline import get_losses
from src.timetensor.utils import get_dirs, set_seed, save_results, get_normal_stats

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.timetensor.utils import symlog

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    logger.info("=====Running eval users script=====")

    #configs
    data_path = cfg.data.path
    lags, horizon = int(cfg.task.lags), int(cfg.task.horizon)

    criterion_name = cfg.training.loss
    criterion, eval_losses = get_losses(criterion_name, complete_evaluation=cfg.training.complete_evaluation)

    model_name, norm_name = cfg.model.name, cfg.normalization.name
    if norm_name == "None":
        norm_name = None
    kwargs = {**(cfg.normalization.configs or {}), **(cfg.model.configs or {})}

    verbose, seed = cfg.misc.verbose, cfg.misc.seed

    output_dir, save_name = cfg.misc.output_dir, cfg.misc.save_name
    save_name, save_dir = get_dirs(output_dir, save_name, model_name, norm_name, criterion_name, cfg.data.subsets)

    if verbose:
        logger.info(f"Fetched main configs, save directory : {save_dir}")
        logger.info(f"Model {model_name}, norm {norm_name}, criterion {criterion_name}, kwargs {kwargs}")

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

    dates = data.shape[0]
    date_split = 1.0  #train / total dates ratio
    if cfg.data.splits.date_splits:
        date_split = float(cfg.data.splits.date_splits.split(";")[0])
    split_date_idx = int(date_split * dates)
    
    eval_stride = int(cfg.data.sampling.eval_stride)
    max_start = dates - (lags + horizon)
    eval_strided_dates = list(range(split_date_idx, max_start + 1, eval_stride))

    logger.info(f"Stride dates: {len(eval_strided_dates)} (eval)")
    logger.info(f"Total eval loops: {len(eval_strided_dates) * individuals}")

    indiv_losses = {indiv: [] for indiv in range(individuals)}
    per_user_losses, stds_per_user_losses = [], []

    t1 = perf_counter()
    for indiv in all_indiv:
        for stride_date_idx in range(len(eval_strided_dates)):
            t = eval_strided_dates[stride_date_idx]
            x = data.iloc[t: t+lags, indiv]
            y = data.iloc[t+lags: t+lags+horizon, indiv]
            x = torch.tensor(x.values).unsqueeze(0).unsqueeze(0)
            y = torch.tensor(y.values).unsqueeze(0).unsqueeze(0)
            c_batch = None

            mean, std = get_normal_stats(x)
            pred = model(x, c_batch)
            loss = criterion(pred, y, mean, std)
            indiv_losses[indiv].append(loss.item())
                
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
    plt.suptitle(
        f"Per-user nMSE of {save_name} (mean:{total_means:.4f}, W10:{w10_means:.4f})",
        fontsize=20)   
    plt.tight_layout()
    plt.savefig(save_dir+ "plots/" + f"{bs}_user_errors.pdf")
    plt.close()

    logger.info('End of script\n')

if __name__ == "__main__":
    run()



## Tests identity as covariate (horizon included)

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
from src.timetensor.utils import get_dirs, set_seed, get_normal_stats, save_results, symlog
from src.timetensor.visu import plot_horizon_errors, plot_errors, plot_pred

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
    
    #user eval
    all_indiv = list(range(data.shape[1]))
    individuals = len(all_indiv)
    max_dates = data.shape[0] - (lags+horizon)
    stride = cfg.data.sampling.train_stride
    strided_dates = (max_dates - 1) // stride + 1

    logger.info(f"Stride dates: {strided_dates}")
    logger.info(f"Total loops: {strided_dates * individuals * cfg.training.eval_runs}")

    indiv_losses = {indiv: [] for indiv in range(individuals)}
    all_losses = []
    per_user_losses, stds_per_user_losses = [], []
    horizon_losses = np.zeros(horizon)

    bs = cfg.training.bs 
    use_context = cfg.data.sampling.use_context

    #example
    rand_t = np.random.randint(strided_dates) * stride
    rand_indiv = np.random.randint(individuals)
    x, y = data.iloc[rand_t : rand_t+lags, rand_indiv].values, data.iloc[rand_t+lags : rand_t+lags+horizon, rand_indiv].values
    x_batch, y_batch = torch.tensor(x).unsqueeze(0).unsqueeze(0), torch.tensor(y).unsqueeze(0).unsqueeze(0) # x: (1, 1, L)
    c_batch = None
    if use_context:
        xc = data.iloc[rand_t : rand_t+lags+horizon, rand_indiv].values
        c_batch = torch.tensor(xc).unsqueeze(0).unsqueeze(0)
    mean, std = get_normal_stats(x_batch)
    pred_batch = model(x_batch, c_batch)
    plot_pred(x_batch[0][0].cpu().detach().tolist(), y_batch[0][0].cpu().detach().tolist(), pred_batch[0,0].cpu().detach().tolist(), save_dir + "examples/", f"example_prediction.pdf", f"Example prediction for {save_name}")

    #eval
    t1 = perf_counter()
    for t in range(strided_dates):
        date_idx = t * stride
        for indiv in all_indiv:                      
            x, y = data.iloc[date_idx : date_idx+lags, indiv], data.iloc[date_idx+lags : date_idx+lags+horizon, indiv]
            x_batch, y_batch = torch.tensor(x.values).unsqueeze(0).unsqueeze(0), torch.tensor(y.values).unsqueeze(0).unsqueeze(0) # x: (1, 1, L)

            c_batch = None
            if use_context:
                xc = data.iloc[date_idx : date_idx+lags+horizon, indiv]
                c_batch = torch.tensor(xc.values).unsqueeze(0).unsqueeze(0)
                
            mean, std = get_normal_stats(x_batch)
            pred_batch = model(x_batch, c_batch)
            loss = criterion(pred_batch, y_batch, mean, std) # (bs, dim, H)
            point_loss = loss[0].mean().item()
            indiv_losses[indiv].append(point_loss)
            horizon_losses += loss[0].mean(dim=0).numpy() # (H)
            all_losses.append(point_loss)

    horizon_losses /= (individuals*strided_dates)
    plot_horizon_errors(horizon_losses, save_dir, name=f"{save_name}_horizon.pdf")
    plot_errors(all_losses, save_dir, name=f"{save_name}_error.pdf", title="Loss distribution")

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
    plt.savefig(save_dir+ "plots/" + f"user_errors.pdf")
    plt.close()

    
    logger.info('End of script\n')

if __name__ == "__main__":
    run()



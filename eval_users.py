import hydra
import logging
import torch

from src.timetensor.dataset import fetch_training_data, get_sizes, apply_stats
from src.timetensor.models import load_model
from src.timetensor.pipeline import get_losses, load_learner
from src.timetensor.utils import get_dirs, set_seed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.timetensor.utils import symlog

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#test
@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    logger.info("=====Running eval script=====")

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

    output_dir, save_name = cfg.misc.output_dir, cfg.misc.save_name, 
    save_name, save_dir = get_dirs(output_dir, save_name, model_name, norm_name, criterion_name, cfg.data.subsets.sizes)

    if verbose:
        logger.info(f"Fetched main configs, save directory : {save_dir}")
        logger.info(f"Model {model_name}, norm {norm_name}, criterion {criterion_name}, kwargs {kwargs}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    #data
    loaders_dict, stats_dict, nodes_stats_dict = fetch_training_data(
        data_path, cfg.data.splits, cfg.data.subsets, cfg.training.bs, lags, horizon,
        clusters=cfg.data.clustering.clusters, seed=seed, random_eval=cfg.training.random_eval, do_nodes=False)
    if cfg.data.normalize:
        apply_stats(loaders_dict, stats_dict)
    shape, shape_str, batch_str = get_sizes(loaders_dict, str_info=True)
    if verbose:
        logger.info("Fetched dataloaders")
        logger.info(shape_str)
        logger.info(batch_str)

    #model
    model = load_model(model_name, shape, norm_name, cfg.training.init, cfg.training.freeze_core, cfg.model.constants, cfg.model.residuals, stats_dict, nodes_stats_dict, device=="cpu", logger, **kwargs)
    learner = load_learner(model, norm_name, criterion, cfg.training.lr, eval_losses, device)

    #per user errors
    logger.info("--Per user eval--")
    suspects = [6, 111, 112, 113, 203]
    per_user_losses = []
    stds_per_user_losses = []
    loss_name = "NMSE"
    for indiv in range(loaders_dict["test1"].dataset.shape[0][0]):
        loaders_dict, stats_dict, nodes_stats_dict = fetch_training_data(
            data_path, cfg.data.splits, cfg.data.subsets, cfg.training.bs, lags, horizon, seed=seed,
            random_eval=cfg.training.random_eval, fetch_cluster=indiv)
        if cfg.data.normalize:
            apply_stats(loaders_dict, stats_dict)
        losses1, exotics = learner.eval(loaders_dict["test1"], return_mode="all", runs=cfg.training.eval_runs, thresholds={loss_name:10})
        if indiv in suspects:
            logger.info(exotics)
        mean = symlog(losses1[loss_name].mean())
        std = symlog(losses1[loss_name].std())
        per_user_losses.append(mean.item())
        stds_per_user_losses.append(std.item())
    
    total_mean = np.mean(per_user_losses)
    w10_mean = np.mean(np.partition(per_user_losses, int(len(per_user_losses)*0.9))[int(len(per_user_losses)*0.9):])
    
    stats_df = pd.DataFrame({
        "log(mean_error)": per_user_losses,
        "log(std_error)": stds_per_user_losses})
    plt.figure(figsize=(10, 7))
    g = sns.jointplot(
        data=stats_df,
        x="log(mean_error)",
        y="log(std_error)",
        kind='scatter',
        palette='Set1',
    )
    plt.suptitle(f"Per-user Test 1 {loss_name} of {save_name} (mean:{total_mean:.3f}, W10:{w10_mean:.3f})")
    plt.tight_layout()
    plt.savefig(save_dir+ "plots/" + "user_errors.pdf")
    plt.close()

    exotics = np.where(np.array(per_user_losses)>1)
    logger.info(exotics)

    logger.info('End of script\n')

if __name__ == "__main__":
    run()



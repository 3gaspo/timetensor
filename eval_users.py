import hydra
import logging
import torch
from time import perf_counter

from src.timetensor.dataset import fetch_training_data, get_sizes, apply_standard_norm
from src.timetensor.models import load_model
from src.timetensor.pipeline import get_losses, load_learner, launch_example
from src.timetensor.utils import get_dirs, set_seed, save_results

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
    loaders_dict, stats_dict = fetch_training_data(
        data_path, cfg.data.splits, cfg.data.sampling, cfg.data.subsets,
        cfg.training.bs, lags, horizon,
        seed=seed)
    if cfg.data.normalize:
        apply_standard_norm(loaders_dict, stats_dict)
    shape, shape_str, batch_str = get_sizes(loaders_dict, str_info=True)
    if verbose:
        logger.info("Fetched dataloaders")


    #model
    model = load_model(model_name, shape, norm_name, cfg.training.init, device.type=="cpu", **kwargs)
    learner = load_learner(model, criterion, cfg.training.lr, eval_losses, device)
    if verbose:
        logger.info("Fetched model and learner")

    #example
    launch_example(data_path, model, lags, horizon, device, save_dir, save_name, cfg.data.sampling.use_context)

    #per user errors
    logger.info("--Per user eval--")
    per_user_losses = {key: [] for key in loaders_dict}
    stds_per_user_losses = {key: [] for key in loaders_dict}
    total_means = {key: [] for key in loaders_dict}
    w10_means = {key: [] for key in loaders_dict}

    for key in loaders_dict:
        t1 = perf_counter()
        losses, exotics = learner.eval(loaders_dict[key], return_mode="indiv",
            runs=cfg.training.eval_runs, thresholds={criterion_name:100}) # {loss_name: {indiv: [steps] }}
        
        indiv_losses = losses[criterion_name]
        for indiv in indiv_losses:
            indiv_loss = indiv_losses[indiv]
            mean = symlog(indiv_loss.mean())
            std = symlog(indiv_loss.std())
            per_user_losses[key].append(mean.item())
            stds_per_user_losses[key].append(std.item())

        total_means[key] = np.mean(per_user_losses[key])
        w10_means[key] = np.mean(np.partition(per_user_losses[key], int(len(per_user_losses[key])*0.9))[int(len(per_user_losses[key])*0.9):])
        
        t2 = perf_counter()
        delta_t = (t2-t1)/60
        
        save_results(total_means[key], output_dir, f"{key}_mean_results.json", save_name, f"{criterion_name}")
        save_results(w10_means[key], output_dir, f"{key}_mean_results.json", save_name, f"w10 {criterion_name}")
        save_results(delta_t, output_dir, f"{key}_mean_results.json", save_name, f"eval time (min)")

        stats_df = pd.DataFrame({
            "log(mean_error)": per_user_losses[key],
            "log(std_error)": stds_per_user_losses[key]})
        plt.figure(figsize=(10, 7))
        g = sns.jointplot(
            data=stats_df,
            x="log(mean_error)",
            y="log(std_error)",
            kind='scatter',
            palette='Set1',
        )
        plt.suptitle(
            f"Per-user {key} {criterion_name} of {save_name} (mean:{total_means[key]:.4f}, W10:{w10_means[key]:.4f})",
            fontsize=20)     
        plt.tight_layout()
        plt.savefig(save_dir+ "plots/" + f"{key}_user_errors.pdf")
        plt.close()

    logger.info('End of script\n')

if __name__ == "__main__":
    run()



import hydra
import logging
import torch
from time import perf_counter

from src.timetensor.dataset import fetch_training_data, get_sizes, apply_standard_norm
from src.timetensor.models import load_model
from src.timetensor.pipeline import get_losses, launch_training
from src.timetensor.utils import get_dirs, set_seed, save_results

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.timetensor.utils import symlog

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    logger.info("=====Running train users script=====")

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
        logger.info(shape_str)
        logger.info(batch_str)

    #per user training
    logger.info("--Per user eval--")
    per_user_losses = {key: [] for key in loaders_dict}
    stds_per_user_losses = {key: [] for key in loaders_dict}
    total_means = {key: [] for key in loaders_dict}
    w10_means = {key: [] for key in loaders_dict}
    train_times = []
    eval_times = {key: [] for key in loaders_dict}

    train_keys = [key for key in loaders_dict if key in ["train","valid1", "test1"]]
    for indiv in range(loaders_dict["train"].dataset.shape[0][0]): #train indivs
        save_dir_ = save_dir + f"user_{indiv}/"
        if not os.path.exists(save_dir_):
            os.makedirs(save_dir_)

        model = load_model(model_name, shape, norm_name, cfg.training.init, device.type=="cpu", **kwargs)

        loaders_dict_ = {}
        for key in train_keys:
            loaders_dict[key].dataset.set_sampler(subset_mode="individuals", subset_indices=[indiv])
            loaders_dict_[key] = loaders_dict[key]
        
        t1 = perf_counter()
        learner = launch_training(model,
            norm_name, criterion, cfg.training.lr, cfg.training.epochs, loaders_dict_, eval_losses, device,
            save_dir_, save_name, cfg.training.eval_freq, cfg.training.print_freq, logger, verbose=0, save=True)
        t2 = perf_counter()
        train_times.append((t2-t1)/60)

        for key in train_keys: #train, (valid1), test1
            
            t1 = perf_counter()
            losses, _ = learner.eval(loaders_dict_[key], return_mode="steps",
                runs=cfg.training.eval_runs)

            mean = symlog(losses[criterion_name].mean())
            std = symlog(losses[criterion_name].std())
            per_user_losses[key].append(mean.item())
            stds_per_user_losses[key].append(std.item())

            t2 = perf_counter()
            eval_times[key].append((t2-t1)/60)

    save_results(np.mean(train_times), output_dir, f"train_mean_results.json", save_name, f"avg train time (min)")
    save_results(np.sum(train_times), output_dir, f"train_mean_results.json", save_name, f"train time (min)")

    for key in train_keys: #train, (valid1), test1  (valid2 valid3 test2 don't have an associated model)
        total_means[key] = np.mean(per_user_losses[key])
        w10_means[key] = np.mean(np.partition(per_user_losses[key], int(len(per_user_losses[key])*0.9))[int(len(per_user_losses[key])*0.9):])
        
        save_results(total_means[key], output_dir, f"{key}_mean_results.json", save_name, f"{criterion_name}")
        save_results(w10_means[key], output_dir, f"{key}_mean_results.json", save_name, f"w10 {criterion_name}")
        save_results(np.mean(eval_times[key]), output_dir, f"{key}_mean_results.json", save_name, f"avg eval time (min)")
        save_results(np.sum(eval_times[key]), output_dir, f"{key}_mean_results.json", save_name, f"eval time (min)")
        
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



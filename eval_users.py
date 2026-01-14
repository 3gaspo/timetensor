import hydra
import logging
import torch

from src.timetensor.dataset import fetch_training_data, get_sizes, apply_standard_norm#, format_individual_splits
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
    save_name, save_dir = get_dirs(output_dir, save_name, model_name, norm_name, criterion_name, cfg.data.subsets)

    if verbose:
        logger.info(f"Fetched main configs, save directory : {save_dir}")
        logger.info(f"Model {model_name}, norm {norm_name}, criterion {criterion_name}, kwargs {kwargs}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    #data
    loaders_dict, stats_dict, _ = fetch_training_data(
        data_path, cfg.data.splits, cfg.data.subsets, cfg.training.bs, lags, horizon,
        seed=seed, random_eval=cfg.training.random_eval)
    if cfg.data.normalize:
        apply_standard_norm(loaders_dict, stats_dict)
    
    shape, shape_str, batch_str = get_sizes(loaders_dict, str_info=True)
    if verbose:
        logger.info("Fetched dataloaders")


    #model
    model = load_model(model_name, shape, norm_name, cfg.training.init, cfg.model.do_constants, device=="cpu", **kwargs)
    learner = load_learner(model, criterion, cfg.training.lr, eval_losses, device)
    if verbose:
        logger.info("Fetched model and learner")


    #per user errors
    logger.info("--Per user eval--")
    per_user_losses = {key: [] for key in loaders_dict}
    stds_per_user_losses = {key: [] for key in loaders_dict}
    total_means = {key: [] for key in loaders_dict}
    w10_means = {key: [] for key in loaders_dict}
    
    all_keys = list(loaders_dict.keys())

    if len(all_keys) == 1: #we can subset individuals using [indiv] for indiv in range(...)
        main_key = all_keys[0]
        for indiv in range(loaders_dict[main_key].dataset.shape[0][0]):
            # splits_, subsets_ = format_individual_splits(cfg.data.splits, cfg.data.subsets)
            loaders_dict_, stats_dict_, _ = fetch_training_data(
                data_path, cfg.data.splits, cfg.data.subsets, cfg.training.bs, lags, horizon,
                seed=seed, random_eval=cfg.training.random_eval, cluster_ids=[indiv])
            if cfg.data.normalize:
                apply_standard_norm(loaders_dict_, stats_dict_)

            losses, _ = learner.eval(loaders_dict_[main_key], return_mode="all",
                runs=cfg.training.eval_runs, thresholds={criterion_name:100})
            mean = symlog(losses[criterion_name].mean())
            std = symlog(losses[criterion_name].std())
            per_user_losses[main_key].append(mean.item())
            stds_per_user_losses[main_key].append(std.item())

    else: #we need to use individuals subset fr om original splits
        assert "train" in all_keys
        indiv_keys = ["train"]
        if "valid2" in all_keys: #6 way split
            indiv_keys.append("valid2")
        elif "test0" in all_keys: #4 way split
            indiv_keys.append("test0")
        
        for key in indiv_keys:
            for indiv in range(loaders_dict[key].dataset.shape[0][0]):
                loaders_dict[key].dataset.set_sampler(subset_mode="individuals", subset_indices=[indiv])
                
                for key in loaders_dict_: #train, (valid1), test1
                    losses, _ = learner.eval(loaders_dict_[key], return_mode="all",
                        runs=cfg.training.eval_runs, thresholds={criterion_name:100})
                    mean = symlog(losses[criterion_name].mean())
                    std = symlog(losses[criterion_name].std())
                    per_user_losses[key].append(mean.item())
                    stds_per_user_losses[key].append(std.item())

    for key in loaders_dict_:
        total_means[key] = np.mean(per_user_losses[key])
        w10_means[key] = np.mean(np.partition(per_user_losses[key], int(len(per_user_losses[key])*0.9))[int(len(per_user_losses[key])*0.9):])
        
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
        plt.suptitle(f"Per-user {key} {criterion_name} of {save_name} (mean:{total_means[key]:.3f}, W10:{w10_means[key]:.3f})")
        plt.tight_layout()
        plt.savefig(save_dir+ "plots/" + f"{key}_user_errors.pdf")
        plt.close()

    logger.info('End of script\n')

if __name__ == "__main__":
    run()



import hydra
import logging
import torch
import numpy as np
from time import perf_counter

from src.timetensor.dataset import fetch_training_data, get_sizes, apply_stats
from src.timetensor.models import load_model
from src.timetensor.pipeline import Learner, get_losses, launch_eval

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    logger.info("=====Running train script=====")

    #configs
    data_path = cfg.data.path
    lags, horizon = int(cfg.task.lags), int(cfg.task.horizon)
    split_kwargs, subset_kwargs = cfg.data.splits, cfg.data.subsets
    clusters = cfg.data.clustering.clusters
    batch_size = cfg.training.bs
    model_name, norm_name, norm_kwargs, model_kwargs = cfg.model.name, cfg.normalization.name, cfg.normalization.configs, cfg.model.configs
    if norm_name == "None":
        norm_name = None
    init_path = cfg.training.init
    kwargs = {**(norm_kwargs or {}), **(model_kwargs or {})}
    seed = cfg.misc.seed
    logger.info(f"Fetched configs")

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    #data
    loaders_dict, stats_dict = fetch_training_data(data_path, split_kwargs, subset_kwargs, batch_size, lags, horizon, clusters=clusters, seed=seed)
    if cfg.data.normalize:
        apply_stats(loaders_dict, stats_dict)
    shape, _, _ = get_sizes(loaders_dict, str_info=True)
    logger.info("Fetched dataloaders")

    #model
    criterion_name = "NMSE"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_name, shape, norm_name, init_path, constants=False, **kwargs).to(device)
    
    criterion, eval_losses = get_losses("NMSE")
    learner = Learner(model, criterion, 1, eval_losses, device=device)
    logger.info("Fetched learner")

    #test
    logger.info(f"{loaders_dict["test1"].dataset.shape}")
    t1 = perf_counter()
    logger.info("launching eval")
    test_losses1, test_losses2 = launch_eval(learner, loaders_dict, eval_losses, None, None, False, save=False)
    t2 = perf_counter()
    logger.info(f"Done in {(t2-t1)/60:.2f} min")
    logger.info(f"{test_losses1["NMSE"].shape}")
    logger.info(f"{np.where(np.isnan(test_losses1["NMSE"]))}")
    logger.info('End of script\n')

if __name__ == "__main__":
    run()



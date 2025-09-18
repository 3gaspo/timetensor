import hydra
import logging
import torch
import numpy as np
from time import perf_counter

from src.timetensor.dataset import fetch_training_data, get_sizes, apply_stats
from src.timetensor.models import load_model
from src.timetensor.pipeline import Learner, get_losses

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
    logger.info("Fetched learner")

    #test
    x = loaders_dict["valid1"].dataset.values[205, :, :lags].unsqueeze(0).to(device)
    norm = model.InstanceNorm(x)
    pred = model(x)
    print(norm)
    print(pred)

    model2 = load_model(model_name, shape, None, init_path, constants=False, **kwargs).to(device)
    norm = model.InstanceNorm(x)
    pred = model(x)
    print(norm)
    print(pred)

    logger.info('End of script\n')

if __name__ == "__main__":
    run()



import hydra
import logging
import torch
import numpy as np
from src.timetensor.dataset import fetch_training_data, apply_stats
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    logger.info("=====Running data script=====")

    #configs
    data_path = cfg.data.path
    lags, horizon = int(cfg.task.lags), int(cfg.task.horizon)
    seed = cfg.misc.seed
    split_kwargs, subset_kwargs = cfg.data.splits, cfg.data.subsets
    clusters = cfg.data.clustering.clusters
    logger.info("Fetched configs")

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    #data
    loaders_dict, stats_dict = fetch_training_data(data_path, split_kwargs, subset_kwargs, cfg.training.bs, lags, horizon, clusters=clusters, seed=seed)
    if cfg.data.normalize:
        apply_stats(loaders_dict, stats_dict)

    #test
    x, c, y = next(iter(loaders_dict["train"]))
    print("context: ", c.shape)

    logger.info('End of script\n')

if __name__ == "__main__":
    run()


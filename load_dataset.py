## This script loads a raw dataset, builds associated tensors and prints split shapes

import hydra
import logging
import os
from time import perf_counter

from src.timetensor.dataset import get_sizes, fetch_training_data, set_random_data
from src.timetensor.visu import plot_named_example

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    logger.info("=====Running data script=====")

    #configs
    data_path, dataset_name, context_cols = cfg.data.path, cfg.data.dataset, cfg.data.context_cols
    lags, horizon = int(cfg.task.lags), int(cfg.task.horizon)
    seed, verbose = cfg.misc.seed, cfg.misc.verbose

    #dirs
    if not os.path.exists(data_path+"examples/"):
        os.makedirs(data_path+"examples/")

    rebuild = cfg.load.rebuild
    new_example = cfg.load.example
    do_shapes = cfg.load.shapes
    do_shapes = True

    if verbose:
        logger.info("Fetched configs")
        logger.info(f"Loading {dataset_name}")
    
    #build pytorch dataset
    if rebuild:
        t1 = perf_counter()
        if "synthetic" in dataset_name:
            from src.timetensor.synthetic import build_dataset
            build_dataset(data_path, n1=cfg.data.n1, n2=cfg.data.n2, r1=cfg.data.r1, r2=cfg.data.r2, seed=seed)
        else:
            from src.timetensor.dataset import build_dataset
            build_dataset(data_path, dataset_name, context_cols, drop_users=cfg.data.splits.drop_users, aggr=cfg.data.aggregation)
        t2 = perf_counter()
        if verbose:
            logger.info(f"Rebuilt dataset in {(t2-t1)/60:.3f} min")

    #get example
    if new_example:
        ex_dir = data_path + "examples/" + f"{lags}_{horizon}/"
        set_random_data(data_path, lags, horizon, name="rand")
        plot_named_example(ex_dir, f"rand")
        if verbose:
            logger.info("Set new example")

    #splits
    if do_shapes:
        loaders_dict, stats_dict = fetch_training_data(data_path, 
            cfg.data.splits, cfg.data.sampling, cfg.data.subsets,
            cfg.training.bs, lags, horizon,
            seed=seed)
        _, shape_str, batch_str = get_sizes(loaders_dict, str_info=True)
        if verbose:
            logger.info("Fetched dataloaders")
            logger.info(shape_str)
            logger.info(batch_str)
        for key in stats_dict:
            stats_str = "\n".join(f"{k}\t{v:.4f}" for k, v in stats_dict[key].items() if k != "shape")
            logger.info(f"{key} stats:\n{stats_str}")

    logger.info('End of script\n')

if __name__ == "__main__":
    run()


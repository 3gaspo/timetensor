## Script to rebuild dataset from scratch if cfg.rebuild=True
## Sets a new random data as example

import hydra
import logging
from time import perf_counter

#remove src if using this script in another working directory
from src.timetensor.dataset import build_datasets, load_datasets, get_train_loaders
from src.timetensor.utils import set_random_data, fetch_example_data
#from src.timetensor.utils import normalize
from src.timetensor.visu import plot_example, scatter_stats#, plot_stats
from src.timetensor.utils import unroll_windows

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


@hydra.main(version_base=None, config_path="config", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    print("\n")
    logger.info("=====Running data script=====")

    #configs
    path = cfg.data.path
    rebuild = cfg.data.rebuild
    reset = cfg.data.reset
    verbose = cfg.misc.verbose

    lags, horizon = cfg.model.lags, cfg.model.horizon
    batch_size, subset_data = cfg.training.bs, cfg.training.subset_data


    if verbose:
        logger.info("Fetched configs")

    #dataset
    if rebuild:
        if verbose:
            logger.info("Rebuilding dataset")
        t1 = perf_counter()
        dataset = cfg.data.dataset
        if dataset == "electricity":
            #remove src if using this script in another working directory
            from src.timetensor.electricity import fetch_data
            fetcher = lambda path: fetch_data(path, raw_format=cfg.data.format, years=None, hourly=None)
        else:
            "Dataset name not recognized"
        build_datasets(fetcher, path,  cfg.data.indiv_split, cfg.data.date_split, cfg.data.seed)
        t2 = perf_counter()
        logger.info(f"Build in {(t2-t1)/60:.3f} min")


    #loaders
    loaders_dict = get_train_loaders(path, batch_size, lags, horizon, subset=subset_data)
    if verbose:
        if subset_data < 1:
            logger.info(f"Fetched dataloaders with subset ratio : {subset_data}")
        else:
            logger.info("Fetched dataloaders")
    #sizes
    if verbose:
        logger.info(f"Training data shape : {loaders_dict['train'].dataset.shape}")
        X, c, y = next(iter(loaders_dict["train"])) # (indiv, dim, lags),  #(nc, dim, horizon),  #(indiv, dim, horizon)
        if c is not None:
            logger.info(f"Batch sizes : X={X.shape}, c={c.shape}, y={y.shape}")
        else:
            logger.info(f"Batch sizes : X={X.shape}, y={y.shape}")

    #stats
    if verbose:
        logger.info("Plotting stats")
    datasets_dict = load_datasets(path)
    scatter_stats({key: datasets_dict[key]["values"] for key in datasets_dict}, path, name="stats_split_central.pdf", dim=0)
    scatter_stats({"train":datasets_dict["train"]["values"], "subset_train":loaders_dict["train"].dataset.values}, path, name="stats_subset_central.pdf", dim=0)
    scatter_stats({key: unroll_windows(loaders_dict[key])[0] for key in loaders_dict}, path, name="unrolled_stats_split_central.pdf", dim=0)

    #examples
    if reset:
        if verbose:
            logger.info("Setting new example")
        lags = cfg.model.lags
        horizon = cfg.model.horizon
        name = cfg.data.example_name
        set_random_data(path, "train", lags, horizon, name=name)
        x, c, y, i, d  = fetch_example_data(path, "rand")
        if verbose:
            logger.info(f"Set indiv {i} date {d} as example")
        x_normalized, mean, std =  normalize(x, return_stats=True)
        y_normalized = (y - mean)/std
        plot_example(x[0], y[0], path, f"{name}_example.pdf", "Example")        
        plot_example(x_normalized[0], y_normalized[0], path, f"{name}_normal_example.pdf", "Normlized Example")        

    logger.info('End of script\n')

if __name__ == "__main__":
    run()



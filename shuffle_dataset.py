## Script to rebuild dataset from scratch if cfg.rebuild=True
## Sets a new random data as example

import hydra
import logging
from time import perf_counter

#remove src if using this script in another working directory
from src.timetensor.dataset import get_train_loaders, build_dataset, get_dataset_splits #,build_datasets, load_datasets
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
    data_path = cfg.data.path
    verbose = cfg.misc.verbose

    lags, horizon = cfg.model.lags, cfg.model.horizon
    batch_size = cfg.training.bs


    if verbose:
        logger.info("Fetched configs")

    #dataset and loaders
    if verbose:
        logger.info("Rebuilding dataset")
    t1 = perf_counter()
    dataset = cfg.data.dataset
    if dataset == "electricity":
        from src.timetensor.electricity import fetch_data  #adapt path if script in another working directory
        fetcher = lambda path: fetch_data(path, raw_format=cfg.data.format, years=None, hourly=None)
    else:
        "Dataset name not recognized"
    build_dataset(fetcher, data_path) #builds dataset from raw data and saves as .pt
    data_dict = get_dataset_splits(data_path, cfg.data.indiv_split, cfg.data.date_split, cfg.misc.seed, save=False) #will only save the train test indices, in path
    loaders_dict = get_train_loaders(data_dict, batch_size, lags, horizon, by_date=True, subsets=cfg.data.subset, path=data_path) #will generate the subsets and save indices
    t2 = perf_counter()
    logger.info(f"Build in {(t2-t1)/60:.3f} min")

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
    scatter_stats({key: data_dict[key][0] for key in data_dict}, data_path, name="stats_split_central.pdf", title="Individuals distributions")
    scatter_stats({"train":data_dict["train"][0], "subset_train":loaders_dict["train"].dataset.values}, data_path, name="stats_subset_central.pdf", title="Individuals distributions")
    scatter_stats({key: unroll_windows(loaders_dict[key])[0] for key in ["train", "test"]}, data_path, name="unrolled_stats_split_central.pdf", title="Inputs distribution")

    #examples
    if verbose:
        logger.info("Setting new example")
    lags = cfg.model.lags
    horizon = cfg.model.horizon
    set_random_data(data_path, lags, horizon, name="rand")
    x, c, y, i, d  = fetch_example_data(data_path + "/examples/", "rand")
    if verbose:
        logger.info(f"Set indiv {i} date {d} as example")
    plot_example(x[0], y[0], data_path + "/examples/rand/", f"example.pdf", "Example")

    logger.info('End of script\n')

if __name__ == "__main__":
    run()



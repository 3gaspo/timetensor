## Script to rebuild dataset from scratch if cfg.rebuild=True
## Sets a new random data as example

import hydra
import logging
from time import perf_counter

from src.dataset import build_datasets, load_datasets
from src.utils import set_random_data, fetch_example_data
from src.utils import normalize
from src.visu import plot_example, plot_stats

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
    logger.info("Fetched configs")

    #data
    if rebuild:
        logger.info("Rebuilding dataset")
        t1 = perf_counter()
        dataset = cfg.data.dataset
        if dataset == "electricity":
            from datasets.electricity import fetch_data
            fetcher = lambda path: fetch_data(path, raw_format=cfg.data.format, years=None, hourly=None)
        else:
            "Dataset name not recognized"
        build_datasets(fetcher, path,  cfg.data.indiv_split, cfg.data.date_split, cfg.data.seed)
        t2 = perf_counter()
        logger.info(f"Build in {(t2-t1)/60:.3f} min")

        data_splits = load_datasets(path)
        plot_stats(data_splits, "mean", path, "mean_distributions.pdf")
        plot_stats(data_splits, "max", path, "max_distributions.pdf") 
    
    if reset:
        logger.info("Setting new example")
        lag = cfg.model.lag
        horizon = cfg.model.horizon
        set_random_data(path, "train_", lag, horizon, name="rand")
        x, c, y, i, d  = fetch_example_data(path, "rand")
        logger.info(f"Set indiv {i} date {d} as example")
        x_normalized, mean, std =  normalize(x, return_stats=True)
        y_normalized = (y - mean)/std
        plot_example(x[0], y[0], path, f"example.pdf", "Example")        
        plot_example(x_normalized[0], y_normalized[0], path, f"normal_example.pdf", "Normlized Example")        

    logger.info('End of script\n')

if __name__ == "__main__":
    run()



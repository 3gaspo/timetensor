import hydra
import logging
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

from src.timetensor.dataset import TimeSeriesDataset, load_data
from src.timetensor.models import load_model
from src.timetensor.xpc import Game, BackgroundDataset

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    logger.info("=====Running main script=====")

    #configs
    data_path = cfg.data.path
    lags, horizon = cfg.task.lags, cfg.task.horizon
    verbose = cfg.misc.verbose
    if verbose:
        logger.info("Fetched main configs")

    #data
    values, _, _ = load_data(data_path)
    dataset = TimeSeriesDataset(values, lags=lags, horizon=horizon, by_date=False)
    
    #model
    shape = (lags, values.shape[1], horizon)
    model = load_model("lookback", shape, "None")

    #game
    background = BackgroundDataset(dataset)
    players = {f"lag_{k}": (0, 0, k) for k in range(lags)}
    game = Game(model, players, background)
    logger.info(f"Loaded game with {game.players} players and {len(game.background)} examples")

    #experiment
    ncoalitions=10
    nexamples=10
    logger.info("Computing shap")
    t1 = perf_counter()
    shapley_values = game.get_shapley_values(dataset[0][0], ncoalitions, nexamples, replace=True, aggregate=False, split=False, logger=logger)
    t2 = perf_counter()
    logger.info(f"Done in {(t2-t1)/60:.2f} min")

    #plot
    shapley_values = np.array(list(shapley_values.values()))
    plt.figure(figsize=(10,10))
    plt.imshow(shapley_values[:50, 0, :].T)
    plt.savefig("shap.pdf")

    logger.info('End of script\n')

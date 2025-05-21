import hydra
import logging
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.timetensor.dataset import TimeSeriesDataset, load_data
from src.timetensor.dataset import get_dataset_splits, get_train_loaders
from src.timetensor.models import load_model
from src.timetensor.pipeline import Learner, Loss

from src.timetensor.xpc import Game, BackgroundDataset


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


@hydra.main(version_base=None, config_path="config", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    logger.info("=====Running main script=====")

    #configs
    data_path = cfg.data.path
    lags, horizon = cfg.model.lags, cfg.model.horizon
    batch_size, lr = cfg.training.bs, cfg.training.lr
    normalization = cfg.model.normalization
    model_name, kwargs = cfg.model.name, cfg.model_configs
    verbose = cfg.misc.verbose
    if verbose:
        logger.info("Fetched main configs")

    #data   
    by_idx = "individuals"
    data_dict = get_dataset_splits(data_path, cfg.data.indiv_split, cfg.data.date_split, cfg.misc.seed, save=False)
    loaders_dict = get_train_loaders(data_dict, batch_size, lags, horizon, by_date=(by_idx=="dates"), subsets=cfg.subset.subsets, path=data_path+"subsets/")
    #sizes
    X, c, y = next(iter(loaders_dict["train"])) # (indiv, dim, lags),  #(nc, dim, horizon),  #(indiv, dim, horizon)
    shape = [X.shape[2], X.shape[1], y.shape[2]]
    #training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = Loss(nn.MSELoss())
    eval_losses = {"MSE": Loss(nn.MSELoss(reduction="none"))}
    model_name = "sklinear"
    model = load_model(model_name, shape, normalization, **kwargs)
    logger.info("Starting scikit-learn fitting...")
    learner = Learner(model, criterion, lr, eval_losses, device=device, pytorch=False)
    learner.fit(loaders_dict["train"])
    logger.info("End of training")
    
    #game
    values, context, datetimes = load_data(data_path) #load dataset
    dataset = TimeSeriesDataset(values, datetimes, context, lags, horizon, by_date=False)
    background = BackgroundDataset(dataset)
    players = {f"lag_{k}": (0, 0, k) for k in range(lags)}
    game = Game(model, players, background)
    logger.info("Loaded game")
    shapley_values = game.get_shapley_values(dataset[0], 1, 1, replace=True, aggregate=False, split=False)
    logger.info("Computed shap")

    plt.figure()
    plt.plot(shapley_values.values())
    plt.savefig("shap.pdf")

    logger.info('End of script\n')

if __name__ == "__main__":
    run()



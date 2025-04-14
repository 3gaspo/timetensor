import hydra
import logging
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.timetensor.dataset import get_dataset_splits, get_train_loaders
from src.timetensor.models import load_model
from src.timetensor.pipeline import train_model
from src.timetensor.utils import unroll_windows

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

@hydra.main(version_base=None, config_path="config", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    print("\n")
    logger.info("=====Running main script=====")

    #configs
    data_path = cfg.data.path
    lags, horizon = cfg.model.lags, cfg.model.horizon
    batch_size = cfg.training.bs
    model_name = cfg.model.name
    kwargs = cfg.model_configs
    verbose = cfg.misc.verbose
    if verbose:
        logger.info("Fetched main configs")

    #save dirs
    output_dir = cfg.misc.output_dir
    save_name = "sklinear"
    if save_name is None:
        save_name = model_name
    save_dir = output_dir + save_name + "/" #current experiment dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    hydra_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir #hydra logs
    with open(save_dir + f'hydra_dir.txt', 'w') as file: 
        file.write(f"{hydra_dir}") #save path of hydra logs to experiment dir
    if verbose:
        logger.info("Fetched output directories")
        logger.info(f"Save directory : {save_dir}")

    #data
    data_dict = get_dataset_splits(data_path, cfg.data.indiv_split, cfg.data.date_split, cfg.misc.seed, save=False)
    loaders_dict = get_train_loaders(data_dict, batch_size, lags, horizon, by_date=True, subsets=cfg.data.subset, path=data_path)
    if verbose:
        logger.info("Fetched dataloaders")

    #sizes
    X, c, y = next(iter(loaders_dict["train"])) # (indiv, dim, lags),  #(nc, dim, horizon),  #(indiv, dim, horizon)
    shape = [X.shape[1], X.shape[2], y.shape[2]]

    #model
    model = load_model("sklinear", shape, False, **kwargs)
    Xtrain, Ytrain = unroll_windows(loaders_dict["train"])
    print("Shapes: ", Xtrain.shape, Ytrain.shape)
    Xtrain, Ytrain = Xtrain[:, 0, :], Ytrain[:, 0, :]
    model.fit(Xtrain, Ytrain)

    #plotting weights
    weights = model.reg.coef_
    print(weights.shape)


    plt.imshow(weights, aspect='auto', cmap='viridis')
    plt.colorbar(label='Weight value')
    plt.xlabel('Inputs (lookback)')
    plt.ylabel('Outputs (horizon)')
    plt.title('Model weights')
    plt.savefig(save_dir + "weights.pdf")

    logger.info('End of script\n')

if __name__ == "__main__":
    run()



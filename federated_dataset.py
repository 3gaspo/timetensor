import hydra
import logging
import os
import torch
import torch.nn as nn
import numpy as np

from src.timetensor.dataset import get_train_loaders, TimeSeriesDataset, load_datasets
from src.timetensor.models import load_model
from src.timetensor.federated import build_split_datasets, Client

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

@hydra.main(version_base=None, config_path="config", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    print("\n\n")
    logger.info("=====Running main script=====")

    #configs
    data_path = cfg.data.path
    splits = cfg.fed.splits
    lags, horizon = cfg.model.lags, cfg.model.horizon
    batch_size = cfg.training.bs
    verbose = cfg.misc.verbose
    if verbose:
        logger.info("Fetched configs")
        logger.info(f"Found splits: {splits}")

    #data
    build_split_datasets(data_path, splits, shuffle=True, replace=False, seed=None, context_by_individuals=False)
    if verbose:
        logger.info("Build split data")

    clients = []
    for k in range(len(splits)):
        path = data_path + f"node_{k}/"
        loaders_dict = get_train_loaders(path, batch_size, lags, horizon, valid_mode=1, by_date=True, subset=data_path+"subset_indices_0.1.pt")
        client = Client(loaders_dict, id=k)
        clients.append(client)

    for client in clients:
        train_data = client.dataloaders["train"].dataset
        print("client_id : ", client.id, " dataset : ", train_data.shape())

if __name__ == "__main__":
    run()



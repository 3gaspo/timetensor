import hydra
import logging
import os
import torch
import torch.nn as nn
import numpy as np

from src.timetensor.dataset import get_train_loaders, TimeSeriesDataset, load_datasets
from src.timetensor.models import load_model
from src.timetensor.federated import build_split_datasets, Client
from src.timetensor.visu import scatter_stats
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
    splits = cfg.fed.splits
    lags, horizon = cfg.model.lags, cfg.model.horizon
    batch_size = cfg.training.bs
    verbose = cfg.misc.verbose
    rebuild = cfg.data.rebuild
    N = len(splits)
    if verbose:
        logger.info("Fetched configs")
        logger.info(f"Found splits: {splits}")

    #data
    if rebuild:
        if verbose:
            logger.info("Rebuilding dataset")
        build_split_datasets(data_path, splits, shuffle=True, replace=False, seed=None, context_by_individuals=False)

    clients = []
    for k in range(N):
        path = data_path + f"node_{k}/"
        loaders_dict = get_train_loaders(path, batch_size, lags, horizon, valid_mode=1, by_date=True, subset=data_path+"subset_indices_0.2.pt")
        client = Client(loaders_dict, id=k)
        clients.append(client)

    for client in clients:
        dataloaders = client.dataloaders
        print("client_id : ", client.id, " train=", dataloaders["train"].dataset.shape, " valid=", dataloaders["valid"].dataset.shape, " test=", dataloaders["test"].dataset.shape)

    scatter_stats({f"node{k}": clients[k].dataloaders["train"].dataset.values for k in range(N)}, data_path, name="stats_nodes.pdf", dim=0)
    scatter_stats({f"node{k}": unroll_windows(clients[k].dataloaders["train"])[0] for k in range(N)}, data_path, "unrolled_stats_nodes.pdf")

    logger.info("Plots done")


if __name__ == "__main__":
    run()



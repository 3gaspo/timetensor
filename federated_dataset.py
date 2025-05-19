##Builds federated dataset from central dataset, including splits and subsets, and plots stats and examples

import hydra
import logging
import os
import matplotlib.pyplot as plt
import numpy as np

from src.timetensor.dataset import get_train_loaders
from src.timetensor.federated import get_client_splits, Client
from src.timetensor.visu import scatter_stats, plot_stats, scatter_input_output
from src.timetensor.utils import unroll_windows

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

@hydra.main(version_base=None, config_path="config", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    logger.info("=====Running main script=====")

    #configs
    data_path = cfg.data.path
    lags, horizon = cfg.model.lags, cfg.model.horizon
    batch_size = cfg.training.bs
    verbose = cfg.misc.verbose
    if verbose:
        logger.info("Fetched configs")
    
    splits=[0.5, 0.3, 0.2]
    for node_split in ["clusters", "random_silo", "all"]:
        if verbose:
            logger.info(f"--Rebuilding nodes with {node_split}--")
        
        if node_split=="clusters":
            path=data_path + "nodes_clusters/"
            nodes=path+"splits/indices/"
        elif node_split=="random_silo":
            path=data_path + "nodes_rand/"
            nodes=[0.5, 0.3, 0.2]
        else:
            path =data_path+"nodes_all/"
            nodes=None
       
        if not os.path.exists(path):
            os.makedirs(path)
        for folder_name in ["splits", "subsets", "plots"]:
            if not os.path.exists(path + folder_name +"/"):
                os.makedirs(path + folder_name +"/")
        if not os.path.exists(path + "splits/indices/"):
            os.makedirs(path + "splits/indices/")
        #data
        node_data_dict = get_client_splits(data_path, nodes, splits, seed=None, context_by_individuals=False, path=path+"splits/")
        N = len(node_data_dict)
        
        #loader
        clients = []
        subsets={"train":0.1, "valid":0.1, "valid2":0.1, "test":0.1}
        for k in range(N):
            _ = get_train_loaders(node_data_dict[f"node_{k}"], batch_size, lags, horizon, by_date=True, subsets=subsets, subset_mode="dates", path=path+f"subsets/")
            loaders_dict = get_train_loaders(node_data_dict[f"node_{k}"], batch_size, lags, horizon, by_date=True)
            client = Client(loaders_dict, id=k)
            clients.append(client)

        #sizes
        if len(clients)<=10 and verbose:
            for client in clients:
                dataloaders = client.dataloaders
                print("client_id : ", client.id, " train=", dataloaders["train"].dataset.shape, " valid=", dataloaders["valid"].dataset.shape, " test=", dataloaders["test"].dataset.shape)

        #plots
        if len(clients)<=10:
            unrolls = {f"node_{k}": unroll_windows(clients[k].dataloaders["train"]) for k in range(N)}
            nunrolls = {f"node_{k}": unroll_windows(clients[k].dataloaders["train"], normal=True) for k in range(N)}
            x_dict = {key: unrolls[key][0] for key in unrolls}
            y_dict = {key: unrolls[key][1] for key in unrolls}
            ny_dict =  {key: nunrolls[key][1] for key in nunrolls}
            
            scatter_input_output(x_dict, y_dict, path+"plots/", name="output_inputs_nodes.pdf")
            scatter_stats(x_dict, path+"plots/", name="inputs_stats_nodes.pdf", title="Inputs statistics")
            plot_stats(ny_dict, path+"plots/", name="normal_outputs_nodes.pdf", title="Normalized outputs distribution", logscale=False, limits=(-5, 5))

        else:
            centroids = [clients[k].dataloaders["train"].dataset.values.mean() for k in range(N)]
            plot_stats(centroids, path+"plots/", name="centroids_nodes.pdf", title="Centroids distribution", logscale=True, limits=(0, 5))


    logger.info("Plots done")
    logger.info('End of script\n')

if __name__ == "__main__":
    run()



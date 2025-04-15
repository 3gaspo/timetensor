import hydra
import logging
import os
import torch
import torch.nn as nn
import numpy as np

from src.timetensor.dataset import get_dataset_splits, get_train_loaders
from src.timetensor.models import load_model
from src.timetensor.pipeline import Learner, train_model
from src.timetensor.federated import get_client_splits, Client
from src.timetensor.utils import save_results, fetch_example_data, append_in_dict
from src.timetensor.fedavg import LocalFedAvg, GlobalFedAvg

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
    batch_size, lr = cfg.training.bs, cfg.training.lr
    criterion_name, normalized, normalization = cfg.training.loss, cfg.training.normalized,  cfg.model.normalization
    model_name, retrain = cfg.model.name, cfg.training.retrain
    kwargs = cfg.model_configs
    verbose, benchmark = cfg.misc.verbose, cfg.misc.benchmark
    splits = cfg.fed.splits
    N = len(splits)

    if verbose:
        logger.info("Fetched main configs")
        logger.info(f"Model {model_name}, normalization {normalization}, kwargs {kwargs}")
        logger.info(f"Found {N} splits")


    #save dirs
    output_dir = cfg.misc.output_dir
    save_name = cfg.misc.save_name
    if save_name is None:
        save_name = model_name
        if normalization:
            save_name = save_name + f"_normalization{normalization}"   
    save_dir = output_dir + save_name + "/" #current experiment dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    hydra_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir #hydra logs
    with open(save_dir + f'hydra_dir.txt', 'w') as file: 
        file.write(f"{hydra_dir}") #save path of hydra logs to experiment dir
    if not os.path.exists(save_dir + "examples/"): #dir for example predictions
        os.makedirs(save_dir + "examples/")
    if verbose:
        logger.info("Fetched output directories")
        logger.info(f"Save directory : {save_dir}")

    #nodes
    node_data_dict = get_client_splits(data_path, splits)
    nodes = []
    for k in range(N):

        #data
        path = data_path + f"node_{k}/"
        loaders_dict = get_train_loaders(node_data_dict[f"node_{k}"], batch_size, lags, horizon, by_date=True, subsets=cfg.data.subset, path=data_path + f"/node_{k}/")
        client = Client(loaders_dict, id=k)

        dataloaders = client.dataloaders
        logger.info(f"Client_id={client.id} | train={dataloaders['train'].dataset.shape} valid={dataloaders['valid'].dataset.shape} test={dataloaders['test'].dataset.shape}")
    
        X, c, y = next(iter(dataloaders["train"]))
        shape = [X.shape[1], X.shape[2], y.shape[2]]

        #learner        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if criterion_name == "MSE":
            criterion = nn.MSELoss()
        else:
            print("Unknown criterion name")
            criterion = None
        eval_losses = {"MSE":nn.MSELoss(reduction="none")}

        if retrain:
            model = load_model(model_name, shape, normalization, **kwargs)
            learner = Learner(model, criterion, lr, eval_losses, device=device, normalized_criterion=True)
        else:
            model = load_model(model_name, shape, normalization, **kwargs)
            learner = Learner(model, criterion, lr, eval_losses, device=device, normalized_criterion=True)
            model.load_state_dict(torch.load(save_dir + f"node_{k}/" + "trained_model.pt"))
            model.to(device)

        node = LocalFedAvg(client, learner)
        nodes.append(node)

    #server
    model = load_model(model_name, shape, normalization, **kwargs)
    server = GlobalFedAvg(model)

    #FedAvg
    E, K = 2, 3
    logger.info(f"==Starting FedAvg==")
    valid_losses = {f"node_{k}": {} for k in range(N)}
    valid_losses2 = {f"node_{k}": {} for k in range(N)}
    for k in range(K):
        server.send(nodes) #send intial model to nodes
        logger.info(f"--Computing round {k}--")
        for i,local in enumerate(nodes):
            logger.info(f"Computing epochs for local {local.id}")
            losses1, losses2 = local.compute_round(E) #computes E steps of local training
            append_in_dict(valid_losses[f"node_{i}"], losses1)
            append_in_dict(valid_losses2[f"node_{i}"], losses2)
        logger.info(f"Aggregating") 
        server.receive(nodes) #averages updates 
    logger.info(f"Finished")

    logger.info('End of script\n')

if __name__ == "__main__":
    run()



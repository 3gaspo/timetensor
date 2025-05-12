import hydra
import logging
import os
import torch
import torch.nn as nn
import numpy as np
import copy

from src.timetensor.dataset import get_train_loaders, aggregate_loaders
from src.timetensor.models import load_model
from src.timetensor.pipeline import Learner, train_model, Loss
from src.timetensor.federated import get_client_splits, Client, get_node_metrics, eval_nodes, average_nodes
from src.timetensor.utils import save_results, append_in_dict, get_dirs
from src.timetensor.visu import plot_losses, plot_multi_losses

from src.timetensor.fedavg import LocalFedAvg, GlobalFedAvg, FedAvgScheme
from src.timetensor.fedrevin import LocalFedRevin, FedRevinScheme


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
    criterion_name, normalization = cfg.training.loss, cfg.model.normalization
    model_name, retrain = cfg.model.name, cfg.training.retrain
    kwargs = cfg.model_configs
    verbose, benchmark = cfg.misc.verbose, cfg.misc.benchmark
    splits = cfg.fed.splits
    N = len(splits)    
    assert model_name not in ["persistence", "repeat", "lookback", "sklinear"]
    if verbose:
        logger.info("Fetched main configs")
        logger.info(f"Model {model_name}, normalization {normalization}, kwargs {kwargs}")
        logger.info(f"Found {N} splits")

    #save dirs
    output_dir = cfg.misc.output_dir
    save_name = cfg.misc.save_name
    save_name, save_dir = get_dirs(output_dir, save_name, model_name, normalization, criterion_name)
    if not os.path.exists(save_dir + "nodes/"):
        os.makedirs(save_dir + "nodes/")
    if verbose:
        logger.info(f"Save directory : {save_dir}")

    #criterion
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if criterion_name == "MSE":
        criterion = Loss(nn.MSELoss())
    elif criterion_name == "MMSE":
        criterion = Loss(nn.MSELoss(), cfg.data.mean, cfg.data.std)
    elif criterion_name == "NMSE":
        criterion = Loss(nn.MSELoss(), instance_norm=True)
    else:
        print("Unknown criterion name")
        criterion = None
    eval_losses = {
        "MSE":Loss(nn.MSELoss(reduction="none")),
        "MMSE":Loss(nn.MSELoss(reduction="none"), cfg.data.mean, cfg.data.std),
        "NMSE": Loss(nn.MSELoss(reduction="none"), instance_norm=True),
        "MAE": Loss(nn.L1Loss(reduction="none")),
        "NMAE": Loss(nn.L1Loss(reduction="none"), instance_norm=True)
        }

    #nodes
    node_data_dict = get_client_splits(data_path, splits)
    nodes, sizes = [], []
    shadow_nodes = [] #performing only local training
    all_loaders = []
    for k in range(N):
        #data
        path = data_path + f"nodes/node_{k}/"
        loaders_dict = get_train_loaders(node_data_dict[f"node_{k}"], batch_size, lags, horizon, by_date=True, subsets=cfg.fed.subsets[f"node{k}"], path=path)
        all_loaders.append(loaders_dict)
        client = Client(loaders_dict, id=k)
        shadow_client =  Client(loaders_dict, id=-k)
        dataloaders = client.dataloaders
        X, c, y = next(iter(dataloaders["train"]))
        shape = [X.shape[2], X.shape[1], y.shape[2]]

        #learner
        model = load_model(model_name, shape, normalization, **kwargs)
        learner = Learner(model, criterion, lr, eval_losses, device=device)
        shadow_learner = Learner(copy.deepcopy(model), criterion, lr, eval_losses, device=device)    
        if normalization == 3:
            node = LocalFedRevin(client, learner)
            shadow_node = LocalFedRevin(shadow_client, shadow_learner)
        else:
            node = LocalFedAvg(client, learner)
            shadow_node = LocalFedAvg(shadow_client, shadow_learner)
        nodes.append(node)
        shadow_nodes.append(shadow_node)
        sizes.append(client.get_size())
            
    size_weights = np.array(sizes) / np.sum(sizes)

    #server
    global_model = load_model(model_name, shape, normalization, **kwargs)
    global_shadow_learner = Learner(copy.deepcopy(global_model), criterion, lr, eval_losses, device=device)
    server_client = Client({key: aggregate_loaders([all_loaders[k][key] for k in range(N)]) for key in ["train","valid", "test"]}, id="server")
    shadow_server = LocalFedAvg(server_client, global_shadow_learner)#TO DO aggregated_client, shadow_learner)
    server = GlobalFedAvg(global_model)
    logger.info("Built all nodes")

    #FedAvg
    E, K = cfg.fed.epochs, cfg.fed.rounds
    if retrain:
        if normalization == 3:
            scheme = FedRevinScheme(server, nodes, shadow_server, shadow_nodes, cfg.fed.reset_revin)
        else:
            scheme = FedAvgScheme(server, nodes, shadow_server, shadow_nodes)

        logger.info("Starting training...")
        valid_losses, shadow_valid_losses, global_valid_losses = scheme.compute_scheme(K, E, plus=True)
        logger.info(f"Finished")

        #save losses
        torch.save(valid_losses, save_dir + f"node_valid_losses.pt")
        torch.save(shadow_valid_losses, save_dir + f"shadow_node_valid_losses.pt")
        
        for k in range(N):
            path = save_dir + f"nodes/node_{k}/"
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(nodes[k].client.get_weights(), path + "trained_model.pt")
            torch.save(shadow_nodes[k].client.get_weights(), path + "shadow_trained_model.pt")
        torch.save(global_valid_losses, save_dir + f"global_valid_losses.pt")
        torch.save(server.update, save_dir + "trained_model.pt")
        torch.save(shadow_server.client.get_weights(), save_dir + "shadow_trained_model.pt")
    else:
        valid_losses = torch.load(save_dir + f"node_valid_losses.pt", weights_only=False)
        shadow_valid_losses = torch.load(save_dir + f"shadow_node_valid_losses.pt", weights_only=False)
        for k in range(N):
            path = save_dir + f"node_{k}/"
            nodes[k].receive(torch.load( path + "trained_model.pt"))
            shadow_nodes[k].receive(torch.load( path + "shadow_trained_model.pt"))
        global_valid_losses = torch.load(save_dir + f"global_valid_losses.pt",weights_only=False)
        shadow_server.receive(torch.load(save_dir + "shadow_trained_model.pt"))
        global_model.load_state_dict(torch.load(save_dir + "trained_model.pt"))

    avg_losses = average_nodes(valid_losses)
    mean_losses =  average_nodes(valid_losses, size_weights)
    shadow_avg_losses = average_nodes(shadow_valid_losses)
    shadow_mean_losses =  average_nodes(shadow_valid_losses, size_weights)
    
    #plots
    plot_nodes=False
    if plot_nodes:
        for k in range(N):
            path = save_dir + f"nodes/node_{k}/"
            for key in eval_losses:
                plot_multi_losses({
                    "valid": valid_losses[f"node_{k}"][key], "shadow valid": shadow_valid_losses[f"node_{k}"][key]},
                    path, f"valid_{key}.pdf", f"Training {key} of {save_name}, node_{k}", x_every=E)
    for key in eval_losses:
        plot_multi_losses({
            f"mean valid {key}": mean_losses[key],
            f"shadow mean valid {key}": shadow_mean_losses[key],
            f"global valid {key}": global_valid_losses[key]},
            save_dir + "plots/", f"valid_{key}.pdf", f"Training {key} of {save_name}", x_every=E)
        plot_multi_losses({
            f"avg valid {key}": avg_losses[key],
            f"shadow avg valid {key}": shadow_avg_losses[key]},
            save_dir + "plots/", f"avg_valid_{key}.pdf", f"Training {key} of {save_name}", x_every=E)

    #eval
    logger.info("Computing test metrics")
    losses_dict = eval_nodes(nodes)
    avg_loss_dict, mean_losses_dict, flop_losses_dict = get_node_metrics(losses_dict, size_weights)
    if benchmark:
        test_dir = output_dir
    else:
        test_dir = save_dir
    for loss_name in eval_losses:
        save_results(avg_loss_dict[loss_name], test_dir, f"{loss_name}_mean_results.json", save_name, f"Uniform {loss_name}")
        save_results(mean_losses_dict[loss_name], test_dir, f"{loss_name}_mean_results.json", save_name, f"Weighted {loss_name}")
        save_results(flop_losses_dict[loss_name], test_dir, f"{loss_name}_mean_results.json", save_name, f"Flop10 {loss_name}")

    logger.info('End of script\n')

if __name__ == "__main__":
    run()



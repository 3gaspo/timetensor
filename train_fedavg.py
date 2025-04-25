import hydra
import logging
import os
import torch
import torch.nn as nn
import numpy as np
import copy

from src.timetensor.dataset import get_train_loaders
from src.timetensor.models import load_model
from src.timetensor.pipeline import Learner, train_model
from src.timetensor.federated import get_client_splits, Client
from src.timetensor.utils import save_results, append_in_dict, get_node_metrics, eval_nodes, get_dirs
from src.timetensor.fedavg import LocalFedAvg, GlobalFedAvg
from src.timetensor.visu import plot_losses, plot_multi_losses
from src.timetensor.dataset import aggregate_loaders

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
    save_name, save_dir = get_dirs(output_dir, save_name, model_name, normalization)
    if verbose:
        logger.info("Fetched output directories")
        logger.info(f"Save directory : {save_dir}")

    #nodes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if criterion_name == "MSE":
        criterion = nn.MSELoss()
    else:
        print("Unknown criterion name")
        criterion = None
    eval_losses = {"MSE":lambda pred, y: nloss(nn.MSELoss(reduction="none"), pred, y, cfg.data.mean, cfg.data.std)}

    node_data_dict = get_client_splits(data_path, splits)
    nodes = []
    sizes = []
    shadow_nodes = [] #performing only local training
    for k in range(N):

        #data
        path = data_path + f"node_{k}/"
        loaders_dict = get_train_loaders(node_data_dict[f"node_{k}"], batch_size, lags, horizon, by_date=True, subsets=cfg.data.subset, path=path)
        client = Client(loaders_dict, id=k)
        if retrain:
            shadow_client =  Client(loaders_dict, id=-k)
        dataloaders = client.dataloaders
        logger.info(f"Client_id={client.id} | train={dataloaders['train'].dataset.shape} valid={dataloaders['valid'].dataset.shape} test={dataloaders['test'].dataset.shape}")

        if k==0:
            X, c, y = next(iter(dataloaders["train"]))
            shape = [X.shape[2], X.shape[1], y.shape[2]]

        #learner        
        if retrain:
            model = load_model(model_name, shape, normalization, **kwargs)
            learner = Learner(model, criterion, lr, eval_losses, device=device)
            shadow_learner = Learner(copy.deepcopy(model), criterion, lr, eval_losses, device=device)
        else:
            model = load_model(model_name, shape, normalization, **kwargs)
            learner = Learner(model, criterion, lr, eval_losses, device=device)
            model.load_state_dict(torch.load(save_dir + f"node_{k}/" + "trained_model.pt"))
            model.to(device)

        node = LocalFedAvg(client, learner)
        sizes.append(node.client.get_size()) 
        nodes.append(node)
        if retrain:
            shadow_node = LocalFedAvg(shadow_client, shadow_learner)
            shadow_nodes.append(shadow_node)
    size_weights = np.array(sizes) / np.sum(sizes)

    #server
    global_model = load_model(model_name, shape, normalization, **kwargs)
    if retrain:
        shadow_learner = Learner(copy.deepcopy(global_model), criterion, lr, eval_losses, device=device)
        server_client = Client({ for key in }, id="server")
        shadow_server = LocalFedAvg()#TO DO aggregated_client, shadow_learner)
    else:
        global_model.load_state_dict(torch.load(save_dir + "trained_model.pt"))
    server = GlobalFedAvg(global_model)

    #FedAvg
    E, K = cfg.fed.epochs, cfg.fed.rounds 
    valid_losses = {f"node_{k}": {} for k in range(N)}
    valid_losses2 = {f"node_{k}": {} for k in range(N)}
    shadow_valid_losses = {f"node_{k}": {} for k in range(N)}
    shadow_valid_losses2 = {f"node_{k}": {} for k in range(N)}
    global_valid_losses = {}
    global_valid_losses2 = {}
    if retrain:
        logger.info(f"==Starting FedAvg==")
        logger.info(f"epochs={E}, rounds={K}")

        for k in range(K):
            server.send(nodes) #send intial model to nodes
            logger.info(f"--Computing round {k}--")
            shadow_losses1, shadow_losses2 = shadow_server.compute_round(E)
            append_in_dict(global_valid_losses, shadow_losses1)
            append_in_dict(global_valid_losses2, shadow_losses2)
            for i,local in enumerate(nodes):
                logger.info(f"Computing epochs for local {local.id}")
                losses1, losses2 = local.compute_round(E) #computes E steps of local training
                shadow_losses1, shadow_losses2 = shadow_nodes[k].compute_round(E)
                append_in_dict(valid_losses[f"node_{i}"], losses1)
                append_in_dict(valid_losses2[f"node_{i}"], losses2)
                append_in_dict(shadow_valid_losses[f"node_{i}"], shadow_losses1)
                append_in_dict(shadow_valid_losses2[f"node_{i}"], shadow_losses2)
            logger.info(f"Aggregating")
            server.receive(nodes) #averages updates 
        server.send(nodes)
        #add a local epoch for FedAvg+
        logger.info(f"Finished")

    #save losses
        for k in range(N):
            path = save_dir + f"node_{k}/"
            if not os.path.exists(path):
                os.makedirs(path)
            for loss_name in eval_losses:
                torch.save(valid_losses[f"node_{k}"][loss_name], path + f"{loss_name}_valid_losses.pt")
                torch.save(valid_losses2[f"node_{k}"][loss_name], path + f"{loss_name}_valid_losses2.pt")
                torch.save(shadow_valid_losses[f"node_{k}"][loss_name], path + f"shadow_{loss_name}_valid_losses.pt")
                torch.save(shadow_valid_losses2[f"node_{k}"][loss_name], path + f"{shadow_loss_name}_valid_losses2.pt")
            torch.save(nodes[k].client.get_weights(), path + "trained_model.pt")
            torch.save(shadow_nodes[k].client.get_weights(), path + "shadow_trained_model.pt")
        torch.save(global_valid_losses[loss_name], save_dir + f"shadow_{loss_name}_valid_losses.pt")
        torch.save(global_valid_losses2[loss_name], save_dir + f"{shadow_loss_name}_valid_losses2.pt")
        torch.save(server.update, save_dir + "trained_model.pt")
        torch.save(shadow_server.client.get_weights(), save_dir + "shadow_trained_model.pt")

    else:
        for k in range(N):
            path = save_dir + f"node_{k}/"
            for loss_name in eval_losses:
                valid_losses[f"node_{k}"][loss_name] = torch.load(path + f"{loss_name}_valid_losses.pt",weights_only=False)
                valid_losses2[f"node_{k}"][loss_name] = torch.load(path + f"{loss_name}_valid_losses2.pt",weights_only=False)
                shadow_valid_losses[f"node_{k}"][loss_name] = torch.load(path + f"shadow_{loss_name}_valid_losses.pt",weights_only=False)
                shadow_valid_losses2[f"node_{k}"][loss_name] = torch.load(path + f"shadow_{loss_name}_valid_losses2.pt",weights_only=False)
        global_valid_losses[loss_name] = torch.load(save_dir + f"shadow_{loss_name}_valid_losses.pt",weights_only=False)
        global_valid_losses2[loss_name] = torch.load(save_dir + f"shadow_{loss_name}_valid_losses2.pt",weights_only=False)
    avg_losses = average_nodes(valid_losses)
    avg_losses2 = average_nodes(valid_losses2)
    mean_losses =  average_nodes(valid_losses, size_weights)
    mean_losses2 = average_nodes(valid_losses2, size_weights)
    shadow_avg_losses = average_nodes(shadow_valid_losses)
    shadow_avg_losses2 = average_nodes(shadow_valid_losses2)
    shadow_mean_losses =  average_nodes(shadow_valid_losses, size_weights)
    shadow_mean_losses2 = average_nodes(shadow_valid_losses2, size_weights)
    
    #plots
    for k in range(N):
        path = save_dir + f"node_{k}/"
        plot_multi_losses({
            "valid 1 MSE": valid_losses[f"node_{k}"]["MSE"],
            "valid 2 MSE": valid_losses2[f"node_{k}"]["MSE"],
            "shadow valid 1 MSE": shadow_valid_losses[f"node_{k}"]["MSE"],
            "shadow valid 2 MSE": shadow_valid_losses2[f"node_{k}"]["MSE"]},
            path, f"valid_losses.pdf", f"Training MSE of {save_name},node_{k}", x_every=E)
    plot_multi_losses({
            "valid 1 MSE": valid_avg_losses[f"node_{k}"]["MSE"],
            "valid 2 MSE": valid_avg_losses2[f"node_{k}"]["MSE"],
            "shadow valid 1 MSE": shadow_avg_valid_losses[f"node_{k}"]["MSE"],
            "shadow valid 2 MSE": shadow_avg_valid_losses2[f"node_{k}"]["MSE"],
            "global valid 1 MSE": global_valid_losses1["MSE"],
            "global valid 2 MSE": global_valid_losses2["MSE"],
            },
            save_dir, f"avg_valid_losses.pdf", f"Training avg MSE of {save_name}", x_every=E)
    plot_multi_losses({
            "valid 1 MSE": valid_mean_losses[f"node_{k}"]["MSE"],
            "valid 2 MSE": valid_mean_losses2[f"node_{k}"]["MSE"],
            "shadow valid 1 MSE": shadow_mean_valid_losses[f"node_{k}"]["MSE"],
            "shadow valid 2 MSE": shadow_mean_valid_losses2[f"node_{k}"]["MSE"],
            "global valid 1 MSE": global_valid_losses1["MSE"],
            "global valid 2 MSE": global_valid_losses2["MSE"],
            },
            save_dir, f"mean_valid_losses.pdf", f"Training mean MSE of {save_name}", x_every=E)

    #eval
    logger.info("Computing test metrics")
    mse_means = []
    nmse_means = []
    losses_dict = eval_nodes(nodes)
    mean_losses_dict, flop_losses_dict = get_node_metrics(losses_dict)
    for k in range(N):
        path = save_dir + f"node_{k}/"
        test_losses = nodes[k].eval() #(ndates*nindividuals, dim, horizon)
        torch.save(test_losses, path + "test_losses.pt")
        mean_test_mse = test_losses["MSE"]
        mean_test_nmse = test_losses["NMSE"]
        mse_means.append(mean_test_mse)
        nmse_means.append(mean_test_nmse)
        save_results(mean_test_mse, path, "mean_results.json", save_name, "Test MSE")
        save_results(mean_test_nmse, path, "mean_results.json", save_name, "Test NMSE")

    mean_mse = np.mean(mse_means)
    mean_nmse = np.mean(nmse_means)
    avg_mse = np.avg(mse_means, weights=size_weights)
    avg_nmse = np.avg(nmse_means, weights=size_weights)
    if benchmark:
        test_dir = output_dir
    else:
        test_dir = save_dir
    save_results(mean_mse, test_dir, "mean_results.json", save_name, "Uniform MSE")
    save_results(mean_nmse, test_dir, "mean_results.json", save_name, "Uniform NMSE")
    save_results(avg_mse, test_dir, "mean_results.json", save_name, "Weighted MSE")
    save_results(avg_nmse, test_dir, "mean_results.json", save_name, "Weighted NMSE")

    logger.info(f"Uniform average: MSE={mean_mse:.2f}, NMSE={mean_nmse:.2f} | Sized average: MSE={avg_mse:.2f}, NMSE={avg_nmse:.2f}")

    logger.info('End of script\n')

if __name__ == "__main__":
    run()



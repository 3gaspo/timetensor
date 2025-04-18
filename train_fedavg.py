import hydra
import logging
import os
import torch
import torch.nn as nn
import numpy as np

from src.timetensor.dataset import get_train_loaders
from src.timetensor.models import load_model
from src.timetensor.pipeline import Learner, train_model
from src.timetensor.federated import get_client_splits, Client
from src.timetensor.utils import save_results, append_in_dict
from src.timetensor.fedavg import LocalFedAvg, GlobalFedAvg
from src.timetensor.visu import plot_losses, plot_multi_losses

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
            if normalization == 1:
                save_name = save_name + f"_normal_train"
            elif normalization == 2:
                save_name = save_name + f"_normal_instance"
            elif normalization == 3:
                save_name = save_name + f"_normal_revin"  
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if criterion_name == "MSE":
        criterion = nn.MSELoss()
    else:
        print("Unknown criterion name")
        criterion = None
    eval_losses = {"MSE":nn.MSELoss(reduction="none")}

    node_data_dict = get_client_splits(data_path, splits)
    nodes = []
    for k in range(N):

        #data
        path = data_path + f"node_{k}/"
        loaders_dict = get_train_loaders(node_data_dict[f"node_{k}"], batch_size, lags, horizon, by_date=True, subsets=cfg.data.subset, path=path)
        client = Client(loaders_dict, id=k)

        dataloaders = client.dataloaders
        logger.info(f"Client_id={client.id} | train={dataloaders['train'].dataset.shape} valid={dataloaders['valid'].dataset.shape} test={dataloaders['test'].dataset.shape}")

        if k==0:
            X, c, y = next(iter(dataloaders["train"]))
            shape = [X.shape[1], X.shape[2], y.shape[2]]

        #learner        
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
    global_model = load_model(model_name, shape, normalization, **kwargs)
    if not retrain:
        global_model.load_state_dict(torch.load(save_dir + "trained_model.pt"))
    server = GlobalFedAvg(global_model)

    #FedAvg
    E, K = 3, 5 
    valid_losses = {f"node_{k}": {} for k in range(N)}
    valid_losses2 = {f"node_{k}": {} for k in range(N)}
    if retrain:
        logger.info(f"==Starting FedAvg==")
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
        server.send(nodes)
        logger.info(f"Finished")
    
    #save losses
        for k in range(N):
            path = save_dir + f"node_{k}/"
            if not os.path.exists(path):
                os.makedirs(path)
            for loss_name in eval_losses:
                torch.save(valid_losses[f"node_{k}"][loss_name], path + f"{loss_name}_valid_losses.pt")
                torch.save(valid_losses[f"node_{k}"]["N"+loss_name], path + f"N{loss_name}_valid_losses.pt")
                torch.save(valid_losses2[f"node_{k}"][loss_name], path + f"{loss_name}_valid_losses2.pt")
                torch.save(valid_losses2[f"node_{k}"]["N"+loss_name], path + f"N{loss_name}_valid_losses2.pt")
            torch.save(nodes[k].client.get_weights(), path + "trained_model.pt")
        torch.save(server.update, save_dir + "trained_model.pt")
    else:
        for k in range(N):
            path = save_dir + f"node_{k}/"
            for loss_name in eval_losses:
                valid_losses[f"node_{k}"][loss_name] = torch.load(path + f"{loss_name}_valid_losses.pt",weights_only=False)
                valid_losses[f"node_{k}"]["N"+loss_name] = torch.load(path + f"N{loss_name}_valid_losses.pt",weights_only=False)
                valid_losses2[f"node_{k}"][loss_name] = torch.load(path + f"{loss_name}_valid_losses2.pt",weights_only=False)
                valid_losses2[f"node_{k}"]["N"+loss_name] = torch.load(path + f"N{loss_name}_valid_losses2.pt",weights_only=False)

    #plots
    for k in range(N):
        path = save_dir + f"node_{k}/"
        plot_multi_losses({"valid 1 MSE": valid_losses[f"node_{k}"]["MSE"],"valid 2 MSE": valid_losses2[f"node_{k}"]["MSE"]},  path, f"valid_losses.pdf", f"Training MSE of {save_name},node_{k}", x_every=E)
        plot_multi_losses({"valid 1 NMSE": valid_losses[f"node_{k}"]["NMSE"],"valid 2 NMSE": valid_losses2[f"node_{k}"]["NMSE"]},  path, f"nvalid_losses.pdf", f"Training NMSE of {save_name},node_{k}",  x_every=E)
    

    #eval
    logger.info("Computing test metrics")
    mse_means = []
    nmse_means = []
    sizes = []
    for k in range(N):
        path = save_dir + f"node_{k}/"
        test_losses = nodes[k].eval() #(ndates*nindividuals, dim, horizon)
        torch.save(test_losses, path + "test_losses.pt")
        mean_test_mse = test_losses["MSE"]
        mean_test_nmse = test_losses["NMSE"]
        mse_means.append(mean_test_mse)
        nmse_means.append(mean_test_nmse)
        sizes.append(nodes[k].client.get_size()) 
        save_results(mean_test_mse, path, "mean_results.json", save_name, "Test MSE")
        save_results(mean_test_nmse, path, "mean_results.json", save_name, "Test NMSE")

    mean_mse = np.mean(mse_means)
    mean_nmse = np.mean(nmse_means)
    avg_mse = np.sum(np.array(mse_means) * np.array(sizes)) / np.sum(sizes)
    avg_nmse = np.sum(np.array(nmse_means) * np.array(sizes)) / np.sum(sizes)
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



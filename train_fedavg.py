import hydra
import logging
import os
import torch
import torch.nn as nn
import numpy as np
import copy

from src.timetensor.dataset import get_train_loaders, aggregate_loaders
from src.timetensor.models import load_model
from src.timetensor.pipeline import Learner, Loss
from src.timetensor.federated import get_client_splits, Client, get_node_metrics, eval_nodes, average_nodes
from src.timetensor.utils import save_results, get_dirs, fetch_example_data, unroll_windows
from src.timetensor.visu import plot_multi_losses, plot_pred, plot_weights, plot_stats

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
    model_name, retrain, kwargs = cfg.model.name, cfg.training.retrain, cfg.model_configs
    verbose, benchmark = cfg.misc.verbose, cfg.misc.benchmark
    nodes, splits, subsets = cfg.fed.nodes, cfg.fed.splits, cfg.fed.subsets
    assert model_name not in ["persistence", "repeat", "lookback", "sklinear", "expected"], "Unsupported model for FedAvg"
    if verbose:
        logger.info("Fetched main configs")
        logger.info(f"Model {model_name}, normalization {normalization}, criterion {criterion_name}, kwargs {kwargs}")
        if nodes==None:
            logger.info(f"Found all users as nodes")
        elif type(nodes)==int:
            logger.info(f"Found {nodes} nodes")
        else:
            logger.info(f"Found a path to nodes at {nodes}")

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
        criterion = Loss(nn.MSELoss(), mode="instance")
    elif criterion_name == "RMSE":
        criterion = Loss(nn.MSELoss(), mode="relative")
    else:
        print("Unknown criterion name")
        criterion = None
    eval_losses = {
        "MMSE": Loss(nn.MSELoss(reduction="none"), cfg.data.mean, cfg.data.std),
        "RMSE": Loss(nn.MSELoss(reduction="none"), mode="relative"),
        "MMAE": Loss(nn.L1Loss(reduction="none"), cfg.data.mean, cfg.data.std)
        }

    #nodes
    node_data_dict = get_client_splits(data_path, nodes, splits)
    N = len(node_data_dict)
    if verbose:
        logger.info(f"Loaded {N} node splits")

    nodes, sizes = [], []
    shadow_nodes = [] #performing only local training
    all_loaders = []
    for k in range(N):
        #data
        loaders_dict = get_train_loaders(node_data_dict[f"node_{k}"], batch_size, lags, horizon, by_date=True, subsets=subsets, subset_mode="dates")
        all_loaders.append(loaders_dict)
        client = Client(loaders_dict, id=k)
        shadow_client =  Client(loaders_dict, id=-k)
        X, c, y = next(iter(client.dataloaders["train"]))
        shape = [X.shape[2], X.shape[1], y.shape[2]]

        #learner
        model = load_model(model_name, shape, normalization, **kwargs)
        learner = Learner(model, criterion, lr, eval_losses, device=device)
        shadow_learner = Learner(copy.deepcopy(model), criterion, lr, eval_losses, device=device)    
        if "revin" in normalization:
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
    shadow_server = LocalFedAvg(server_client, global_shadow_learner)
    server = GlobalFedAvg(global_model)
    logger.info("Built all nodes")

    #FedAvg
    E, K = cfg.fed.epochs, cfg.fed.rounds
    if retrain:
        if "revin" in normalization:
            reset_revin = cfg.fed.reset_revin
            logger.info(f"Loading FedRevin scheme with {reset_revin}")
            scheme = FedRevinScheme(server, nodes, shadow_server, shadow_nodes, reset_revin)
        else:
            logger.info(f"Loading FedAvg scheme")
            scheme = FedAvgScheme(server, nodes, shadow_server, shadow_nodes)

        logger.info("Starting training...")
        valid_losses, shadow_valid_losses, global_valid_losses = scheme.compute_scheme(K, E, plus=True, verbose=True)
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
            path = save_dir + f"nodes/node_{k}/"
            nodes[k].receive(torch.load(path + "trained_model.pt"))
            shadow_nodes[k].receive(torch.load( path + "shadow_trained_model.pt"))
        global_valid_losses = torch.load(save_dir + f"global_valid_losses.pt",weights_only=False)
        shadow_server.receive(torch.load(save_dir + "shadow_trained_model.pt")) #final global model
        global_model.load_state_dict(torch.load(save_dir + "trained_model.pt")) #final FedAvg model

    avg_losses = average_nodes(valid_losses)
    mean_losses =  average_nodes(valid_losses, size_weights)
    shadow_avg_losses = average_nodes(shadow_valid_losses)
    shadow_mean_losses =  average_nodes(shadow_valid_losses, size_weights)
    
    #plots
    if N<=10:
        plot_nodes=True
    else:
        plot_nodes=False
    if plot_nodes:
        for k in range(N):
            path = save_dir + f"nodes/node_{k}/"
            for key in eval_losses:
                plot_multi_losses({
                    "valid": valid_losses[f"node_{k}"][key], "shadow valid": shadow_valid_losses[f"node_{k}"][key]},
                    path, f"valid_{key}.pdf", f"Training {key} of {save_name}, node_{k}", x_every=E)
    for key in eval_losses: #TODO: if global=partial, aura pas le bon nombre de points (cf train/valid: ajouter un dict de train et un dict de valid dans params de plot function)
        plot_multi_losses({
            f"mean valid {key}": mean_losses[key],
            f"shadow mean valid {key}": shadow_mean_losses[key],
            f"global valid {key}": global_valid_losses[key]},
            save_dir + "plots/", f"valid_{key}.pdf", f"Training {key} of {save_name}", x_every=E)
        plot_multi_losses({
            f"avg valid {key}": avg_losses[key],
            f"shadow avg valid {key}": shadow_avg_losses[key]},
            save_dir + "plots/", f"avg_valid_{key}.pdf", f"Training {key} of {save_name}", x_every=E)

    #examples
    dico = fetch_example_data(data_path + "examples/", [f"ex{k}" for k in range(1,6)])
    for data_name, data_tuple in dico.items():
        x, c, y = data_tuple[0].unsqueeze(0).to(device), data_tuple[1], data_tuple[2].unsqueeze(0).to(device)
        if c is not None:
            c = c.unsqueeze(0).to(device)
        if not retrain:
            global_model.to(device)
        pred = global_model(x,c)
        plot_pred(x[0,0].cpu().detach().numpy(), y[0,0].cpu().detach().numpy(), pred[0,0].cpu().detach().numpy(), save_dir + "examples/", f"{data_name}_predictions.pdf", f"Example {data_name} prediction for {save_name}")        
    logger.info('Saved plots')

    #weights
    if model_name == "DLinear":
        if normalization is not None:
            linear_weights = global_model.model.Linear_Seasonal[0].weight.detach().cpu().numpy()
            season_weights = global_model.model.Linear_Trend[0].weight.detach().cpu().numpy()
        else:
            linear_weights = global_model.Linear_Seasonal[0].weight.detach().cpu().numpy()
            season_weights = global_model.Linear_Trend[0].weight.detach().cpu().numpy()
        plot_weights(linear_weights, save_dir + "plots/", name="season_weights.pdf", title=f'{save_name} seasonal weights')
        plot_weights(season_weights, save_dir + "plots/", name="trend_weights.pdf", title=f'{save_name} trend weights')

    #revin
    if "revin" in normalization:
        betas, gammas = [], []
        for k in range(N):
            beta = nodes[k].learner.model.beta.data.detach().cpu().numpy()[0][0][0]
            gamma = nodes[k].learner.model.gamma.data.detach().cpu().numpy()[0][0][0]
            betas.append(beta)
            gammas.append(gamma)
            params = {"beta": beta, "gamma": gamma}
            logger.info(f"Final revin parameters: {params}")
        if N<10:
            y_dict = {f"node_{k}": unroll_windows(nodes[k].client.dataloaders["train"], normal=True)[1] for k in range(N)}
            ry_dict = {f"node_{k}": unroll_windows(nodes[k].client.dataloaders["train"], normal=True, beta=betas[k], gamma=gammas[k])[1] for k in range(N)}
            plot_stats(y_dict, save_dir + "plots/", name="normal_outputs_nodes.pdf", title="Normalized outputs distribution", logscale=False, limits=(-1e-6,1e-6))
            plot_stats(ry_dict, save_dir + "plots/", name="revin_outputs_nodes.pdf", title="Normalized outputs distribution", logscale=False, limits=(-1e-6,1e-6))

    #eval
    logger.info("Computing test metrics")
    tune_losses_dict = eval_nodes(nodes)
    global_losses_dict = shadow_server.eval()
    tune_avg_loss_dict, tune_mean_losses_dict, tune_flop_losses_dict = get_node_metrics(tune_losses_dict, size_weights)

    for k in range(N):
        flat_losses_dict = eval_nodes(nodes, global_model.state_dict())
        flat_avg_loss_dict, flat_mean_losses_dict, flat_flop_losses_dict = get_node_metrics(flat_losses_dict, size_weights)

    if benchmark:
        test_dir = output_dir
    else:
        test_dir = save_dir
    for loss_name in eval_losses:
        save_results(global_losses_dict[loss_name], test_dir, f"{loss_name}_mean_results.json", save_name, f"Global {loss_name}")
        save_results(tune_avg_loss_dict[loss_name], test_dir, f"{loss_name}_mean_results.json", save_name, f"Tuned Uniform {loss_name}")
        save_results(tune_mean_losses_dict[loss_name], test_dir, f"{loss_name}_mean_results.json", save_name, f"Tuned Weighted {loss_name}")
        save_results(tune_flop_losses_dict[loss_name], test_dir, f"{loss_name}_mean_results.json", save_name, f"Tuned Flop10 {loss_name}")
        save_results(flat_avg_loss_dict[loss_name], test_dir, f"{loss_name}_mean_results.json", save_name, f"Flat Uniform {loss_name}")
        save_results(flat_mean_losses_dict[loss_name], test_dir, f"{loss_name}_mean_results.json", save_name, f"Flat Weighted {loss_name}")
        save_results(flat_flop_losses_dict[loss_name], test_dir, f"{loss_name}_mean_results.json", save_name, f"Flat Flop10 {loss_name}")
    logger.info('End of script\n')

if __name__ == "__main__":
    run()



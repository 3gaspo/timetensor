import hydra
import logging
import os
import torch
import numpy as np
import json
import copy

from src.timetensor.dataset import fetch_training_data, get_sizes, set_random_data, fetch_example_data, fetch_stats
from src.timetensor.models import load_model
from src.timetensor.pipeline import get_losses, launch_training
from src.timetensor.visu import plot_errors, plot_horizon_errors, plot_pred, plot_weights, plot_named_example
from src.timetensor.utils import save_results, get_dirs

from src.timetensor.dataset import get_train_loaders, aggregate_loaders_dict
from src.timetensor.pipeline import Learner
from src.timetensor.federated import get_client_splits, eval_nodes, get_node_metrics
from src.timetensor.fedavg import LocalFedAvg, GlobalFedAvg, FedAvgScheme

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    logger.info("=====Running train script=====")

    #configs
    data_path = cfg.data.path
    subsets = cfg.data.subsets
    lags, horizon = int(cfg.task.lags), int(cfg.task.horizon)
    batch_size, lr, epochs, criterion_name = cfg.training.bs, cfg.training.lr, cfg.training.epochs, cfg.training.loss
    
    retrain, init_path = cfg.training.retrain, cfg.training.init
    eval_freq, print_freq, complete_evaluation = cfg.training.eval_freq, cfg.training.print_freq, cfg.misc.complete_evaluation
    model_name, normalization, norm_kwargs, model_kwargs = cfg.model.name, cfg.normalization.name, cfg.normalization.configs, cfg.model.configs
    kwargs = {**(norm_kwargs or {}), **(model_kwargs or {})}

    clusters = cfg.data.clusters #if clusters is None, will split randomly users        
    splits = cfg.task.splits
    assert (clusters is not None or splits is not None)

    verbose, seed = cfg.misc.verbose, cfg.misc.seed
    if seed == "None":
        seed = None

    benchmark, output_dir, save_name = cfg.misc.benchmark, cfg.misc.output_dir, cfg.misc.save_name, 
    save_name, save_dir = get_dirs(output_dir, save_name, model_name, normalization, criterion_name, subsets["sizes"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion, eval_losses = get_losses(criterion_name, mean=None, std=None, complete_evaluation=complete_evaluation)

    if verbose:
        logger.info(f"Fetched main configs, save directory : {save_dir}")
        logger.info(f"Model {model_name}, normalization {normalization}, criterion {criterion_name}, kwargs {kwargs}")

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    #data
    stats_dict=None
    if clusters is not None:
        #stats_dict = fetch_stats(data_path, clusters, normalization, subsets)
        loaders_dicts = fetch_training_data(data_path,
            cfg.data.indiv_split, cfg.data.date_splits, subsets,
            batch_size, lags, horizon, by_date=(cfg.data.by_idx=="dates"), context_by_individuals=cfg.data.context_by_individuals,
            reshuffle=cfg.data.reshuffle, remove_cte=cfg.data.remove_cte,
            clusters=clusters, stats=stats_dict, seed=seed, aggregate=False)
    else:
        node_data_dict = get_client_splits(data_path, splits,
            cfg.data.indiv_split, cfg.data.date_splits, context_by_individuals=cfg.data.context_by_individuals,
            reshuffle=cfg.data.reshuffle)
        #TODO ajouter node stats dict
        loaders_dicts = {node: get_train_loaders(node_data_dict[node],
            batch_size, lags, horizon, by_date=(cfg.data.by_idx=="dates"), subsets=subsets["sizes"], subset_mode=subsets["mode"],
            save_path=data_path+"subsets/", remove_cte=cfg.data.remove_cte, stats=None)
            for node in node_data_dict}

    M = len(loaders_dicts)
    shapes = {node: loader['train'].dataset.shape for node, loader in loaders_dicts.items()}    
    shape = get_sizes(loaders_dicts["node0"])
    if verbose:
        logger.info("Fetched dataloaders")
        shape_str = "Splits shapes:\n" + "\n".join("{}\t{}".format(k, v) for k, v in shapes.items())
        logger.info(shape_str)

    #model
    global_model = load_model(model_name, shape, normalization, **kwargs)

    #training
    E, K, B = cfg.training.epochs, cfg.task.rounds, cfg.task.sampled_clients
    def client_builder(client, learner):
        return LocalFedAvg(client, learner)
    def server_builder(global_model):
        return GlobalFedAvg(global_model)
    def scheme_builder(server, nodes, shadow_server=None, shadow_nodes=None):
        return FedAvgScheme(E, K, B, server, nodes, shadow_server, shadow_nodes, plus=True, server_side="partial")
  
    server, shadow_server, nodes, shadow_nodes, size_weights = launch_training(client_builder, server_builder, scheme_builder, loaders_dicts, global_model,
        criterion, lr, eval_losses, device, logger,
        save_dir, retrain=True)

    #example
    global_model.load_state_dict(server.update)
    ex_dir = data_path + "examples/" + f"{lags}_{horizon}/"
    if not os.path.exists(ex_dir):
        set_random_data(data_path, lags, horizon, name="rand")
        plot_named_example(ex_dir, f"rand")
    dico = fetch_example_data(ex_dir)
    for data_name, data_tuple in dico.items():
        x, c, y = data_tuple[0].unsqueeze(0).to(device), data_tuple[1], data_tuple[2].unsqueeze(0).to(device)
        if c is not None:
            c = c.unsqueeze(0).to(device)
        pred = global_model(x,c)
        plot_pred(x[0,0].cpu().detach().numpy(), y[0,0].cpu().detach().numpy(), pred[0,0].cpu().detach().numpy(), save_dir + "examples/", f"{data_name}_predictions.pdf", f"Example {data_name} prediction for {save_name}")        
    logger.info('Saved plots')

    #eval
    if verbose:
        logger.info("Computing test metrics")
    tune_losses_dict = eval_nodes(nodes)
    global_losses_dict = shadow_server.eval()
    tune_avg_loss_dict, tune_mean_losses_dict, tune_flop_losses_dict = get_node_metrics(tune_losses_dict, size_weights)

    for k in range(M):
        flat_losses_dict = eval_nodes(nodes, global_model.state_dict())
        flat_avg_loss_dict, flat_mean_losses_dict, flat_flop_losses_dict = get_node_metrics(flat_losses_dict, size_weights)

    if benchmark:
        test_dir = output_dir + "errors/"
    else:
        test_dir = save_dir + "errors/"
    for loss_name in eval_losses:
        save_results(global_losses_dict[loss_name], test_dir, f"{loss_name}_mean_results.json", save_name, f"Global {loss_name}")
        save_results(tune_avg_loss_dict[loss_name], test_dir, f"{loss_name}_mean_results.json", save_name, f"Tuned Uniform {loss_name}")
        save_results(tune_mean_losses_dict[loss_name], test_dir, f"{loss_name}_mean_results.json", save_name, f"Tuned Weighted {loss_name}")
        save_results(tune_flop_losses_dict[loss_name], test_dir, f"{loss_name}_mean_results.json", save_name, f"Tuned Flop10 {loss_name}")
        save_results(flat_avg_loss_dict[loss_name], test_dir, f"{loss_name}_mean_results.json", save_name, f"Flat Uniform {loss_name}")
        save_results(flat_mean_losses_dict[loss_name], test_dir, f"{loss_name}_mean_results.json", save_name, f"Flat Weighted {loss_name}")
        save_results(flat_flop_losses_dict[loss_name], test_dir, f"{loss_name}_mean_results.json", save_name, f"Flat Flop10 {loss_name}")
   
    if verbose and ("revin" in normalization or "mIN" in normalization):
        global_params = {"beta": global_model.beta.data.detach().cpu().numpy()[0][0][0], "alpha": global_model.alpha.data.detach().cpu().numpy()[0][0][0]}
        logger.info(f"Final global modulations: {global_params}")
        if M<=10:
            for k in range(M):
                model = nodes[k].get_latest_weights()
                global_params = {"beta": model.beta.data.detach().cpu().numpy()[0][0][0], "alpha": model.alpha.data.detach().cpu().numpy()[0][0][0]}
                logger.info(f"Final global modulations: {global_params}")

    logger.info('End of script\n')

if __name__ == "__main__":
    run()



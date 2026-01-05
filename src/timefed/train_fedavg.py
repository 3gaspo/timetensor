import hydra
import logging
import torch

from src.timetensor.dataset import fetch_training_data, get_sizes, get_train_loaders, apply_stats
from src.timetensor.models import load_model
from src.timetensor.pipeline import get_losses, launch_example
from src.timetensor.utils import get_dirs, set_seed
from src.timetensor.analysis import get_dataset_stats

from src.timefed.federated import get_client_splits, launch_training, launch_eval
from src.timefed.fedavg import LocalFedAvg, GlobalFedAvg, FedAvgScheme

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    logger.info("=====Running train FL script=====")

    #configs
    data_path = cfg.data.path
    # subsets = cfg.data.subsets
    lags, horizon = int(cfg.task.lags), int(cfg.task.horizon)
    # batch_size, lr, epochs, criterion_name = cfg.training.bs, cfg.training.lr, cfg.training.epochs, cfg.training.loss
    
    # retrain, init_path = cfg.training.retrain, cfg.training.init
    # eval_freq, print_freq, complete_evaluation = cfg.training.eval_freq, cfg.training.print_freq, cfg.misc.complete_evaluation
    # complete_evaluation = cfg.misc.complete_evaluation
    # model_name, normalization, norm_kwargs, model_kwargs = cfg.model.name, cfg.normalization.name, cfg.normalization.configs, cfg.model.configs
    # kwargs = {**(norm_kwargs or {}), **(model_kwargs or {})}

    criterion_name = cfg.training.loss
    criterion, eval_losses = get_losses(criterion_name, complete_evaluation=cfg.training.complete_evaluation)

    model_name, norm_name = cfg.model.name, cfg.normalization.name
    if norm_name == "None":
        norm_name = None
    kwargs = {**(cfg.normalization.configs or {}), **(cfg.model.configs or {})}

    verbose, seed = cfg.misc.verbose, cfg.misc.seed

    clusters = cfg.data.clusters #if clusters is None, will split randomly users        
    splits = cfg.fl.splits
    assert (clusters is not None or splits is not None)

    # verbose, seed = cfg.misc.verbose, cfg.misc.seed
    # if seed == "None":
    #     seed = None

    # benchmark, output_dir, save_name = cfg.misc.benchmark, cfg.misc.output_dir, cfg.misc.save_name, 
    # save_name, save_dir = get_dirs(output_dir, save_name, model_name, normalization, criterion_name, subsets["sizes"])
    
    output_dir, save_name = cfg.misc.output_dir, cfg.misc.save_name, 
    save_name, save_dir = get_dirs(output_dir, save_name, model_name, norm_name, criterion_name, cfg.data.subsets.sizes)

    if verbose:
        logger.info(f"Fetched main configs, save directory : {save_dir}")
        logger.info(f"Model {model_name}, norm {norm_name}, criterion {criterion_name}, kwargs {kwargs}")
        logger.info(f"Clusters {clusters} splits {splits}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # criterion, eval_losses = get_losses(criterion_name, mean=None, std=None, complete_evaluation=complete_evaluation)

    # E, K, B, fedmin = cfg.training.epochs, cfg.task.rounds, cfg.task.sampled_clients, cfg.task.fedmin
    # if fedmin:
    #     from src.timetensor.fedmin import LocalFedmIN

    # if verbose:
    #     logger.info(f"Fetched main configs, save directory : {save_dir}")
    #     logger.info(f"Model {model_name}, normalization {normalization}, criterion {criterion_name}, kwargs {kwargs}")
    #     if fedmin:
    #         logger.info("Training FedmIN (local modulations)")
    #     if not fedmin:
    #         logger.info("Training FedAvg")

    # if seed is not None:
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed(seed)
    #     np.random.seed(seed)

    #data
    if clusters is not None:
        #stats_dict = fetch_stats(data_path, clusters, normalization, subsets)
        # loaders_dicts = fetch_training_data(data_path,
        #     cfg.data.indiv_split, cfg.data.date_splits, subsets,
        #     batch_size, lags, horizon, by_date=(cfg.data.by_idx=="dates"), context_by_individuals=cfg.data.context_by_individuals,
        #     reshuffle=cfg.data.reshuffle, remove_cte=cfg.data.remove_cte,
        #     clusters=clusters, stats=stats_dict, seed=seed, aggregate=False)
    
        loaders_dicts, _, nodes_stats_dict = fetch_training_data(
        data_path, cfg.data.splits, cfg.data.subsets, cfg.training.bs, lags, horizon,
        clusters=cfg.data.clustering.clusters, seed=seed, random_eval=cfg.training.random_eval)
        if cfg.data.normalize:
            for node, loaders_dict in loaders_dicts.items():        
                apply_stats(loaders_dict, nodes_stats_dict[node])
    else:
        node_data_dict = get_client_splits(data_path, splits,
            cfg.data.indiv_split, cfg.data.date_splits, context_by_individuals=cfg.data.context_by_individuals,
            reshuffle=cfg.data.reshuffle)
        loaders_dicts = {node: get_train_loaders(node_data_dict[node],
            cfg.training.batch_size, lags, horizon, cfg.data.splits, cfg.subsets,
            data_path+"subsets/", random_eval=cfg.training.random_eval)
            for node in node_data_dict}
        nodes_stats_dict = {}
        for node, loaders_dict in loaders_dicts.items():        
            node_dict = {subkey: loader.dataset.get_df() for subkey, loader in loaders_dict.items()}
            nodes_stats_dict[node] = get_dataset_stats(node_dict, lags, horizon, cfg.data.splits["remove_train_cte"], cfg.data.splits["remove_eval_cte"], save_path=None)
            if cfg.data.normalize:
                apply_stats(loaders_dict, nodes_stats_dict[node])


    M = len(loaders_dicts)
    # shape = get_sizes(loaders_dicts["node0"])
    # if verbose:
    #     logger.info("Fetched dataloaders")
    #     shape_str = "Splits shapes:\n" + "\n".join("{}\t{}".format(k, v) for k, v in shapes.items())
    #     logger.info(shape_str)

    shape, shape_str, batch_str = get_sizes(loaders_dicts["node0"], str_info=True)
    shapes = {node: loader['train'].dataset.shape for node, loader in loaders_dicts.items()}
    if verbose:
        logger.info("Fetched dataloaders")
        logger.info(shape_str)
        logger.info(batch_str)
        if len(shapes) <= 5:
            logger.info(str(shapes))

    # #model
    # global_model = load_model(model_name, shape, normalization, **kwargs)
    #model
    global_model = load_model(model_name, shape, norm_name, cfg.training.init, cfg.training.freeze_core, cfg.model.constants, cfg.model.residuals, None, nodes_stats_dict, device=="cpu", logger, **kwargs)

    #training
    def client_builder(client, learner):
        # if fedmin:
        #     return LocalFedmIN(client, learner, device)
        # else:
        return LocalFedAvg(client, learner)
    def server_builder(global_model):
        return GlobalFedAvg(global_model)
    def scheme_builder(server, nodes, shadow_server=None, shadow_nodes=None):
        return FedAvgScheme(cfg.training.epochs, cfg.fl.rounds, cfg.fl.sampled_clients, server, nodes, shadow_server, shadow_nodes, plus=True, server_side="full")
  
    server, shadow_server, nodes, shadow_nodes, size_weights = launch_training(client_builder, server_builder, scheme_builder, loaders_dicts, global_model,
        E, K, criterion, lr, eval_losses, device, logger,
        save_dir, save_name, retrain=True, verbose=1)

    #example
    global_model.load_state_dict(server.update)
    global_model.to(device)
    launch_example(data_path, global_model, lags, horizon, device, save_dir, save_name, logger)

    #eval
    launch_eval(global_model, nodes, shadow_server, eval_losses, size_weights, output_dir, save_name, logger)

    # #betas
    # if verbose and ("revin" in normalization or "mIN" in normalization):
    #     params = {"beta": global_model.beta.data.detach().cpu().numpy()[0][0][0], "alpha": global_model.alpha.data.detach().cpu().numpy()[0][0][0]}
    #     logger.info(f"Final global modulations: {params}")
    #     if M<=10:
    #         for k in range(M):
    #             #model = nodes[k].get_client_weights()
    #             model = nodes[k].client.model
    #             params = {"beta": model.beta.data.detach().cpu().numpy()[0][0][0], "alpha": model.alpha.data.detach().cpu().numpy()[0][0][0]}
    #             logger.info(f"Final global modulations node{k}: {params}")

    logger.info('End of script\n')

if __name__ == "__main__":
    run()



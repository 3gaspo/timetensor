import hydra
import logging
import os
import torch
import numpy as np
import json

from src.timetensor.dataset import fetch_training_data, get_sizes, set_random_data, fetch_example_data, fetch_stats
from src.timetensor.models import load_model
from src.timetensor.pipeline import get_losses, launch_training
from src.timetensor.visu import plot_errors, plot_horizon_errors, plot_pred, plot_weights, plot_named_example
from src.timetensor.utils import save_results, get_dirs

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

    clusters = cfg.data.clusters

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
    stats_dict = fetch_stats(data_path, clusters, normalization, subsets)
    loaders_dict = fetch_training_data(data_path,
            cfg.data.indiv_split, cfg.data.date_splits, subsets,
            batch_size, lags, horizon, by_date=(cfg.data.by_idx=="dates"), context_by_individuals=cfg.data.context_by_individuals,
            reshuffle=cfg.data.reshuffle, remove_cte=cfg.data.remove_cte,
            clusters=clusters,  stats=stats_dict, seed=seed)
    shape, shape_str, batch_str = get_sizes(loaders_dict, str_info=True)
    if verbose:
        logger.info("Fetched dataloaders")
        logger.info(batch_str)

    #model
    if kwargs.get("init_alpha") is True:
        kwargs["init_alpha"] = stats_dict["train"]["alpha"]
    if kwargs.get("init_beta") is True:
        kwargs["init_beta"] = stats_dict["train"]["beta"]
    model = load_model(model_name, shape, normalization, init_path, cfg.training.freeze_core, logger, **kwargs)

    #training
    learner = launch_training(model, normalization, criterion, lr, batch_size, epochs, loaders_dict, eval_losses, device, save_dir, save_name, eval_freq, print_freq, logger, retrain)

    #eval
    logger.info("Computing test metrics")
    test_losses1 = learner.eval(loaders_dict["test1"], return_all=True, verbose=1, logger=logger) #(ndates*nindividuals, dim, horizon)
    test_losses2 = learner.eval(loaders_dict["test2"], return_all=True, verbose=1, logger=logger) #(ndates*nindividuals, dim, horizon)
    torch.save(test_losses1, save_dir + "test_losses1.pt")
    torch.save(test_losses2, save_dir + "test_losses2.pt")

    if benchmark:
        test_dir = output_dir
    else:
        test_dir = save_dir
    for loss_name in eval_losses:
        mean, std = test_losses1[loss_name].mean(), test_losses1[loss_name].std()
        save_results(mean, test_dir, "test1_mean_results.json", save_name, f"Test {loss_name}")
        save_results(std, test_dir, "test1_std_results.json", save_name, f"Test {loss_name}")
        if complete_evaluation:
            plot_errors(test_losses1[loss_name].sum(axis=1).mean(axis=1), save_dir + "plots/", f"test1_{loss_name}.pdf", f"Test 1 {loss_name} of {save_name} : {mean}")
            plot_horizon_errors(test_losses1[loss_name].sum(axis=1).mean(axis=0), save_dir + "plots/", f"test1_horizon_{loss_name}.pdf", f"Test 1 {loss_name} of {save_name} : {mean}")
    for loss_name in eval_losses:
        mean, std = test_losses2[loss_name].mean(), test_losses2[loss_name].std()
        save_results(mean, test_dir, "test2_mean_results.json", save_name, f"Test {loss_name}")
        save_results(std, test_dir, "test2_std_results.json", save_name, f"Test {loss_name}")
        if complete_evaluation:
            plot_errors(test_losses2[loss_name].sum(axis=1).mean(axis=1), save_dir + "plots/", f"test2_{loss_name}.pdf", f"Test 2 {loss_name} of {save_name} : {mean}")
            plot_horizon_errors(test_losses2[loss_name].sum(axis=1).mean(axis=0), save_dir + "plots/", f"test2_horizon_{loss_name}.pdf", f"Test 2 {loss_name} of {save_name} : {mean}")
    
    #examples
    ex_dir = data_path + "examples/" + f"{lags}_{horizon}/"
    if not os.path.exists(ex_dir):
        set_random_data(data_path, lags, horizon, name="rand")
        plot_named_example(ex_dir, f"rand")
    dico = fetch_example_data(ex_dir)
    for data_name, data_tuple in dico.items():
        x, c, y = data_tuple[0].unsqueeze(0).to(device), data_tuple[1], data_tuple[2].unsqueeze(0).to(device)
        if c is not None:
            c = c.unsqueeze(0).to(device)
        pred = model(x,c)
        plot_pred(x[0,0].cpu().detach().numpy(), y[0,0].cpu().detach().numpy(), pred[0,0].cpu().detach().numpy(), save_dir + "examples/", f"{data_name}_predictions.pdf", f"Example {data_name} prediction for {save_name}")        
    logger.info('Saved plots')

    #weights
    plot_weights(model, learner, save_dir, save_name)
    if "revin" in normalization or "mIN" in normalization:
        params = {"beta": model.beta.data.detach().cpu().numpy()[0][0][0], "alpha": model.alpha.data.detach().cpu().numpy()[0][0][0]}
        logger.info(f"Final modulations: {params}")

    logger.info('End of script\n')

if __name__ == "__main__":
    run()



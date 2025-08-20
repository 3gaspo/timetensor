import hydra
import logging
import os
import torch
import numpy as np

from src.timetensor.dataset import get_dataset_splits, get_train_loaders, get_sizes
from src.timetensor.models import load_model
from src.timetensor.pipeline import Learner, train_model, get_losses
from src.timetensor.visu import plot_losses, plot_multi_losses, plot_errors, plot_horizon_errors, plot_pred, plot_weights, plot_stats, plot_named_example, plot_serie
from src.timetensor.utils import save_results, fetch_example_data, get_dirs, unroll_windows, set_random_data

from src.timetensor.visu import plot_example

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    logger.info("=====Running main script=====")

    #configs
    data_path = cfg.data.path
    lags, horizon, remove_cte = cfg.task.lags, cfg.task.horizon, cfg.data.remove_cte
    indiv_split, date_splits, subsets, reshuffle = cfg.data.indiv_split, cfg.data.date_splits, cfg.data.subsets, cfg.data.reshuffle
    batch_size, lr, epochs, criterion_name = cfg.training.bs, cfg.training.lr, cfg.training.epochs, cfg.training.loss
    init_path = cfg.training.init
    eval_freq, print_freq = cfg.training.eval_freq, cfg.training.print_freq
    model_name, normalization, kwargs = cfg.model.name, cfg.normalization.name, cfg.model.configs
    if kwargs is None:
        kwargs = {}
    verbose, seed = 2, cfg.misc.seed
    logger.info(f"Model {model_name}, normalization {normalization}, criterion {criterion_name}, kwargs {kwargs}")
    
    if seed == "None":
        seed = None
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    #data   
    by_idx = cfg.data.by_idx
    data_dict = get_dataset_splits(data_path, indiv_split, date_splits, reshuffle=reshuffle)
    loaders_dict = get_train_loaders(data_dict, batch_size, lags, horizon, by_date=(by_idx=="dates"), subsets=subsets["sizes"], subset_mode=subsets["mode"], save_path=data_path+"subsets/", remove_cte=remove_cte)
    logger.info("Fetched dataloaders")

    #sizes
    shape, shape_str, batch_str = get_sizes(loaders_dict["train"], str_info=True)

    #model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion, eval_losses = get_losses(criterion_name, mean=None, std=None, complete_evaluation=True)
    model = load_model(model_name, shape, cfg.normalization, **kwargs)
    if init_path is not None:
        weights = torch.load(init_path)
        model.load_state_dict(weights)
        logger.info("Loaded previous state dict")
    if model_name in ["persistence", "repeat", "lookback", "expected"] and normalization not in ["revin", "mIN", "cmIN"]:
        learner = Learner(model, criterion, lr, eval_losses, device=device, do_train=False)
    elif model_name == "sklinear":
        learner = Learner(model, criterion, lr, eval_losses, device=device, pytorch=False)
        logger.info("Starting scikit-learn fitting...")
        learner.fit(loaders_dict["train"])
        logger.info("End of training")
    else:
        learner = Learner(model, criterion, lr, eval_losses, device=device)

    #eval
    logger.info("Computing test metrics")
    test_losses1 = learner.eval(loaders_dict["test1"], return_all=True, verbose=1, logger=logger) #(ndates*nindividuals, dim, horizon)
    
    #debug
    logger.info("DEBUG")
    errors = test_losses1["NMSE"]
    shape = loaders_dict["test1"].dataset.shape
    print("shape:",shape)
    print("errors:",errors.shape)
    idxs = np.where(errors>1e5)[0]
    idx = idxs[-1]
    print("idx:",idx)

    n_test_individuals, n_test_dates = shape[0], shape[2]
    indiv, date = idx % n_test_individuals, idx // n_test_individuals
    print("indiv:",indiv, "date:",date)

    data = loaders_dict["test1"].dataset.values[indiv][0]
    x, y = data[date - lags:date], data[date: date+horizon]
    print("x:",x.shape, "y:", y.shape)
    mean = x.mean(dim=-1, keepdim=True).detach()
    std =  x.std(dim=-1, keepdim=True).detach()
    print("mean:", mean, "std:", std)
    plot_example(x, y, path="", name="example.pdf", title="Example", axis=True)

if __name__ == "__main__":
    run()



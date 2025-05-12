import hydra
import logging
import os
import torch
import torch.nn as nn
import numpy as np

from src.timetensor.dataset import get_dataset_splits, get_train_loaders
from src.timetensor.models import load_model
from src.timetensor.pipeline import Learner, train_model, Loss
from src.timetensor.visu import plot_losses, plot_multi_losses, plot_errors, plot_horizon_errors, plot_pred, plot_weights, plot_stats
from src.timetensor.utils import save_results, fetch_example_data, get_dirs, unroll_windows, get_normal_stats, normalize

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
    batch_size, lr, epochs = cfg.training.bs, cfg.training.lr, cfg.training.epochs
    criterion_name, normalization = cfg.training.loss, cfg.model.normalization
    model_name, retrain = cfg.model.name, cfg.training.retrain
    kwargs = cfg.model_configs
    verbose, benchmark = cfg.misc.verbose, cfg.misc.benchmark
    if verbose:
        logger.info("Fetched main configs")
        logger.info(f"Model {model_name}, normalization {normalization}, criterion {criterion_name}, kwargs {kwargs}")


    #save dirs
    output_dir = cfg.misc.output_dir
    save_name = cfg.misc.save_name
    save_name, save_dir = get_dirs(output_dir, save_name, model_name, normalization, criterion_name)
    if verbose:
        logger.info("Fetched output directories")
        logger.info(f"Save directory : {save_dir}")

    #data
    data_dict = get_dataset_splits(data_path, cfg.data.indiv_split, cfg.data.date_split, cfg.misc.seed, save=False)
    loaders_dict = get_train_loaders(data_dict, batch_size, lags, horizon, by_date=True, subsets=cfg.subset.subsets, path=data_path+"subsets/")
    if verbose:
        logger.info("Fetched dataloaders")

    #sizes
    X, c, y = next(iter(loaders_dict["train"])) # (indiv, dim, lags),  #(nc, dim, horizon),  #(indiv, dim, horizon)
    shape = [X.shape[2], X.shape[1], y.shape[2]]
    if verbose:
        logger.info(f"Training data shape : {loaders_dict['train'].dataset.shape}")
        
        if c is not None:
            logger.info(f"Batch sizes : X={X.shape}, c={c.shape}, y={y.shape}")
        else:
            logger.info(f"Batch sizes : X={X.shape}, y={y.shape}")

    #training
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

    model = load_model(model_name, shape, normalization, **kwargs)
    if model_name in ["persistence", "repeat", "lookback"] and normalization!=3:
        learner = Learner(model, criterion, lr, eval_losses, device=device, do_train=False)
        logger.info("No training needed")
    elif model_name == "sklinear":
        logger.info("Starting scikit-learn fitting...")
        learner = Learner(model, criterion, lr, eval_losses, device=device, pytorch=False)
        learner.fit(loaders_dict["train"])
        logger.info("End of training")
    else:
        learner = Learner(model, criterion, lr, eval_losses, device=device)
        logger.info(f"batch_size={batch_size}, learning_rate={lr}, steps={len(loaders_dict['train'])}")
        if retrain:
            logger.info("Starting training...")
            train_losses, valid_losses, valid_losses2 = train_model(learner, loaders_dict, epochs=epochs)
            torch.save(learner.model.state_dict(), save_dir + "trained_model.pt")
            torch.save(train_losses, save_dir + f"train_losses.pt")
            torch.save(valid_losses, save_dir + f"valid_losses.pt")
            torch.save(valid_losses2, save_dir + f"valid_losses2.pt")

            logger.info("End of training")
        else:
            weights = torch.load(save_dir + "trained_model.pt")
            model.load_state_dict(weights)
            model.to(device)
            learner.reset_model(weights)
            train_losses = torch.load(save_dir + f"train_losses.pt",weights_only=False)
            valid_losses = torch.load(save_dir + f"valid_losses.pt", weights_only=False)
            valid_losses2 = torch.load(save_dir + f"valid_losses2.pt", weights_only=False)

        #plots
        for loss_name in eval_losses:
            if loss_name == criterion_name:
                plot_losses(train_losses, valid_losses[loss_name], valid_losses2[loss_name],  save_dir + "plots/", f"{loss_name}_plot.pdf", f"Training {loss_name} of {save_name}")
            else:
                plot_multi_losses({"valid1": valid_losses[loss_name], "valid2":valid_losses2[loss_name]},  save_dir + "plots/", f"{loss_name}_plot.pdf", f"Training {loss_name} of {save_name}")
        logger.info("Plotted losses")

    #eval
    logger.info("Computing test metrics")
    if model_name=="sklinear" and normalization==2:
        test_losses = learner.eval(loaders_dict["test"], return_all=True, verbose=1, normal=True) #(ndates*nindividuals, dim, horizon)
    else:
        test_losses = learner.eval(loaders_dict["test"], return_all=True, verbose=1) #(ndates*nindividuals, dim, horizon)
    torch.save(test_losses, save_dir + "test_losses.pt")
    if benchmark:
        test_dir = output_dir
    else:
        test_dir = save_dir
    for loss_name in eval_losses:
        mean, std = test_losses[loss_name].mean(), test_losses[loss_name].std()
        save_results(mean, test_dir, "mean_results.json", save_name, f"Test {loss_name}")
        save_results(std, test_dir, "std_results.json", save_name, f"Test {loss_name}")

        plot_errors(test_losses[loss_name].sum(axis=1).mean(axis=1), save_dir + "plots/", f"test_{loss_name}.pdf", f"Test {loss_name} of {save_name} : {mean}")
        plot_horizon_errors(test_losses[loss_name].sum(axis=1).mean(axis=0), save_dir + "plots/", f"horizon_{loss_name}.pdf", f"Test {loss_name} of {save_name} : {mean}")
    
    #examples
    dico = fetch_example_data(data_path + "examples/", [f"ex{k}" for k in range(1,6)])
    for data_name, data_tuple in dico.items():
        x, c, y = data_tuple[0].unsqueeze(0).to(device), data_tuple[1], data_tuple[2].unsqueeze(0).to(device)
        if c is not None:
            c = c.unsqueeze(0).to(device)
        pred = model(x,c)
        plot_pred(x[0,0].cpu().detach().numpy(), y[0,0].cpu().detach().numpy(), pred[0,0].cpu().detach().numpy(), save_dir + "examples/", f"{data_name}_predictions.pdf", f"Example {data_name} prediction for {save_name}")        
    logger.info('Saved plots')

    #weights
    if model_name in ["linear", "sklinear"]:
        if model_name == "sklinear":
            weights = learner.get_weights()
        else:
            if normalization!=0:
                weights = model.model.fc.weight.detach().cpu().numpy()
            else:
                weights = model.fc.weight.detach().cpu().numpy()
        plot_weights(weights, save_dir + "plots/", title=f'{save_name} weights')
    if model_name == "DLinear":
        if normalization!=0:
            linear_weights = model.model.Linear_Seasonal[0].weight.detach().cpu().numpy()
            season_weights = model.model.Linear_Trend[0].weight.detach().cpu().numpy()
        else:
            linear_weights = model.Linear_Seasonal[0].weight.detach().cpu().numpy()
            season_weights = model.Linear_Trend[0].weight.detach().cpu().numpy()
        plot_weights(linear_weights, save_dir + "plots/", name="season_weights.pdf", title=f'{save_name} seasonal weights')
        plot_weights(season_weights, save_dir + "plots/", name="trend_weights.pdf", title=f'{save_name} trend weights')

    #revin
    if normalization == 3:
        params = {"beta": model.beta.data.detach().cpu().numpy()[0][0][0], "gamma": model.gamma.data.detach().cpu().numpy()[0][0][0]}
        logger.info(f"Final revin parameters: {params}")
        unroll = unroll_windows(loaders_dict["train"], normal=True)
        runroll = unroll_windows(loaders_dict["train"], normal=True, beta=model.beta.data.detach().cpu().numpy()[0][0][0], gamma=model.gamma.data.detach().cpu().numpy()[0][0][0])
        x_dict = {"train": unroll[0]}
        rx_dict = {"train": runroll[0]}
        plot_stats(x_dict, save_dir + "plots/", name="normal_outputs.pdf", title="Normalized outputs distribution", logscale=False, limits=(-1e-6,1e-6))
        plot_stats(rx_dict, save_dir + "plots/", name="revin_outputs.pdf", title="Normalized outputs distribution", logscale=False, limits=(-1e-6,1e-6))


    logger.info('End of script\n')

if __name__ == "__main__":
    run()



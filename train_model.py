import hydra
import logging
import os
import torch
import torch.nn as nn
import numpy as np

from src.timetensor.dataset import get_dataset_splits, get_train_loaders
from src.timetensor.models import load_model
from src.timetensor.pipeline import Learner, train_model
from src.timetensor.visu import plot_losses, plot_errors, plot_horizon_errors, plot_pred, plot_weights
from src.timetensor.utils import save_results, fetch_example_data, nloss, get_dirs

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
    criterion_name, normalize_criterion, normalization = cfg.training.loss, cfg.training.normalize_criterion, cfg.model.normalization
    model_name, retrain = cfg.model.name, cfg.training.retrain
    kwargs = cfg.model_configs
    verbose, benchmark = cfg.misc.verbose, cfg.misc.benchmark
    if verbose:
        logger.info("Fetched main configs")
        logger.info(f"Model {model_name}, normalization {normalization}, kwargs {kwargs}")


    #save dirs
    output_dir = cfg.misc.output_dir
    save_name = cfg.misc.save_name
    save_name, save_dir = get_dirs(output_dir, save_name, model_name, normalization)
    if verbose:
        logger.info("Fetched output directories")
        logger.info(f"Save directory : {save_dir}")

    #data
    data_dict = get_dataset_splits(data_path, cfg.data.indiv_split, cfg.data.date_split, cfg.misc.seed, save=False)
    loaders_dict = get_train_loaders(data_dict, batch_size, lags, horizon, by_date=True, subsets=cfg.data.subset, path=data_path)
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
        criterion = lambda pred,y: nloss(nn.MSELoss(), pred, y, cfg.data.mean, cfg.data.std)
    else:
        print("Unknown criterion name")
        criterion = None
    #eval_losses = {"MSE":nn.MSELoss(reduction="none")}
    eval_losses = {"MSE":lambda pred, y: nloss(nn.MSELoss(reduction="none"), pred, y, cfg.data.mean, cfg.data.std)}

    model = load_model(model_name, shape, normalization, **kwargs)
    if model_name in ["persistence", "repeat", "lookback"] and normalization!=3:
        learner = Learner(model, criterion, lr, eval_losses, device=device, normalize_criterion=normalize_criterion, do_train=False)
        logger.info("No training needed")
    elif model_name == "sklinear":
        logger.info("Starting scikit-learn fitting...")
        learner = Learner(model, criterion, lr, eval_losses, device=device, normalize_criterion=normalize_criterion, pytorch=False)
        learner.fit(loaders_dict["train"])
        logger.info("End of training")
    else:
        learner = Learner(model, criterion, lr, eval_losses, device=device, normalize_criterion=normalize_criterion)
        logger.info(f"batch_size={batch_size}, learning_rate={lr}, steps={len(loaders_dict['train'])}")
        if retrain:
            logger.info("Starting training...")
            train_losses, valid_losses, valid_losses2 = train_model(learner, loaders_dict)
            torch.save(learner.model.state_dict(), save_dir + "trained_model.pt")
            torch.save(train_losses, save_dir + f"{criterion_name}_train_losses.pt")
            for loss_name, loss_values in valid_losses.items():
                torch.save(loss_values, save_dir + f"{loss_name}_valid_losses.pt")
                torch.save(loss_values, save_dir + f"{loss_name}_valid_losses2.pt")
            logger.info("End of training")
        else:
            model.load_state_dict(torch.load(save_dir + "trained_model.pt"))
            model.to(device)
            train_losses = torch.load(save_dir + f"{criterion_name}_train_losses.pt",weights_only=False)
            valid_losses, valid_losses2 = {}, {}
            for loss_name in eval_losses:
                valid_losses[loss_name] = torch.load(save_dir + f"{loss_name}_valid_losses.pt",weights_only=False)
                valid_losses["N"+loss_name] = torch.load(save_dir + f"N{loss_name}_valid_losses.pt",weights_only=False)
                valid_losses2[loss_name] = torch.load(save_dir + f"{loss_name}_valid_losses2.pt",weights_only=False)
                valid_losses2["N"+loss_name] = torch.load(save_dir + f"N{loss_name}_valid_losses2.pt",weights_only=False)

        #plots
        plot_losses(train_losses, valid_losses["MSE"], valid_losses2["MSE"],  save_dir, "train_losses.pdf", f"Training MSE of {save_name}")
        plot_losses(train_losses, valid_losses["NMSE"], valid_losses2["NMSE"],  save_dir, "train_nlosses.pdf", f"Training NMSE of {save_name}")

        logger.info("Plotted losses")

    #eval
    logger.info("Computing test metrics")
    test_losses = learner.eval(loaders_dict["test"], return_all=True, verbose=1) #(ndates*nindividuals, dim, horizon)
    torch.save(test_losses, save_dir + "test_losses.pt")
    mean_test_mse, std_test_mse = test_losses["MSE"].mean(), test_losses["MSE"].std()
    mean_test_nmse, std_test_nmse = test_losses["NMSE"].mean(), test_losses["NMSE"].std()
    if benchmark:
        test_dir = output_dir
    else:
        test_dir = save_dir
    save_results(mean_test_mse, test_dir, "mean_results.json", save_name, "Test MSE")
    save_results(std_test_mse, test_dir, "std_results.json", save_name, "Test MSE")
    save_results(mean_test_nmse, test_dir, "mean_results.json", save_name, "Test NMSE")
    save_results(std_test_nmse, test_dir, "std_results.json", save_name, "Test NMSE")
    logger.info(f"Test MSE : {mean_test_mse:.4f} (+/- {std_test_mse:.4f}), Test NMSE : {mean_test_nmse:.4f} (+/- {std_test_nmse:.4f})")


    #errors
    plot_errors(test_losses["MSE"].sum(axis=1).mean(axis=1), save_dir, "test_mse.pdf", f"Test MSE of {save_name} : {mean_test_mse}")
    plot_errors(test_losses["NMSE"].sum(axis=1).mean(axis=1), save_dir, "test_nme.pdf", f"Test NMSE of {save_name} : {mean_test_nmse}")
    plot_horizon_errors(test_losses["MSE"].sum(axis=1).mean(axis=0), save_dir, "horizon_mse.pdf", f"Test MSE of {save_name} : {mean_test_mse}")
    plot_horizon_errors(test_losses["NMSE"].sum(axis=1).mean(axis=0), save_dir, "horizon_nmse.pdf", f"Test NMSE of {save_name} : {mean_test_nmse}")
    
    #example
    dico = fetch_example_data(data_path + "examples/", ["rand"])
    for data_name, data_tuple in dico.items():
        x, c, y = data_tuple[0].unsqueeze(0).to(device), data_tuple[1], data_tuple[2].unsqueeze(0).to(device)
        if c is not None:
            c = c.unsqueeze(0).to(device)
        pred = model(x,c)
        if model_name == "sklinear":
            pred = pred.unsqueeze(dim=1)
        plot_pred(x[0,0].cpu().detach().numpy(), y[0,0].cpu().detach().numpy(), pred[0,0].cpu().detach().numpy(), save_dir + "examples/", f"{data_name}_predictions.pdf", f"Example {data_name} prediction for {save_name}")        
    logger.info('Saved plots')

    #visu
    if model_name in ["linear", "sklinear"]:
        if model_name == "sklinear":
            weights = learner.get_weights()
        else:
            if normalization!=0:
                weights = model.model.fc.weight.detach().cpu().numpy()
            else:
                weights = model.fc
        plot_weights(weights, save_dir, title=f'{save_name} weights')

    logger.info('End of script\n')

if __name__ == "__main__":
    run()



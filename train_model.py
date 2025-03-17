import hydra
import logging
import os
import torch
import torch.nn as nn
import numpy as np

from src.timetensor.dataset import get_train_loaders
from src.timetensor.models import load_model
from src.timetensor.pipeline import train_model, eval_model
from src.timetensor.visu import plot_losses, plot_errors, plot_horizon_errors, plot_pred
from src.timetensor.utils import save_results, fetch_example_data

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

@hydra.main(version_base=None, config_path="config", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    print("\n\n")
    logger.info("=====Running main script=====")

    #configs
    data_path = cfg.data.path
    lags, horizon = cfg.model.lags, cfg.model.horizon
    batch_size, subset_data, lr = cfg.training.bs, cfg.training.subset_data, cfg.training.lr
    criterion_name, normalized = cfg.training.loss, cfg.training.normalized
    model_name, retrain = cfg.model.name, cfg.training.retrain
    revin = cfg.model.revin
    kwargs = cfg.model_configs
    output_dir = cfg.misc.output_dir
    save_name = cfg.misc.save_name
    verbose = cfg.misc.verbose
    if verbose:
        logger.info("Fetched configs")
        logger.info(f"Model {model_name}, revin {revin}, kwargs {kwargs}")


    #save dirs
    if save_name is None:
        save_name = model_name
        if revin:
            save_name = save_name + "_revin"   
    save_dir = output_dir + save_name + "/"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    hydra_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    with open(save_dir + f'hydra_dir.txt', 'w') as file: 
        file.write(f"{hydra_dir}")
    if not os.path.exists(save_dir + "examples/"):
        os.makedirs(save_dir + "examples/")
    if verbose:
        logger.info("Fetched output directories")
        logger.info(f"Save directory : {save_dir}")

    #data
    data_dict = get_train_loaders(data_path, batch_size, lags, horizon, subset=subset_data)
    if verbose:
        if subset_data < 1:
            logger.info(f"Fetched dataloaders with subset ratio : {subset_data}")
        else:
            logger.info("Fetched dataloaders")
    
    #sizes
    logger.info(f"Dataset shape : {data_dict['train'].dataset.shape()}")
    X, c, y = next(iter(data_dict["train"])) # (indiv, dim, lags),  #(nc, dim, horizon),  #(indiv, dim, horizon)
    shape = [X.shape[1], X.shape[2], y.shape[2]]
    if verbose:
        if c is not None:
            logger.info(f"Batch sizes : X={X.shape}, c={c.shape}, y={y.shape}")
        else:
            logger.info(f"Batch sizes : X={X.shape}, y={y.shape}")

    #model
    if verbose:
        logger.info(f"Fetching model")
    model = load_model(model_name, shape, revin, **kwargs)

    if criterion_name == "MSE":
        criterion = nn.MSELoss()
    else:
        criterion = None

    #training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_losses = {"MSE":nn.MSELoss(reduction="none")}
    if model_name in ["MLP", "linear","patch_tst"] or revin:
        logger.info(f"batch_size={batch_size}, learning_rate={lr}, steps={len(data_dict['train'])}")
        if retrain:
            logger.info("Starting training...")
            # for name, param in model.named_parameters():
            #     print(f"Parameter name: {name} | ", param)
            model, train_losses, valid_losses = train_model(model, data_dict, lr, criterion, normalized, device=device)
            # for name, param in model.named_parameters():
            #     print(f"Parameter name: {name} | ", param)
            
            torch.save(model.state_dict(), save_dir + "model.pt")
            torch.save(train_losses, save_dir + f"{criterion_name}_train_losses.pt")
            for loss_name, loss_values in valid_losses.items():
                torch.save(loss_values, save_dir + f"{loss_name}_valid_losses.pt")
            logger.info("End of training")
        else:
            valid_losses = {}
            for key in eval_losses:
                valid_losses[key] = []
                valid_losses["N"+key] = []
            model = load_model(model_name, horizon, revin, **kwargs)
            model.load_state_dict(torch.load(save_dir + "model.pt"))
            train_losses = torch.load(save_dir + f"{criterion_name}_train_losses.pt",weights_only=False)
            for loss_name, _ in valid_losses.items():
                valid_losses[loss_name] = torch.load(save_dir + f"{loss_name}_valid_losses.pt",weights_only=False)
        #plots
        plot_losses(train_losses, valid_losses["NMSE"], save_dir, "train_losses.pdf", f"Training NMSE of {save_name}")
        plot_losses(valid_losses["MSE"], None, save_dir, "vaild_losses.pdf", f"Validation MSE of {save_name}")
        plot_losses(valid_losses["NMSE"], None, save_dir, "vaild_nlosses.pdf", f"Validation NMSE of {save_name}")
        logger.info("Plotted losses")
    else:
        logger.info("No training needed")

    #eval
    logger.info("Computing test metrics")
    test_losses = eval_model(model, data_dict["test"], device, eval_losses, verbose=1) #(bs * steps, dim, horizon)
    torch.save(test_losses, save_dir + "test_losses.pt")
    mean_test_mse, std_test_mse = test_losses["MSE"].mean(), test_losses["MSE"].std()
    mean_test_nmse, std_test_nmse = test_losses["NMSE"].mean(), test_losses["NMSE"].std()
    save_results(mean_test_mse, output_dir, "mean_results.json", save_name, "Test MSE")
    save_results(std_test_mse, output_dir, "std_results.json", save_name, "Test MSE")
    save_results(mean_test_nmse, output_dir, "mean_results.json", save_name, "Test NMSE")
    save_results(std_test_nmse, output_dir, "std_results.json", save_name, "Test NMSE")
    logger.info(f"Test MSE : {mean_test_mse:.2f} (+/- {std_test_mse:.2f}), Test NMSE : {mean_test_nmse:.2f} (+/- {std_test_nmse:.2f})")


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
        plot_pred(x[0,0].cpu().detach().numpy(), y[0,0].cpu().detach().numpy(), pred[0,0].cpu().detach().numpy(), save_dir + "examples/", f"{data_name}_predictions.pdf", f"Example {data_name} prediction for {save_name}")        
    logger.info('Saved plots')

    logger.info('End of script\n')

if __name__ == "__main__":
    run()



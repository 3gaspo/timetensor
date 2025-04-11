import hydra
import logging
import os
import torch
import torch.nn as nn
import numpy as np

from src.timetensor.dataset import get_dataset_splits, get_train_loaders
from src.timetensor.models import load_model
from src.timetensor.pipeline import Learner, train_model
from src.timetensor.visu import plot_losses, plot_errors, plot_horizon_errors, plot_pred
from src.timetensor.utils import save_results, fetch_example_data

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
    criterion_name, normalized, revin = cfg.training.loss, cfg.training.normalized,  cfg.model.revin
    model_name, retrain = cfg.model.name, cfg.training.retrain
    kwargs = cfg.model_configs
    verbose, benchmark = cfg.misc.verbose, cfg.misc.benchmark
    # if verbose:
    #     logger.info("Fetched main configs")
    #     logger.info(f"Model {model_name}, revin {revin}, kwargs {kwargs}")


    #save dirs
    output_dir = cfg.misc.output_dir
    save_name = cfg.misc.save_name
    if save_name is None:
        save_name = model_name
        if revin:
            save_name = save_name + "_revin"   
    save_dir = output_dir + save_name + "/" #current experiment dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    hydra_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir #hydra logs
    with open(save_dir + f'hydra_dir.txt', 'w') as file: 
        file.write(f"{hydra_dir}") #save path of hydra logs to experiment dir
    if not os.path.exists(save_dir + "examples/"): #dir for example predictions
        os.makedirs(save_dir + "examples/")
    # if verbose:
    #     logger.info("Fetched output directories")
    #     logger.info(f"Save directory : {save_dir}")

    #data
    data_dict = get_dataset_splits(data_path, cfg.data.indiv_split, cfg.data.date_split, cfg.misc.seed, save=False)
    loaders_dict = get_train_loaders(data_dict, batch_size, lags, horizon, by_date=True, subsets=cfg.data.subset, path=data_path)
    if verbose:
        logger.info("Fetched dataloaders")

    #sizes
    X, c, y = next(iter(loaders_dict["train"])) # (indiv, dim, lags),  #(nc, dim, horizon),  #(indiv, dim, horizon)
    shape = [X.shape[1], X.shape[2], y.shape[2]]
    # if verbose:
    #     logger.info(f"Training data shape : {loaders_dict['train'].dataset.shape}")
        
    #     if c is not None:
    #         logger.info(f"Batch sizes : X={X.shape}, c={c.shape}, y={y.shape}")
    #     else:
    #         logger.info(f"Batch sizes : X={X.shape}, y={y.shape}")

    #training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if criterion_name == "MSE":
        criterion = nn.MSELoss()
    else:
        print("Unknown criterion name")
        criterion = None
    eval_losses = {"MSE":nn.MSELoss(reduction="none")}
    if model_name in ["MLP", "linear","patch_tst"] or revin:
        logger.info(f"batch_size={batch_size}, learning_rate={lr}, steps={len(loaders_dict['train'])}")

        model = load_model(model_name, shape, revin, **kwargs)
        # learner = Learner(model, criterion, lr, eval_losses, device=device, normalized_criterion=True)
        model.load_state_dict(torch.load(save_dir + "trained_model.pt"))
        model.to(device)
        # train_losses = torch.load(save_dir + f"{criterion_name}_train_losses.pt",weights_only=False)
        # valid_losses = {}
        # for loss_name in eval_losses:
        #     valid_losses[loss_name] = torch.load(save_dir + f"{loss_name}_valid_losses.pt",weights_only=False)
        #     valid_losses["N"+loss_name] = torch.load(save_dir + f"N{loss_name}_valid_losses.pt",weights_only=False)
        # valid_losses2 = {}
        # for loss_name in eval_losses:
        #     valid_losses2[loss_name] = torch.load(save_dir + f"{loss_name}_valid_losses2.pt",weights_only=False)
        #     valid_losses2["N"+loss_name] = torch.load(save_dir + f"N{loss_name}_valid_losses2.pt",weights_only=False)

    else:
        logger.info("No training needed")

    # #example
    # dico = fetch_example_data(data_path + "examples/", ["rand"])
    # for data_name, data_tuple in dico.items():
    #     x, c, y = data_tuple[0].unsqueeze(0).to(device), data_tuple[1], data_tuple[2].unsqueeze(0).to(device)
    #     if c is not None:
    #         c = c.unsqueeze(0).to(device)
    #     pred = model(x,c)
    #     plot_pred(x[0,0].cpu().detach().numpy(), y[0,0].cpu().detach().numpy(), pred[0,0].cpu().detach().numpy(), save_dir + "examples/", f"{data_name}_predictions.pdf", f"Example {data_name} prediction for {save_name}")        
    # logger.info('Saved plots')


    weights = model.fc.weight.detach().cpu().numpy()  # shape: (5, 10)
    print(weights.shape)

    import matplotlib.pyplot as plt

    plt.imshow(weights, aspect='auto', cmap='viridis')
    plt.colorbar(label='Weight value')
    plt.xlabel('Inputs (lookback)')
    plt.ylabel('Outputs (horizon)')
    plt.title('Model weights')
    plt.savefig("weights.pdf")

    logger.info('End of script\n')

if __name__ == "__main__":
    run()



import hydra
import logging
import os
import torch

from src.timetensor.dataset import get_train_loaders
from src.timetensor.models import load_model
from src.timetensor.pipeline import train_model, eval_model
from src.timetensor.visu import plot_losses#, plot_pred, plot_errors, plot_horizon_errors

# from src.data.dataloader import get_data_loaders
# from src.data.process import fetch_example_data
# from src.training.pipeline import train_model, eval_model
# from src.training.utils import save_results, normalize
# from src.models.network import MLP
# from src.models.patchtst.patch_tst import PatchTST
# from src.models.naive import persistence, repeat, lookback, linear

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from omegaconf import OmegaConf

from hydra import

@hydra.main(version_base=None, config_path="config", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    print("\n\n")
    logger.info("=====Running main script=====")

    #configs
    path = cfg.data.path
    lags, horizon = cfg.model.lags, cfg.model.horizon
    batch_size, lr = cfg.training.bs, cfg.training.lr
    model_name = cfg.model.name
    revin = cfg.model.revin
    kwargs = cfg.model_configs
    output_dir = cfg.misc.output_dir
    save_name = cfg.misc.save_name
    verbose = cfg.misc.verbose
    if verbose:
        logger.info("Fetched configs")

    #save dirs
    if save_name is None:
        save_name = model_name
        if revin:
            save_name + "_revin"    
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

    # lr, schedule = cfg.training.lr, cfg.training.schedule
    # loss_name = cfg.training.loss
    # do_print = cfg.training.print
    # valid_steps, test_steps = cfg.training.valid_steps, cfg.training.test_steps
    # n_prints, n_evals = cfg.training.n_prints, cfg.training.n_evals
    # seed = cfg.misc.seed

    #data
    data_dict = get_train_loaders(path, batch_size, lags, horizon)
    if verbose:
        logger.info("Fetched dataloaders")
    
    #sizes
    logger.info(f"Dataset shape : {data_dict['train'].dataset.shape()}")
    X, y = next(iter(data_dict["train"]))
    if verbose:
        logger.info(f"Batch sizes : X={X.shape}, y={y.shape}")

    #model
    model = load_model(model_name, horizon, revin, **kwargs)
    if verbose:
        logger.info(f"Fetched model {model_name}")


    #training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name in ["MLP", "linear","patch_tst"]:
        logger.info("Starting training...")
        logger.info(f"batch_size={batch_size}, learning_rate={lr}, steps={len(data_dict['train'])}")
    
        model, train_losses, valid_losses = train_model(model, data_dict, lr, device=device)
        torch.save(model.state_dict(), save_dir + "model.pt")
        plot_losses(train_losses, valid_losses, save_dir, "train_losses.pdf", f"Losses of {save_name}")
        plot_losses(valid_losses, None, save_dir, "vaild_losses.pdf", f"Losses of {save_name}")
        torch.save(train_losses, save_dir + "train_losses.pt")
        torch.save(valid_losses, save_dir + "valid_losses.pt")
        logger.info("End of training")
    else:
        logger.info("No training needed")

    # #eval
    # test_losses, normalized_test_losses = eval_model(model, test_loader, device, normal=normal, return_all=True) #(bs * steps, dim, horizon)
    # test_mse, test_nmse = test_losses.mean().item(), normalized_test_losses.mean().item()
    # torch.save(test_losses, save_dir + "test_losses.pt")
    # torch.save(normalized_test_losses, save_dir + "normalized_losses.pt")
    # logger.info(f"Test MSE : {test_mse:.2f}, Test NMSE : {test_nmse:.5f}")
    # save_results(test_mse, output_dir, "results.json", save_name, "Test MSE")
    # save_results(test_nmse, output_dir, "results.json", save_name, "Test NMSE")

    # #errors
    # plot_errors(test_losses[:, 0, :].mean(axis=1).cpu().numpy(), save_dir, "test_mse.pdf", f"Test MSE of {save_name} : {test_mse}")
    # plot_errors(normalized_test_losses[:, 0, :].mean(axis=1).cpu().numpy(), save_dir, "test_nme.pdf", f"Test NMSE of {save_name} : {test_nmse}")
    # plot_horizon_errors(test_losses[:, 0, :].mean(axis=0).cpu().numpy(), save_dir, "horizon_mse.pdf", f"Test MSE of {save_name} : {test_nmse}")
    # plot_horizon_errors(normalized_test_losses[:, 0, :].mean(axis=0).cpu().numpy(), save_dir, "horizon_nmse.pdf", f"Test NMSE of {save_name} : {test_nmse}")
    
    # #example
    # dico = fetch_example_data("datasets/examples/", ["motif", "big_motif", "anomalie"])
    # for data_name, data_tuple in dico.items():
    #     x, c, y = data_tuple[0].unsqueeze(0).to(device), data_tuple[1].unsqueeze(0).to(device), data_tuple[2].unsqueeze(0).to(device)
    #     x_normalized, mean, std =  normalize(x, return_stats=True)
    #     if normal:
    #         pred_normalized = model(x_normalized,c)
    #         pred = pred_normalized*std + mean
    #     else:
    #         pred = model(x,c)
    #         pred_normalized = (pred - mean)/std
    #     y_normalized = (y - mean)/std
    #     plot_pred(x[0,0].cpu().detach().numpy(), y[0,0].cpu().detach().numpy(), pred[0,0].cpu().detach().numpy(), save_dir + "examples/", f"{data_name}_predictions.pdf", f"Example {data_name} prediction for {save_name}")        
    #     plot_pred(x_normalized[0,0].cpu().detach().numpy(), y_normalized[0,0].cpu().detach().numpy(), pred_normalized[0,0].cpu().detach().numpy(), save_dir + "examples/", f"{data_name}_normal_predictions.pdf", f"Example {data_name} normalized prediction for {save_name}")        
    # logger.info('Saved plots')

    logger.info('End of script\n')

if __name__ == "__main__":
    run()



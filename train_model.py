import hydra
import logging
import os
import torch
import numpy as np

from src.timetensor.dataset import get_dataset_splits, get_train_loaders, get_sizes, aggregate_loaders_dict
from src.timetensor.models import load_model
from src.timetensor.pipeline import Learner, train_model, get_losses
from src.timetensor.visu import plot_losses, plot_multi_losses, plot_errors, plot_horizon_errors, plot_pred, plot_weights, plot_stats, plot_named_example, plot_serie
from src.timetensor.utils import save_results, fetch_example_data, get_dirs, set_random_data

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    logger.info("=====Running main script=====")

    #configs
    data_path = cfg.data.path
    lags, horizon, remove_cte = cfg.task.lags, cfg.task.horizon, cfg.data.remove_cte
    indiv_split, date_splits, subsets, reshuffle, by_idx = cfg.data.indiv_split, cfg.data.date_splits, cfg.data.subsets, cfg.data.reshuffle, cfg.data.by_idx
    batch_size, lr, epochs, criterion_name = cfg.training.bs, cfg.training.lr, cfg.training.epochs, cfg.training.loss
    retrain, init_path = cfg.training.retrain, cfg.training.init
    eval_freq, print_freq = cfg.training.eval_freq, cfg.training.print_freq
    model_name, normalization, kwargs = cfg.model.name, cfg.normalization.name, cfg.model.configs
    do_clusters, n_clusters = cfg.data.do_clusters, cfg.data.n_clusters
    if kwargs is None:
        kwargs = {}
    verbose, complete_evaluation, seed = cfg.misc.verbose, cfg.misc.complete_evaluation, cfg.misc.seed
    benchmark, output_dir, save_name = cfg.misc.benchmark, cfg.misc.output_dir, cfg.misc.save_name, 
    save_name, save_dir = get_dirs(output_dir, save_name, model_name, normalization, criterion_name, subsets["sizes"])
    if verbose:
        logger.info(f"Fetched main configs, save directory : {save_dir}")
        logger.info(f"Model {model_name}, normalization {normalization}, criterion {criterion_name}, kwargs {kwargs}")
    if seed == "None":
        seed = None
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    #data   
    if do_clusters:
        loaders_dicts = []
        for k in range(1,n_clusters+1):
            data_dict = get_dataset_splits(data_path + f"node{k}/", indiv_split, date_splits, context_by_individuals=True, reshuffle=reshuffle)
            loaders_dicts.append(get_train_loaders(data_dict, batch_size, lags, horizon, by_date=False))
        loaders_dict = aggregate_loaders_dict(loaders_dicts)  
    else:
        data_dict = get_dataset_splits(data_path, indiv_split, date_splits, reshuffle=reshuffle)
        loaders_dict = get_train_loaders(data_dict, batch_size, lags, horizon, by_date=(by_idx=="dates"), subsets=subsets["sizes"], subset_mode=subsets["mode"], save_path=data_path+"subsets/", remove_cte=remove_cte)
    if verbose:
        logger.info("Fetched dataloaders")

    #sizes
    shape, shape_str, batch_str = get_sizes(loaders_dict["train"], str_info=True)
    #X, c, y = next(iter(loaders_dict["train"])) # (indiv, dim, lags),  #(nc, dim, horizon),  #(indiv, dim, horizon)
    if verbose:
        logger.info(shape_str)
        logger.info(batch_str)

    #training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion, eval_losses = get_losses(criterion_name, mean=None, std=None, complete_evaluation=complete_evaluation)

    model = load_model(model_name, shape, cfg.normalization, **kwargs)
    if init_path is not None:
        weights = torch.load(init_path)
        model.load_state_dict(weights)
        logger.info("Loaded previous state dict")
    if cfg.training.freeze_core:
        if normalization is None or normalization=="None":
            for param in model.parameters():
                param.requires_grad = False
        else:
            for param in model.model.parameters():
                param.requires_grad = False
        trainable_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
        logger.info(f"Froze parameters. Trainable: {trainable_params}")

    if model_name in ["persistence", "repeat", "lookback", "expected"] and normalization not in ["revin", "mIN", "cmIN"]:
        learner = Learner(model, criterion, lr, eval_losses, device=device, do_train=False)
        logger.info("No training needed")
    elif model_name == "sklinear":
        learner = Learner(model, criterion, lr, eval_losses, device=device, pytorch=False)
        if retrain:
            logger.info("Starting scikit-learn fitting...")
            learner.fit(loaders_dict["train"])
            logger.info("End of training")
        else:
            logger.info("No training needed")
    else:
        if retrain:
            learner = Learner(model, criterion, lr, eval_losses, device=device)
            logger.info(f"batch_size={batch_size}, learning_rate={lr}, steps per epoch={len(loaders_dict['train'])}, epochs={epochs}")
            logger.info("Starting training...")
            if normalization in ["revin", "mIN"]:
                weight_follow = lambda model: {"beta": model.beta.data.detach().cpu().numpy()[0][0][0], "alpha": model.alpha.data.detach().cpu().numpy()[0][0][0]}
            else:
                weight_follow=None
            train_losses, valid_losses1, valid_losses2, valid_losses3, followed_weights = train_model(learner, loaders_dict, epochs=epochs, logger=logger, eval_runs=1, eval_freq=eval_freq, print_freq=print_freq, weight_follow=weight_follow)
            torch.save(learner.model.state_dict(), save_dir + "trained_model.pt")
            torch.save(train_losses, save_dir + f"train_losses.pt")
            torch.save(valid_losses1, save_dir + f"valid_losses1.pt")
            torch.save(valid_losses2, save_dir + f"valid_losses2.pt")
            torch.save(valid_losses3, save_dir + f"valid_losses3.pt")
            torch.save(followed_weights, save_dir + f"followed_weights.pt")
            logger.info("End of training")
            
            #plots
            for loss_name in eval_losses:
                valid_dict = {"valid1": valid_losses1[loss_name], "valid2": valid_losses2[loss_name], "valid3": valid_losses3[loss_name]}
                if loss_name == criterion_name or (loss_name=="NMSE" and "NMSE" in criterion_name):
                    plot_losses(train_losses, valid_dict, save_dir + "plots/", f"{loss_name}_plot.pdf", f"Training {loss_name} of {save_name}", eval_freq=eval_freq)
                else:
                    plot_multi_losses(valid_dict,  save_dir + "plots/", f"{loss_name}_plot.pdf", f"Training {loss_name} of {save_name}", eval_freq=eval_freq)
            for weight_name in followed_weights:
                plot_serie(followed_weights[weight_name], save_dir + "plots/", f"{weight_name}.pdf", title=f"{weight_name} during training")
            logger.info("Plotted losses")
        
        else:
            logger.info("No training needed")
            if init_path is None:
                weights = torch.load(save_dir + "trained_model.pt")
                model.load_state_dict(weights)
                learner.reset_model(weights)
            learner = Learner(model, criterion, lr, eval_losses, device=device)

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
    if do_clusters:
        ex_dir = data_path+"node0/"+"examples/" + f"{lags}_{horizon}/"
    else:
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
    if model_name in ["linear", "sklinear"]:
        if model_name == "sklinear":
            weights = learner.get_weights()
        else:
            if normalization != "None":
                weights = model.model.fc.weight.detach().cpu().numpy()
            else:
                weights = model.fc.weight.detach().cpu().numpy()
        plot_weights(weights, save_dir + "plots/", title=f'{save_name} weights')
    if model_name == "DLinear":
        if normalization != "None":
            linear_weights = model.model.Linear_Seasonal[0].weight.detach().cpu().numpy()
            season_weights = model.model.Linear_Trend[0].weight.detach().cpu().numpy()
        else:
            linear_weights = model.Linear_Seasonal[0].weight.detach().cpu().numpy()
            season_weights = model.Linear_Trend[0].weight.detach().cpu().numpy()
        plot_weights(linear_weights, save_dir + "plots/", name="season_weights.pdf", title=f'{save_name} seasonal weights')
        plot_weights(season_weights, save_dir + "plots/", name="trend_weights.pdf", title=f'{save_name} trend weights')

    #mIN
    if normalization in ["revin","mIN"]:
        params = {"beta": model.beta.data.detach().cpu().numpy()[0][0][0], "alpha": model.alpha.data.detach().cpu().numpy()[0][0][0]}
        logger.info(f"Final mIN parameters: {params}")


    logger.info('End of script\n')

if __name__ == "__main__":
    run()



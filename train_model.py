import hydra
import logging
import torch
import numpy as np

from src.timetensor.dataset import fetch_training_data, get_sizes, apply_stats
from src.timetensor.models import load_model
from src.timetensor.pipeline import get_losses
from src.timetensor.visu import plot_weights
from src.timetensor.utils import get_dirs

from src.timetensor.pipeline import launch_training, launch_eval, launch_example


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    logger.info("=====Running train script=====")

    #configs
    data_path = cfg.data.path
    lags, horizon = int(cfg.task.lags), int(cfg.task.horizon)

    split_kwargs, subset_kwargs = cfg.data.splits, cfg.data.subsets
    clusters = cfg.data.clustering.clusters

    batch_size, lr, epochs = cfg.training.bs, cfg.training.lr, cfg.training.epochs
    criterion_name, complete_evaluation = cfg.training.loss, cfg.misc.complete_evaluation
    eval_freq, print_freq,  = cfg.training.eval_freq, cfg.training.print_freq
    criterion, eval_losses = get_losses(criterion_name, complete_evaluation=complete_evaluation)

    model_name, norm_name, norm_kwargs, model_kwargs = cfg.model.name, cfg.normalization.name, cfg.normalization.configs, cfg.model.configs
    if norm_name == "None":
        norm_name = None
    retrain, init_path = cfg.training.retrain, cfg.training.init
    kwargs = {**(norm_kwargs or {}), **(model_kwargs or {})}

    verbose, seed = cfg.misc.verbose, cfg.misc.seed

    output_dir, save_name = cfg.misc.output_dir, cfg.misc.save_name, 
    save_name, save_dir = get_dirs(output_dir, save_name, model_name, norm_name, criterion_name, subset_kwargs.sizes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        logger.info(f"Fetched main configs, save directory : {save_dir}")
        logger.info(f"Model {model_name}, norma {norm_name}, criterion {criterion_name}, kwargs {kwargs}")

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    #data
    loaders_dict, stats_dict, nodes_stats_dict = fetch_training_data(data_path, split_kwargs, subset_kwargs, batch_size, lags, horizon, clusters=clusters, seed=seed)
    if cfg.data.normalize:
        apply_stats(loaders_dict, stats_dict)
    shape, shape_str, batch_str = get_sizes(loaders_dict, str_info=True)
    if verbose:
        logger.info("Fetched dataloaders")
        logger.info(shape_str)
        logger.info(batch_str)

    #model
    if kwargs.get("init_alpha") is True:
        if "cmIN" in model_name:
            kwargs["init_alpha"] = [nodes_stats_dict[node]["train"]["alpha"] for node in nodes_stats_dict]
        else:
            kwargs["init_alpha"] = stats_dict["train"]["alpha"]
    if kwargs.get("init_beta") is True:
        if "cmIN" in model_name:
            kwargs["init_beta"] = [nodes_stats_dict[node]["train"]["alpha"] for node in nodes_stats_dict]
        else:
            kwargs["init_beta"] = stats_dict["train"]["beta"]
    model = load_model(model_name, shape, norm_name, init_path, cfg.training.freeze_core, cfg.model.constants, cfg.model.residuals, **kwargs)
    if cfg.training.freeze_core:
        trainable_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
        logger.info(f"Trainable params: {trainable_params}")
    
    #training
    logger.info("--Training--")
    learner = launch_training(model, norm_name, criterion, lr, epochs, loaders_dict, eval_losses, device, save_dir, save_name, eval_freq, print_freq, logger, retrain)
    logger.info("--Eval--")
    launch_eval(learner, loaders_dict, eval_losses, save_dir, save_name, complete_evaluation, results_dir=output_dir)
    launch_example(data_path, model, lags, horizon, device, save_dir, save_name)

    #weights
    plot_weights(model, learner, save_dir, save_name)
    if (norm_name is not None) and (("revin" in norm_name) or ("mIN" in norm_name)):
        params = {"beta": model.beta.data.detach().cpu().numpy()[0][0][0], "alpha": model.alpha.data.detach().cpu().numpy()[0][0][0]}
        logger.info(f"Final modulations: {params}")

    logger.info('End of script\n')

if __name__ == "__main__":
    run()



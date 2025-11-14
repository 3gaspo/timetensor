import hydra
import logging
import torch

from src.timetensor.dataset import fetch_training_data, get_sizes, apply_stats
from src.timetensor.models import load_model, format_kwargs
from src.timetensor.pipeline import get_losses, load_learner
from src.timetensor.visu import plot_weights
from src.timetensor.utils import get_dirs, set_seed

from src.timetensor.pipeline import launch_training, launch_eval, launch_example


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    logger.info("=====Running eval script=====")

    #configs
    data_path = cfg.data.path
    lags, horizon = int(cfg.task.lags), int(cfg.task.horizon)

    criterion_name = cfg.training.loss
    criterion, eval_losses = get_losses(criterion_name, complete_evaluation=cfg.misc.complete_evaluation)

    model_name, norm_name = cfg.model.name, cfg.normalization.name
    if norm_name == "None":
        norm_name = None
    kwargs = {**(cfg.normalization.configs or {}), **(cfg.model.configs or {})}

    verbose, seed = cfg.misc.verbose, cfg.misc.seed

    output_dir, save_name = cfg.misc.output_dir, cfg.misc.save_name, 
    save_name, save_dir = get_dirs(output_dir, save_name, model_name, norm_name, criterion_name, cfg.data.subsets.sizes)

    if verbose:
        logger.info(f"Fetched main configs, save directory : {save_dir}")
        logger.info(f"Model {model_name}, norm {norm_name}, criterion {criterion_name}, kwargs {kwargs}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    #data
    loaders_dict, stats_dict, nodes_stats_dict = fetch_training_data(data_path, cfg.data.splits, cfg.data.subsets, cfg.training.bs, lags, horizon, clusters=cfg.data.clustering.clusters, seed=seed)
    if cfg.data.normalize:
        apply_stats(loaders_dict, stats_dict)
    shape, shape_str, batch_str = get_sizes(loaders_dict, str_info=True)
    if verbose:
        logger.info("Fetched dataloaders")
        logger.info(shape_str)
        logger.info(batch_str)

    #model
    format_kwargs(kwargs, norm_name, nodes_stats_dict, stats_dict, logger)
    model = load_model(model_name, shape, norm_name, cfg.training.init, cfg.training.freeze_core, cfg.model.constants, cfg.model.residuals, **kwargs)
    
    #training
    logger.info("--Training--")
    learner = load_learner(model, norm_name, criterion, cfg.training.lr, eval_losses, device)

    logger.info("--Eval--")
    launch_eval(learner, loaders_dict, stats_dict, eval_losses, save_dir, save_name, cfg.misc.complete_evaluation, results_dir=output_dir, mode="Test", denormalize=cfg.data.normalize)
    launch_example(data_path, model, lags, horizon, device, save_dir, save_name)

    #weights
    plot_weights(model, save_dir + "plots/", save_name)
    if (norm_name is not None) and (("revin" in norm_name) or ("mIN" in norm_name and "cmIN" not in norm_name)):
        params = {"beta": model.beta.data.detach().cpu().numpy()[0][0][0], "alpha": model.alpha.data.detach().cpu().numpy()[0][0][0]}
        logger.info(f"Final modulations: {params}")
    elif (norm_name is not None and "cmIN" in norm_name):
        params = {f"beta_{k}": value.data.detach().cpu().numpy()[0][0][0] for k,value in enumerate(model.betas)}
        logger.info(f"Final modulations: {params}")

    logger.info('End of script\n')

if __name__ == "__main__":
    run()



import hydra
import logging
import torch

from src.timetensor.dataset import fetch_training_data, get_sizes, apply_standard_norm
from src.timetensor.models import load_model
from src.timetensor.pipeline import get_losses, launch_training, launch_eval, launch_example
from src.timetensor.utils import get_dirs, set_seed

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    logger.info("=====Running train script=====")

    #configs
    data_path = cfg.data.path
    lags, horizon = int(cfg.task.lags), int(cfg.task.horizon)

    criterion_name = cfg.training.loss
    criterion, eval_losses = get_losses(criterion_name, complete_evaluation=cfg.training.complete_evaluation)

    model_name, norm_name = cfg.model.name, cfg.normalization.name
    if norm_name == "None":
        norm_name = None
    kwargs = {**(cfg.normalization.configs or {}), **(cfg.model.configs or {})}

    verbose, seed = cfg.misc.verbose, cfg.misc.seed

    output_dir, save_name = cfg.misc.output_dir, cfg.misc.save_name, 
    save_name, save_dir = get_dirs(output_dir, save_name, model_name, norm_name, criterion_name, cfg.data.subsets)

    if verbose:
        logger.info(f"Fetched main configs, save directory : {save_dir}")
        logger.info(f"Model {model_name}, norm {norm_name}, criterion {criterion_name}, kwargs {kwargs}")

    device = torch.device("cuda" if cfg.misc.device=="gpu" and torch.cuda.is_available() else "cpu")
    set_seed(seed)

    #data
    loaders_dict, stats_dict = fetch_training_data(
        data_path, cfg.data.splits, cfg.data.sampling, cfg.data.subsets,
        cfg.training.bs, lags, horizon,
        seed=seed)
    if cfg.data.normalize:
        apply_standard_norm(loaders_dict, stats_dict)
    shape, shape_str, batch_str = get_sizes(loaders_dict, str_info=True)
    if verbose:
        logger.info("Fetched dataloaders")
        logger.info(shape_str)
        logger.info(batch_str)

    #model
    model = load_model(model_name, shape, norm_name, cfg.training.init, cfg.model.do_constants, device.type=="cpu", **kwargs)
    if verbose:
        logger.info("Fetched model")

    #training
    logger.info("--Training--")
    learner = launch_training(model,
        norm_name, criterion, cfg.training.lr, cfg.training.epochs, loaders_dict, eval_losses, device,
        save_dir, save_name, cfg.training.eval_freq, cfg.training.print_freq, logger, save=True)
    
    #eval
    logger.info("--Eval--")
    if cfg.training.valid_eval:
        _ = launch_eval(learner, loaders_dict, eval_losses, save_dir, save_name, cfg.training.complete_evaluation, results_dir=output_dir, mode="valid", runs=cfg.training.eval_runs)

    if cfg.training.test_eval:
        _ = launch_eval(learner, loaders_dict, eval_losses, save_dir, save_name, cfg.training.complete_evaluation, results_dir=output_dir, mode="test", runs=cfg.training.eval_runs)
        launch_example(data_path, model, lags, horizon, device, save_dir, save_name, use_context=cfg.data.sampling.use_context)

    logger.info('End of script\n')

if __name__ == "__main__":
    run()



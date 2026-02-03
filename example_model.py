import hydra
import logging
import torch
from time import perf_counter

from src.timetensor.dataset import fetch_training_data, get_sizes, apply_standard_norm
from src.timetensor.models import load_model
from src.timetensor.pipeline import get_losses, load_learner
from src.timetensor.visu import plot_weights
from src.timetensor.utils import get_dirs, set_seed

from src.timetensor.pipeline import launch_eval, launch_example

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

    model_name, norm_name = cfg.model.name, cfg.normalization.name
    if norm_name == "None":
        norm_name = None
    kwargs = {**(cfg.normalization.configs or {}), **(cfg.model.configs or {})}

    verbose, seed = cfg.misc.verbose, cfg.misc.seed

    output_dir, save_name = cfg.misc.output_dir, cfg.misc.save_name
    save_name, save_dir = get_dirs(output_dir, save_name, model_name, norm_name, criterion_name, cfg.data.subsets)

    if verbose:
        logger.info(f"Fetched main configs, save directory : {save_dir}")
        logger.info(f"Model {model_name}, norm {norm_name}, criterion {criterion_name}, kwargs {kwargs}")

    device = torch.device("cuda" if cfg.misc.device=="gpu" and torch.cuda.is_available() else "cpu")
    set_seed(seed)

    #model
    model = load_model(model_name, (lags,1,horizon), norm_name, cfg.training.init, device.type=="cpu", **kwargs)
    if verbose:
        logger.info("Fetched model and learner")

    #example
    logger.info("--Example--")
    launch_example(data_path, model, lags, horizon, device, save_dir, save_name, cfg.data.sampling.use_context)

    #weights
    plot_weights(model, save_dir + "plots/", save_name)

    logger.info('End of script\n')

if __name__ == "__main__":
    run()



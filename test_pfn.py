import hydra
import logging
import torch
from time import perf_counter

from src.timetensor.models import load_model
from src.timetensor.visu import plot_weights, plot_2D
from src.timetensor.utils import get_dirs, set_seed

from src.timetensor.pipeline import launch_example

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    logger.info("=====Running eval script=====")

    #configs

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
    model = load_model(model_name, (1344,1,336), norm_name, cfg.training.init, device.type=="cpu", **kwargs)
    if verbose:
        logger.info("Fetched model")

    #example
    logger.info("--Example--")
    x = torch.full((100,1,1344), 1)
    print("x shape: ", x.shape)

    features = model._generate_time_features(1344, 1344+336, device, x.dtype)
    print("features shape: ", features.shape)
    plot_2D(features.numpy().T, save_dir, name="features.pdf", title='Time features', x_name="x", y_name="y")


    Xtrain, y_train, X_test = model._prepare_matrix(x, features, None, None)
    print("Xtrain shape: ", Xtrain.shape)
    print("y_train shape: ", y_train.shape)
    print("X_test shape: ", X_test.shape)

    t1 = perf_counter()
    preds = model(x)
    t2 = perf_counter()
    print(f"Preds done in {(t2-t1):.2f} s")

    logger.info('End of script\n')

if __name__ == "__main__":
    run()



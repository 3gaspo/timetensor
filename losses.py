import hydra
import logging
import os
import torch

from src.timetensor.visu import plot_multi_losses

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


@hydra.main(version_base=None, config_path="config", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    print("\n")
    logger.info("=====Running data script=====")

    #configs
    output_dir = cfg.misc.output_dir
    logger.info("Fetched configs")
 
    expe_names = [name for name in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, name))]

    losses_dict = {}
    Nlosses_dict = {}

    for expe_name in expe_names:
        if expe_name in ["linear", "linear_revin"]:
            valid_losses = torch.load(output_dir + expe_name + "/" + "MSE_valid_losses.pt")
            Nvalid_losses = torch.load(output_dir + expe_name + "/" + "NMSE_valid_losses.pt")
            losses_dict[expe_name] = valid_losses
            Nlosses_dict[expe_name] = Nvalid_losses

    plot_multi_losses(losses_dict, output_dir, "losses.pdf", f"Valid losses of Linear model")
    plot_multi_losses(Nlosses_dict, output_dir, "normal_losses.pdf", f"Normalized valid losses of Linear model")

    logger.info('End of script\n')

if __name__ == "__main__":
    run()



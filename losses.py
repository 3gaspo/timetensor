import hydra
import logging
import os
import torch

from src.timetensor.visu import plot_multi_losses
from src.timetensor.utils import append_in_dict

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


@hydra.main(version_base=None, config_path="config", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    print("\n")
    logger.info("=====Running loss script=====")

    #configs
    output_dir = cfg.misc.output_dir
    logger.info("Fetched configs")
 
    expe_names = [name for name in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, name)) and name not in ["persistence", "repeat", "lookback", "sklinear"]]

    if len(expe_names) >0:
        losses_dict = {}
        losses_dict2 = {}

        for expe_name in expe_names:
            valid_losses = torch.load(output_dir + expe_name + "/" + "valid_losses.pt", weights_only=False)
            valid_losses2 = torch.load(output_dir + expe_name + "/" + "valid_losses2.pt", weights_only=False)


            for loss_name in valid_losses:
                if loss_name not in losses_dict:
                    losses_dict[loss_name] = {}
                    losses_dict2[loss_name] = {}
                losses_dict[loss_name][expe_name] = valid_losses[loss_name]
                losses_dict2[loss_name][expe_name] = valid_losses2[loss_name]

        for loss_name in valid_losses:
            plot_multi_losses(losses_dict[loss_name], output_dir, f"{loss_name}_valid.pdf", f"Valid {loss_name}")
            plot_multi_losses(losses_dict2[loss_name], output_dir, f"{loss_name}_valid2.pdf", f"Valid2 {loss_name}")
        logger.info('Plotted multi losses')
    else:
        logger.info('No losses to plot')

    logger.info('End of script\n')

if __name__ == "__main__":
    run()



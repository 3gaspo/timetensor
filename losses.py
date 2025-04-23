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
 
    expe_names = [name for name in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, name)) and name not in ["persistence", "repeat", "lookback", "sklinear"]]

    if len(expe_names) >0:
        losses_dict = {}
        Nlosses_dict = {}
        losses_dict2 = {}
        Nlosses_dict2 = {}

        for expe_name in expe_names:
            valid_losses = torch.load(output_dir + expe_name + "/" + "MSE_valid_losses.pt")
            Nvalid_losses = torch.load(output_dir + expe_name + "/" + "NMSE_valid_losses.pt")
            losses_dict[expe_name] = valid_losses
            Nlosses_dict[expe_name] = Nvalid_losses

            valid_losses2 = torch.load(output_dir + expe_name + "/" + "MSE_valid_losses2.pt")
            Nvalid_losses2 = torch.load(output_dir + expe_name + "/" + "NMSE_valid_losses2.pt")
            losses_dict2[expe_name] = valid_losses2
            Nlosses_dict2[expe_name] = Nvalid_losses2

        plot_multi_losses(losses_dict, output_dir, "losses.pdf", f"Valid losses")
        plot_multi_losses(Nlosses_dict, output_dir, "normal_losses.pdf", f"Normalized valid losses")
        plot_multi_losses(losses_dict, output_dir, "losses2.pdf", f"Valid2 losses")
        plot_multi_losses(Nlosses_dict, output_dir, "normal_losses2.pdf", f"Normalized valid2 losses")
        logger.info('Plotted multi losses')
    else:
        logger.info('No losses to plot')

    logger.info('End of script\n')

if __name__ == "__main__":
    run()



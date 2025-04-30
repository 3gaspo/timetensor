import hydra
import os

from src.timetensor.visu import print_nice_table

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


@hydra.main(version_base=None, config_path="config", config_name="config")
def run(cfg):
    
    output_dir = cfg.misc.output_dir
    multipliers = cfg.misc.table_coeffs
    if type(multipliers) == str:
        multipliers = multipliers.split(" ")
        multipliers = [int(w) for w in multipliers]
    print_nice_table(output_dir + "mean_results.json", multipliers=multipliers)

if __name__ == "__main__":
    run()



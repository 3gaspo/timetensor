## Rebuilds dataset from raw data, including splits and subsets, and plots stats and examples

import hydra
import logging
from time import perf_counter
import os

from src.timetensor.dataset import get_train_loaders, build_dataset, get_dataset_splits
from src.timetensor.utils import set_random_data
from src.timetensor.visu import plot_named_example, scatter_stats, plot_stats, scatter_input_output
from src.timetensor.utils import unroll_windows

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


@hydra.main(version_base=None, config_path="config", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    logger.info("=====Running data script=====")

    #configs
    data_path = cfg.data.path
    verbose = cfg.misc.verbose
    lags, horizon = cfg.model.lags, cfg.model.horizon
    batch_size = cfg.training.bs

    if verbose:
        logger.info("Fetched configs")

    rebuild_pt=True
    new_example=False
    replot=True
    
    for byname in ["by_date", "by_indiv"]:
        for folder_name in ["plots/", "subsets/"]:
            if not os.path.exists(data_path + folder_name + byname + "/"):
                os.makedirs(data_path + folder_name + byname + "/")

    #dataset
    if rebuild_pt:
        if verbose:
            logger.info("Rebuilding dataset")
            t1 = perf_counter()
        dataset = cfg.data.dataset
        if dataset == "electricity":
            from src.timetensor.electricity import fetch_data  #adapt path if script in another working directory
            fetcher = lambda path: fetch_data(path, raw_format=cfg.data.format, years=None, hourly=None)
        else:
            "Dataset name not recognized"
        build_dataset(fetcher, data_path) #builds dataset values from raw data and saves as .pt
        if verbose:
            t2 = perf_counter()
            logger.info(f"Build in {(t2-t1)/60:.3f} min")

        #splits
        data_dict = get_dataset_splits(data_path, cfg.data.indiv_split, cfg.data.date_split, cfg.misc.seed, save=False) #save will save the train test indices, in path
        
        #subsets
        subsets={"train":0.1, "valid":0.1, "valid2":0.1, "test":0.1}
        for by_date in [True, False]:
            #will generate the subsets and save indices
            byname="by_date" if by_date else "by_indiv"
            subset_mode="dates" if by_date else "individuals"
            partial_loaders_dict = get_train_loaders(data_dict, batch_size, lags, horizon, by_date=by_date, subsets=subsets, subset_mode=subset_mode, path=data_path+f"subsets/{byname}/")
            full_loaders_dict = get_train_loaders(data_dict, batch_size, lags, horizon, by_date=by_date)
            unrolls = {"full": unroll_windows(full_loaders_dict["train"]), "subset": unroll_windows(partial_loaders_dict["train"])}
            x_dict = {key: unrolls[key][0] for key in unrolls}
            y_dict = {key: unrolls[key][1] for key in unrolls}            
            scatter_input_output(x_dict, y_dict, data_path+"plots/"+byname+"/", name=f"subset.pdf")
            
    #example
    if new_example:
        if verbose:
            logger.info("Setting new example")
        set_random_data(data_path, lags, horizon, name="rand")
        plot_named_example(data_path + "/examples/", "rand")

    if replot:
        data_dict = get_dataset_splits(data_path, cfg.data.indiv_split, cfg.data.date_split, cfg.misc.seed, save=False) #save will save the train test indices, in path
        for by_date in [True, False]:
            byname="by_date" if by_date else "by_indiv"
            loaders_dict = get_train_loaders(data_dict, batch_size, lags, horizon, by_date=by_date)
            
            #sizes
            if verbose:
                
                logger.info(f"Sizes for {byname}")
                logger.info(f"Training data : {loaders_dict['train'].dataset.shape}")
                X, c, y = next(iter(loaders_dict["train"])) # (indiv, dim, lags),  #(nc, dim, horizon),  #(indiv, dim, horizon)
                if c is not None:
                    logger.info(f"Batch : X={X.shape}, c={c.shape}, y={y.shape}")
                else:
                    logger.info(f"Batch : X={X.shape}, y={y.shape}")

            #stats
            if verbose:
                logger.info("Plotting stats")

            unrolls = {key: unroll_windows(loaders_dict[key]) for key in ["train", "test"]}
            nunrolls = {key: unroll_windows(loaders_dict[key], normal=True) for key in ["train", "test"]}
            x_dict, y_dict = {key: unrolls[key][0] for key in unrolls}, {key: unrolls[key][1] for key in unrolls}
            nx_dict, ny_dict =  {key: nunrolls[key][0] for key in nunrolls}, {key: nunrolls[key][1] for key in nunrolls}

            scatter_input_output(x_dict, y_dict, data_path+"plots/"+byname+"/", name="output_inputs.pdf")
            scatter_stats(x_dict, data_path+"plots/"+byname+"/", name="inputs_stats.pdf", title="Inputs statistics")
            plot_stats(nx_dict, data_path+"plots/"+byname+"/", name="normal_inputs.pdf", title="Normalized inputs distribution", logscale=False, limits=(-1e-6,1e-6))
            plot_stats(ny_dict, data_path+"plots/"+byname+"/", name="normal_outputs.pdf", title="Normalized outputs distribution", logscale=False, limits=(-5,5))

    logger.info('End of script\n')

if __name__ == "__main__":
    run()


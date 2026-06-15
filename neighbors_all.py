## Adding both neighboring past windows of user and neighbors, for each user as context

import hydra
import logging
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from time import perf_counter

from src.timetensor.dataset import fetch_csv
from src.timetensor.models import load_model
from src.timetensor.pipeline import Loss
from src.timetensor.utils import get_dirs, set_seed, get_normal_stats, save_results, symlog, normalize
from src.timetensor.analysis import get_fourier
from src.timetensor.visu import plot_series

import faiss
import psutil

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    logger.info("=====Running cross learning clusters script=====")

    #configs
    data_path = cfg.data.path
    lags, horizon = int(cfg.task.lags), int(cfg.task.horizon)

    criterion = Loss(nn.MSELoss(reduction="none")) #, mode="instance") #TODO : change to MSE?

    model_name, norm_name = cfg.model.name, cfg.normalization.name
    if norm_name == "None":
        norm_name = None
    kwargs = {**(cfg.normalization.configs or {}), **(cfg.model.configs or {})}

    verbose, seed = cfg.misc.verbose, cfg.misc.seed

    output_dir, save_name = cfg.misc.output_dir, cfg.misc.save_name
    save_name, save_dir = get_dirs(output_dir, save_name, model_name, norm_name)

    if verbose:
        logger.info(f"Fetched main configs, save directory : {save_dir}")
        logger.info(f"Model {model_name}, norm {norm_name}, kwargs {kwargs}")

    device = torch.device("cuda" if cfg.misc.device == "gpu" and torch.cuda.is_available() else "cpu")
    logger.info(f"Device memory: {round(psutil.virtual_memory().total / 1024**3,3)} GB")

    set_seed(seed)

    #model
    model = load_model(model_name, (lags, 1, horizon), norm_name, cfg.training.init, device.type=="cpu", **kwargs)
    model.eval()
    improvement_list = []
    distance_list = []
    with torch.inference_mode():

        #data
        data, _, datetimes = fetch_csv(data_path, cfg.data.dataset, drop_users=cfg.data.splits.drop_users, aggr=cfg.data.aggregation)
        data = data.reset_index(drop=True)

        all_indiv = list(range(data.shape[1]))
        individuals = len(all_indiv)
        bs = cfg.training.bs
        is_context = (bs > 1)

        dates = len(datetimes)
        assert dates % 24 == 0, f"Dataset is not hourly! {(dates, datetimes[0], datetimes[1])}"

        if verbose:
            logger.info("Fetched data csv")
            logger.info(f"Shape (dates, indiv): {data.shape}")

        #splits
        date_split = float(cfg.data.splits.date_splits.split(";")[0]) #train/test ratio
        split_date_idx = int(date_split * dates)
        max_start_idx = dates - (lags + horizon)
        logger.info(f"Eval dates: {datetimes[split_date_idx]} - {datetimes[max_start_idx]}")

        eval_stride = int(cfg.data.sampling.eval_stride)
        train_stride = int(cfg.data.sampling.train_stride)

        eval_strided_dates = np.array(range(split_date_idx, max_start_idx + 1, eval_stride))
        logger.info(f"Eval windows: {len(eval_strided_dates)} dates, {len(eval_strided_dates)*individuals} total")

        #train datastore
        max_train_windows = cfg.extra.max_windows
        max_train_start = split_date_idx - (lags + horizon) - 1

        def get_train_strided_dates(current_eval_idx, train_stride, individuals, lags, horizon, max_train_windows=None, max_train_start=None):            
            """returns list of strided dates with satisfied constraints"""
            
            eval_hour=0
            if current_eval_idx is not None:
                eval_hour = current_eval_idx % 24

            if max_train_start is None:
                assert current_eval_idx is not None, "No max_train_start or current_eval_idx provided."
                max_train_start = current_eval_idx - (lags+horizon) - 1
                assert max_train_start >= eval_hour, f"Not enough dates to satisfy max_windows at eval idx {current_eval_idx}."

            if max_train_windows is not None:
                allowed_date_steps = max_train_windows // individuals
                proposed_start = max_train_start - (allowed_date_steps - 1)*train_stride
                proposed_start = max(proposed_start, eval_hour)
                proposed_start -= (proposed_start - eval_hour) % 24
            else:
                proposed_start = eval_hour

            train_strided_dates = np.arange(proposed_start, max_train_start + 1, train_stride)
            if max_train_windows is not None:
                train_strided_dates = train_strided_dates[-allowed_date_steps:]
            assert len(train_strided_dates) > 0, "Not enough dates!"

            return train_strided_dates
        
        train_strided_dates = get_train_strided_dates(0, train_stride, individuals, lags, horizon, max_train_windows, max_train_start)
        logger.info(f"Train windows: {len(train_strided_dates)} dates, {len(train_strided_dates)*individuals} total")


        distance_kws = cfg.extra.distance.split("_")
        if len(distance_kws) == 2:
            distance_space, distance_metric = distance_kws[0], distance_kws[1]
        else:
            distance_space, distance_metric = "raw", distance_kws[0]
        
        def get_windows_representations(data, strided_dates, distance_space, device=None, normal=True, verbose=0):
            """returns train datastore as window features and values matrices"""
            feature_idxs = strided_dates[:, None] + np.arange(lags) # (len(strided_dates), lags)
            value_idxs = strided_dates[:, None] + np.arange(lags + horizon) # (len(strided_dates), lags+horizon)

            X_context = data.values[feature_idxs].transpose(2, 0, 1).reshape(-1, lags) # (len(strided_dates)*individuals, lags)
            X_windows = data.values[value_idxs].transpose(2, 0, 1).reshape(-1, lags + horizon) # (len(strided_dates)*individuals, lags+horizon)

            if normal:
                mean, std = np.mean(X_context,axis=1, keepdims=True), np.std(X_context,axis=1, keepdims=True)
                X_features_raw = normalize(X_context, mean, std)
                X_windows = normalize(X_windows, mean, std)
            else:
                X_features_raw = X_context

            if verbose == 1:
                logger.info(f"Built raw features: shape {X_features_raw.shape}, elements {X_features_raw.size}, {round(X_features_raw.nbytes / 1024**2,2)} MB")
            
            if distance_space == "fourier": 
                X_features = get_fourier(X_features_raw).astype(np.float32) #  (len(strided_dates)*individuals, n_freqs)
            elif distance_space == "chronos":
                if verbose == 1:
                    logger.info(f"Building Chronos representation")
                
                X_features = torch.from_numpy(X_features_raw).float().unsqueeze(1)
                if device is not None:
                    X_features = X_features.to(device)
                X_features = model.representation(X_features, pool=False)
                X_features = X_features.cpu().numpy().astype("float32") # (len(strided_dates)*individuals, n_chrs)
            else:
                X_features = np.ascontiguousarray(X_features_raw, dtype=np.float32)

            X_windows = torch.from_numpy(X_windows).float()

            if verbose == 1:
                logger.info(f"Built features: shape {X_features.shape}, elements {X_features.size}, {round(X_features.nbytes / 1024**2,2)} MB")
            return X_features, X_windows

        if is_context:
            train_features, train_windows = get_windows_representations(data, train_strided_dates, distance_space, device=device, verbose=1)

        def fit_kNN(X_features, distance_space, distance_metric, verbose=0):
            """returns FAISS kNN"""
            N, d = X_features.shape[0], X_features.shape[1]

            if distance_space == "chronos":
                nlist = min(int(4 * np.sqrt(N)), 100) #int(4 * np.sqrt(N)) #number of centroids
                nbytes, nprobe = 8, 5

                def get_valid_m(d, target_m=64): #number of sub_vectors (dim quantization)
                    """Find the divisor of d closest to target_m"""
                    for offset in range(0, target_m):
                        # Check upwards and downwards
                        for sign in [1, -1]:
                            m = target_m + (offset * sign)
                            if m > 0 and d % m == 0:
                                return m
                    return 1 # Fallback to 1 (Slow/High Memory, but won't crash)
                m = get_valid_m(d, target_m=64)

                if distance_metric == "euclidean":
                    # quantizer = faiss.IndexFlatL2(d)
                    # index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbytes, faiss.METRIC_L2)
                    index = faiss.IndexPQ(d, m, nbytes, faiss.METRIC_L2)
                else: 
                    # quantizer = faiss.IndexFlatIP(d)
                    if distance_metric == "cosine":
                        faiss.normalize_L2(X_features) # transform L2 norm from sqrt(L) to 1.0
                    elif distance_metric == "pearson":
                        X_features -= X_features.mean(axis=1, keepdims=True)
                        faiss.normalize_L2(X_features)
                    # index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbytes, faiss.METRIC_INNER_PRODUCT)
                    index = faiss.IndexPQ(d, m, nbytes, faiss.METRIC_INNER_PRODUCT)

                if verbose == 1:
                    # logger.info(f"Fitting FAISS kNN with N={N}, d={d}, nlist={nlist}, m={m}, nbytes={nbytes}, nprobe={nprobe}")
                    logger.info(f"Fitting FAISS kNN with N={N}, d={d}, m={m}, nbytes={nbytes}")

                index.train(X_features)
                index.add(X_features)
                # index.nprobe = nprobe

            else: #note: with IN, raw d_cos = d_pears = d_eucl^2 / 2L
                if distance_metric == "euclidean": 
                    index = faiss.IndexFlatL2(d)
                elif distance_metric == "cosine":
                    faiss.normalize_L2(X_features)
                    index = faiss.IndexFlatIP(d)
                elif distance_metric == "pearson":
                    X_features -= X_features.mean(axis=1, keepdims=True)
                    faiss.normalize_L2(X_features)
                    index = faiss.IndexFlatIP(d)
                else:
                    raise ValueError(f"Unknown distance metric: {distance_kws}")
                if verbose == 1:
                    logger.info(f"Fitting exact kNN")
                index.add(X_features)

            if verbose == 1:
                logger.info(f"Done fitting kNN.")

            return index

        if is_context:
            train_index = fit_kNN(train_features, distance_space, distance_metric, verbose=1)

        def predict_kNN(X_features, store_index, distance_metric, k, verbose=0):
            """runs the kNN prediction and returns distances"""
            if distance_metric == "cosine":
                faiss.normalize_L2(X_features)
            elif distance_metric == "pearson":
                X_features -= X_features.mean(axis=1, keepdims=True)
                faiss.normalize_L2(X_features)
            
            if verbose == 1:
                logger.info(f"Predicting FAISS kNN")
                
            metric, indices = store_index.search(X_features, k)
            if distance_metric in ["cosine", "pearson"]:
                distances = 1 - metric
            else:
                distances = metric
            return distances, indices

        indiv_losses = {indiv: [] for indiv in range(individuals)}
        per_user_losses, stds_per_user_losses = [], []
        indiv_improvements = {indiv: [] for indiv in range(individuals)}
        per_user_improvements, stds_per_user_improvements = [], []

        improve_counts, total_counts = 0, 0
        
        t1 = perf_counter()
        logger.info(f"Starting eval loop.")
        for i, stride_date_idx in enumerate(range(len(eval_strided_dates))):
            t = eval_strided_dates[stride_date_idx]

            if not is_context:
                X, Y = data.iloc[t: t+lags, :].values.T, data.iloc[t+lags: t+lags+horizon, :].values.T # X: (individuals, lags)
                X, Y = torch.from_numpy(X).float().unsqueeze(1), torch.from_numpy(Y).float().unsqueeze(1)
                mean, std = get_normal_stats(X)
                X, Y = normalize(X, mean, std), normalize(Y, mean, std)
            else:
                if i==0:
                    logger.info(f"Building date representation")
                X_features, X_windows = get_windows_representations(data, np.array([t]), distance_space)
                if cfg.extra.online:
                    store_strided_dates = get_train_strided_dates(t, train_stride, individuals, lags, horizon, max_train_windows, None)
                    store_features, store_values = get_windows_representations(data, store_strided_dates, distance_space)
                    store_index = fit_kNN(store_features, distance_space, distance_metric)
                else:
                    store_strided_dates = get_train_strided_dates(t, train_stride, individuals, lags, horizon, max_train_windows, max_train_start)
                    store_features, store_values = get_windows_representations(data, store_strided_dates, distance_space)
                    store_index = fit_kNN(store_features, distance_space, distance_metric)
                if i==0:
                    logger.info(f"Starting search on X_features: {X_features.shape}")
                distances, indices = predict_kNN(X_features, store_index, distance_metric, bs-1)
                if i==0:
                    logger.info("Done search")
                X, Y = X_windows[:, :lags].unsqueeze(1), X_windows[:, lags:].unsqueeze(1)

            if is_context:
                if bs-1 >= len(store_values):
                    Xc = store_values.unsqueeze(0).expand(individuals, -1, -1) # (individuals, train_strided_dates * individuals, lags+horizon)
                else:
                    Xc = store_values[indices] #(individuals, bs-1, lags+horizon)

            if i==0 and is_context:
                plot_series({"target": X[0][0], "neighbor": Xc[0][0]}, save_dir, "neighbors.pdf", "Example neighbors")

            X, Y = X.to(device), Y.to(device)   
            if i==0:
                logger.info(f"Running vanilla pred")

            pred = model(X)
            loss = criterion(pred, Y) # (individuals, dim, H)
            loss = loss.mean(dim=1).mean(dim=1).cpu().numpy()  # (individuals)
            if is_context:
                Xc = Xc.to(device)
                if i==0:
                    logger.info(f"Running context pred")
                pred_context = model(X, Xc)
                loss_context = criterion(pred_context, Y)
                loss_context = loss_context.mean(dim=1).mean(dim=1).cpu().numpy()
                if i==0:
                    logger.info(f"Done context pred")             
                
                improvement = 100 * (loss - loss_context) / loss
                distance = np.mean(distances, axis=1)
                if cfg.extra.oracle: #replace better loss
                    for k in range(len(loss)):
                        if improvement[k]>0:
                            improve_counts += 1
                            loss[k] = loss_context[k]
                        else:     
                            improvement[k]=0       
                else:
                    if cfg.extra.thresh > 0:
                        for k in range(len(loss)):
                            if distance[k] < cfg.extra.thresh: 
                                improve_counts += 1
                                loss[k] = loss_context[k]
                    else:
                        loss = loss_context
                
                total_counts += len(loss)
                improvement_list += list(improvement)
                distance_list += list(distance)

            for k in range(len(loss)):
                indiv_losses[k].append(loss[k])
                if is_context:
                    indiv_improvements[k].append(improvement[k])

        if is_context:
            improve_ratio = 100 * (improve_counts / total_counts)
        else:
            improve_ratio = 0

        for indiv in all_indiv:
            indiv_loss = indiv_losses[indiv]
            mean = symlog(np.mean(indiv_loss))
            std = symlog(np.std(indiv_loss))
            per_user_losses.append(mean)
            stds_per_user_losses.append(std)

            if is_context:
                indiv_improvement = indiv_improvements[indiv]
                mean_improvement = symlog(np.mean(indiv_improvement))
                std_improvement = symlog(np.std(indiv_improvement))
                per_user_improvements.append(mean_improvement)
                stds_per_user_improvements.append(std_improvement)

        total_means = np.mean(per_user_losses)
        w10_means = np.mean(np.partition(per_user_losses, int(len(per_user_losses)*0.9))[int(len(per_user_losses)*0.9):])

        t2 = perf_counter()
        delta_t = (t2-t1)/60

        save_results(improve_ratio, output_dir, f"mean_results.json", save_name, f"improvements")
        save_results(total_means, output_dir, f"mean_results.json", save_name, f"nMSE")
        save_results(w10_means, output_dir, f"mean_results.json", save_name, f"w10 nMSE")
        save_results(delta_t, output_dir, f"mean_results.json", save_name, f"eval time (min)")

        stats_df = pd.DataFrame({
            "log(mean_error)": per_user_losses,
            "log(std_error)": stds_per_user_losses}).dropna()
        fig = plt.figure()
        g = sns.jointplot(
            data=stats_df,
            x="log(mean_error)",
            y="log(std_error)",
            kind='scatter',
        )
        plt.suptitle(
            f"Per-user nMSE of {save_name} (mean:{total_means:.4f}, W10:{w10_means:.4f})",
            fontsize=20)   
        plt.tight_layout()
        plt.savefig(save_dir + "plots/" + f"user_errors.pdf")
        plt.close()

        if is_context:
            stats_df = pd.DataFrame({
                "log(mean_improvement)": per_user_improvements,
                "log(std_improvement)": stds_per_user_improvements}).dropna()
            fig = plt.figure()
            g = sns.jointplot(
                data=stats_df,
                x="log(mean_improvement)",
                y="log(std_improvement)",
                kind='scatter',
            )
            plt.suptitle(
                f"Per-user improvement of {save_name} (mean ratio:{improve_ratio:.4f})",
                fontsize=20)   
            plt.tight_layout()
            plt.savefig(save_dir + "plots/" + f"user_improvements.pdf")
            plt.close()

            plt.figure(figsize=(10, 7))
            plt.scatter(np.log(np.array(distance_list)+1e-8), improvement_list)
            plt.title("Improvements")
            plt.xlabel("log(distance)")
            plt.ylabel("improvement")
            plt.tight_layout()
            plt.savefig(save_dir + "plots/" + f"improvements.pdf")
            plt.close()

    logger.info('End of script\n')

if __name__ == "__main__":
    run()



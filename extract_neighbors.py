"""Extract prediction features from foundation model and neighbors"""

import hydra
import logging
import warnings
from pathlib import Path

import faiss
import numpy as np
import torch

from src.timetensor.analysis import get_fourier
from src.timetensor.dataset import fetch_csv
from src.timetensor.models import load_model
from src.timetensor.utils import get_dirs, set_seed
from src.timetensor.visu import plot_series

warnings.simplefilter(action="ignore", category=FutureWarning)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    logger.info(" ")
    logger.info("===== Running neighbors' features extraction script =====")

    ## Loading configs
    data_path = cfg.data.path
    lags = int(cfg.task.lags)
    horizon = int(cfg.task.horizon)
    period = int(cfg.extra.period)

    model_name, norm_name = cfg.model.name, cfg.normalization.name
    if norm_name == "None":
        norm_name = None
    kwargs = {**(cfg.normalization.configs or {}), **(cfg.model.configs or {})}

    verbose = cfg.misc.verbose
    seed = cfg.misc.seed
    output_dir = cfg.misc.output_dir
    save_name = cfg.misc.save_name

    _, save_dir = get_dirs(output_dir, save_name, model_name, norm_name)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if cfg.misc.device == "gpu" and torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Device: {device}")

    set_seed(seed)

    ## Loading model
    model = load_model(
        model_name,
        (lags, 1, horizon),
        norm_name,
        cfg.training.init,
        device.type == "cpu",
        **kwargs,
    )
    model.eval()

    with torch.inference_mode():

        ## Loading data
        data, _, datetimes = fetch_csv(
            data_path,
            cfg.data.dataset,
            drop_users=cfg.data.splits.drop_users,
            aggr=cfg.data.aggregation,
        )
        data = data.reset_index(drop=True)

        individuals = data.shape[1]
        n_neighbors = int(cfg.extra.neighbors)
        is_context = n_neighbors > 0

        dates = len(datetimes)
        # assert dates % period == 0, (
        #     f"Dataset length is not divisible by period={period}! "
        #     f"{(dates, datetimes[0], datetimes[1])}"
        # )

        if verbose:
            logger.info(f"Shape (dates, indiv): {data.shape}")

        ## Building temporal splits
        split_ratios = [float(x) for x in cfg.data.splits.date_splits.split(";")]
        assert len(split_ratios) == 3, "date_splits must contain exactly three ratios: T0;T1;T2"
        assert abs(sum(split_ratios) - 1.0) < 1e-6, f"date_splits must sum to 1, got {split_ratios}"

        t0_end = int(split_ratios[0] * dates)
        t1_end = int((split_ratios[0] + split_ratios[1]) * dates)
        t2_end = dates

        max_start_idx = dates - (lags + horizon)
        assert t0_end > 0, "T0 is empty."
        assert t1_end > t0_end, "T1 is empty."
        assert t2_end > t1_end, "T2 is empty."
        assert t0_end <= max_start_idx + 1, "T0 split too large for available windows."

        eval_stride = int(cfg.data.sampling.eval_stride)
        train_stride = int(cfg.data.sampling.train_stride)
        max_train_windows = cfg.extra.max_windows
        online = bool(cfg.extra.online)

        assert train_stride % period == 0, (
            f"train_stride={train_stride} must be a multiple of period={period} "
            "to preserve periodic alignment."
        )

        logger.info(
            f"T0: [0, {t0_end}), T1: [{t0_end}, {t1_end}), T2: [{t1_end}, {t2_end})"
        )

        ## Configuring retrieval distance
        distance_kws = cfg.extra.distance.split("_")
        if len(distance_kws) == 2:
            distance_space, distance_metric = distance_kws[0], distance_kws[1]
        else:
            distance_space, distance_metric = "raw", distance_kws[0]


        ## Main helper
        def get_query_stats(t):
            """Compute raw mean and std of the query lookback for all individuals."""
            x_raw = data.iloc[t:t + lags, :].values.T  # (individuals, lags)
            mu_x = torch.from_numpy(x_raw.mean(axis=1)).float()  # (individuals,)
            sigma_x = torch.from_numpy(x_raw.std(axis=1)).float()  # (individuals,)
            return mu_x, sigma_x
        
        def normalize(x, mean, std, eps=1e-8):
            return (x - mean) / (std + eps)

        def distances_to_weights(
            distances,
            indices,
            store_dates,
            query_t,
            individuals,
            period,
            alpha=1.0,
            beta=0.1,
            gamma=0.25,
            eps=1e-8,
        ):
            d = torch.from_numpy(distances).float()
            d_std = d.std(dim=-1, keepdim=True, unbiased=False)
            d_norm = (d - d.min(dim=-1, keepdim=True).values) / (d_std + eps)

            idx = torch.from_numpy(indices).long()
            n_store_dates = len(store_dates)

            neighbor_user_idx = idx // n_store_dates
            neighbor_date_pos = idx % n_store_dates

            store_dates_t = torch.from_numpy(store_dates).long()
            neighbor_start_dates = store_dates_t[neighbor_date_pos]

            query_user_idx = torch.arange(individuals).view(-1, 1)
            same_individual = (neighbor_user_idx == query_user_idx).float()

            delta_periods = (query_t - neighbor_start_dates).float() / period
            recency_penalty = torch.log1p(delta_periods.clamp_min(0.0))

            scores = (
                -alpha * d_norm
                -beta * recency_penalty
                +gamma * same_individual
            )

            weights = torch.softmax(scores, dim=-1)
            return weights, same_individual, delta_periods

        def weighted_residual_variance(e, w):
            # e: (n_eval, individuals, K, horizon)
            # w: (n_eval, individuals, K)
            w_exp = w.unsqueeze(-1)
            mean_e = (w_exp * e).sum(dim=2, keepdim=True)
            return (w_exp * (e - mean_e).pow(2)).sum(dim=2).mean(dim=-1)

        def get_windows_representations(data, strided_dates, distance_space, device=None, normal=True, verbose=0):
            """Build feature representations and normalized windows for start dates."""
            feature_idxs = strided_dates[:, None] + np.arange(lags)  # (S, lags)
            value_idxs = strided_dates[:, None] + np.arange(lags + horizon)  # (S, lags + horizon)

            x_context = data.values[feature_idxs].transpose(2, 0, 1).reshape(-1, lags)  # (individuals * S, lags)
            x_windows = data.values[value_idxs].transpose(2, 0, 1).reshape(-1, lags + horizon)  # (individuals * S, lags + horizon)

            if normal:
                mean = np.mean(x_context, axis=1, keepdims=True)  # (individuals * S, 1)
                std = np.std(x_context, axis=1, keepdims=True)  # (individuals * S, 1)
                x_features_raw = normalize(x_context, mean, std)  # (individuals * S, lags)
                # x_windows = normalize(x_windows, mean, std)
            else:
                x_features_raw = x_context  # (individuals * S, lags)

            if verbose == 1:
                logger.info(
                    f"Built raw features: shape {x_features_raw.shape}, "
                    f"elements {x_features_raw.size}, "
                    f"{round(x_features_raw.nbytes / 1024**2, 2)} MB"
                )

            if distance_space == "fourier":
                x_features = get_fourier(x_features_raw).astype(np.float32)  # (individuals * S, D_fourier)
            elif distance_space == "chronos":
                if verbose == 1:
                    logger.info("Building Chronos representation")
                x_features = torch.from_numpy(x_features_raw).float().unsqueeze(1)  # (individuals * S, 1, lags)
                if device is not None:
                    x_features = x_features.to(device)  # (individuals * S, 1, lags)
                x_features = model.representation(x_features, pool=kwargs["pool_representation"]) #CHANGED: to true  # (individuals * S, D_chronos)
                x_features = x_features.cpu().numpy().astype("float32")  # (individuals * S, D_chronos)
            else:
                x_features = np.ascontiguousarray(x_features_raw, dtype=np.float32)  # (individuals * S, lags)

            x_windows = torch.from_numpy(x_windows).float()  # (individuals * S, lags + horizon)

            if verbose == 1:
                logger.info(
                    f"Built features: shape {x_features.shape}, "
                    f"elements {x_features.size}, "
                    f"{round(x_features.nbytes / 1024**2, 2)} MB"
                )

            return x_features, x_windows

        def fit_kNN(x_features, distance_space, distance_metric, verbose=0):
            """Fit a FAISS index on the provided feature matrix."""
            n, d = x_features.shape  # x_features: (n, d)

            if distance_space == "chronos" and not kwargs["pool_representation"]:
                nbytes = 8

                def get_valid_m(d, target_m=64):
                    """Find a product-quantization subvector count dividing d."""
                    for offset in range(0, target_m):
                        for sign in [1, -1]:
                            m = target_m + (offset * sign)
                            if m > 0 and d % m == 0:
                                return m
                    return 1

                m = get_valid_m(d, target_m=64)

                if distance_metric == "euclidean":
                    index = faiss.IndexPQ(d, m, nbytes, faiss.METRIC_L2)
                else:
                    if distance_metric == "cosine":
                        faiss.normalize_L2(x_features)  # (n, d)
                    elif distance_metric == "pearson":
                        x_features -= x_features.mean(axis=1, keepdims=True)  # (n, d)
                        faiss.normalize_L2(x_features)  # (n, d)
                    index = faiss.IndexPQ(d, m, nbytes, faiss.METRIC_INNER_PRODUCT)

                if verbose == 1:
                    logger.info(f"Fitting FAISS PQ kNN with N={n}, d={d}, m={m}, nbytes={nbytes}")

                index.train(x_features)
                index.add(x_features)

            else:
                if distance_metric == "euclidean":
                    index = faiss.IndexFlatL2(d)
                elif distance_metric == "cosine":
                    faiss.normalize_L2(x_features)  # (n, d)
                    index = faiss.IndexFlatIP(d)
                elif distance_metric == "pearson":
                    x_features -= x_features.mean(axis=1, keepdims=True)  # (n, d)
                    faiss.normalize_L2(x_features)  # (n, d)
                    index = faiss.IndexFlatIP(d)
                else:
                    raise ValueError(f"Unknown distance metric: {distance_metric}")

                if verbose == 1:
                    logger.info("Fitting exact kNN")

                index.add(x_features)

            return index

        def predict_kNN(x_features, store_index, distance_metric, k):
            """Query a FAISS index and return distances and neighbor indices."""
            if distance_metric == "cosine":
                faiss.normalize_L2(x_features)  # (Q, D)
            elif distance_metric == "pearson":
                x_features -= x_features.mean(axis=1, keepdims=True)  # (Q, D)
                faiss.normalize_L2(x_features)  # (Q, D)

            distances, indices = store_index.search(x_features, k)  # distances: (Q, k), indices: (Q, k)
            if distance_metric in ["cosine", "pearson"]:
                distances = 1 - distances  # (Q, k)
            return distances, indices

        def get_period_eval_dates(period_start, period_end, stride):
            """Return valid query start dates fully contained in a target period."""
            last_valid_start = min(period_end - (lags + horizon), max_start_idx)
            if last_valid_start < period_start:
                return np.array([], dtype=int)
            return np.arange(period_start, last_valid_start + 1, stride)

        def get_store_dates_fixed(store_start, store_end):
            """Return datastore dates from a fixed historical interval."""
            last_valid_store = store_end - (lags + horizon)
            if last_valid_store < store_start:
                return np.array([], dtype=int)

            dates_arr = np.arange(store_start, last_valid_store + 1, train_stride)
            if max_train_windows is not None and len(dates_arr) > 0:
                allowed_date_steps = max_train_windows // individuals
                if allowed_date_steps > 0:
                    dates_arr = dates_arr[-allowed_date_steps:]
                else:
                    dates_arr = np.array([], dtype=int)
            return dates_arr

        def get_store_dates_online(query_t):
            """Return datastore dates using all valid history before the query date."""
            last_valid_store = query_t - (lags + horizon) - 1
            if last_valid_store < 0:
                return np.array([], dtype=int)

            eval_phase = query_t % period
            first_aligned = eval_phase
            last_aligned = last_valid_store - ((last_valid_store - eval_phase) % period)
            if last_aligned < first_aligned:
                return np.array([], dtype=int)

            dates_arr = np.arange(first_aligned, last_aligned + 1, train_stride)
            if max_train_windows is not None and len(dates_arr) > 0:
                allowed_date_steps = max_train_windows // individuals
                if allowed_date_steps > 0:
                    dates_arr = dates_arr[-allowed_date_steps:]
                else:
                    dates_arr = np.array([], dtype=int)
            return dates_arr

        def plot_random_neighbors(eval_dates, store_start, store_end):
            """Plot one random target lookback and all retrieved neighbor lookbacks."""
            if not is_context or len(eval_dates) == 0:
                logger.info("Skipping neighbor plot: no context neighbors or no eval dates.")
                return

            set_seed(seed)
            date_idx = torch.randint(0, len(eval_dates), (1,)).item()
            individual_idx = torch.randint(0, individuals, (1,)).item()

            t = int(eval_dates[date_idx])

            if online:
                store_dates = get_store_dates_online(t)
            else:
                store_dates = get_store_dates_fixed(store_start, store_end)

            if len(store_dates) == 0:
                logger.info(f"Skipping neighbor plot: no datastore windows available at query t={t}.")
                return

            x_features, x_windows = get_windows_representations(
                data, np.array([t]), distance_space
            )
            # x_features: (individuals, D)
            # x_windows: (individuals, lags + horizon)

            store_features, store_windows = get_windows_representations(
                data, store_dates, distance_space
            )
            # store_features: (individuals * S_store, D)
            # store_windows: (individuals * S_store, lags + horizon)

            store_index = fit_kNN(store_features, distance_space, distance_metric)
            distances, indices = predict_kNN(x_features, store_index, distance_metric, n_neighbors)
            # distances: (individuals, n_neighbors)
            # indices: (individuals, n_neighbors)

            x = x_windows[:, :lags]  # (individuals, lags)
            xy_c = store_windows[indices]  # (individuals, n_neighbors, lags + horizon)
            x_c = xy_c[:, :, :lags]  # (individuals, n_neighbors, lags)

            target = x[individual_idx].cpu().numpy().reshape(-1)  # (lags,)
            neighbors = x_c[individual_idx].cpu().numpy()  # (n_neighbors, lags)

            series = {"target": target}
            for j in range(n_neighbors):
                series[f"neighbor_{j + 1}"] = neighbors[j].reshape(-1)  # (lags,)

            plot_series(
                series,
                f"{save_dir}/",
                "neighbors.pdf",
                f"Example neighbors - date={datetimes[t]}, individual={individual_idx}",
            )
            logger.info(
                f"Saved neighbor plot to: {save_dir / 'neighbors.pdf'} "
                f"(date={datetimes[t]}, individual={individual_idx})"
            )


        def extract_period(eval_dates, prefix, store_start, store_end):
            """Extract compact features and predictions for one temporal period."""
            n_eval = len(eval_dates)
            logger.info(f"Building {prefix}: {n_eval} x {individuals} = {n_eval * individuals} windows")

            preds = torch.empty((n_eval, individuals, horizon), dtype=torch.float32)  # (n_eval, individuals, horizon)
            preds_context = torch.empty((n_eval, individuals, horizon), dtype=torch.float32)  # (n_eval, individuals, horizon)
            E_values = torch.empty((n_eval, individuals, n_neighbors, horizon), dtype=torch.float32)  # (n_eval, individuals, n_neighbors, horizon)
            Ec_values = torch.empty((n_eval, individuals, n_neighbors, horizon), dtype=torch.float32)  # (n_eval, individuals, n_neighbors, horizon)
            Y_values = torch.empty((n_eval, individuals, horizon), dtype=torch.float32)  # (n_eval, individuals, horizon)
            Yc_values = torch.empty((n_eval, individuals, n_neighbors, horizon), dtype=torch.float32)  # (n_eval, individuals, n_neighbors, horizon)
            mu_x_values = torch.empty((n_eval, individuals), dtype=torch.float32)  # (n_eval, individuals)
            sigma_x_values = torch.empty((n_eval, individuals), dtype=torch.float32)  # (n_eval, individuals)
            mu_xc_values = torch.empty((n_eval, individuals, n_neighbors), dtype=torch.float32)  # (n_eval, individuals, n_neighbors)
            sigma_xc_values = torch.empty((n_eval, individuals, n_neighbors), dtype=torch.float32)  # (n_eval, individuals, n_neighbors)
            w_x_xc_values = torch.empty((n_eval, individuals, n_neighbors), dtype=torch.float32)  # (n_eval, individuals, n_neighbors)
            same_individual_values = torch.empty((n_eval, individuals, n_neighbors), dtype=torch.float32)  # (n_eval, individuals, n_neighbors)
            delta_periods_values = torch.empty((n_eval, individuals, n_neighbors), dtype=torch.float32)  # (n_eval, individuals, n_neighbors)

            for i, t in enumerate(eval_dates):
                mu_x, sigma_x = get_query_stats(t)  # each: (individuals,)
                mu_x_values[i] = mu_x  # (individuals,)
                sigma_x_values[i] = sigma_x  # (individuals,)

                if not is_context:
                    x_np = data.iloc[t:t + lags, :].values.T  # (individuals, lags)
                    y_np = data.iloc[t + lags:t + lags + horizon, :].values.T  # (individuals, horizon)
                    x = torch.from_numpy(x_np).float().unsqueeze(1)  # (individuals, 1, lags)
                    y = torch.from_numpy(y_np).float()  # (individuals, horizon)

                    #CHANGED: no normalization of window
                    # mean = x.mean(dim=-1, keepdim=True)
                    # std = x.std(dim=-1, keepdim=True)
                    # x = normalize(x, mean, std)

                    pred = model(x.to(device)).detach().cpu().squeeze(1)  # (individuals, horizon)

                    preds[i] = pred  # (individuals, horizon)
                    preds_context[i] = pred  # (individuals, horizon)
                    E_values[i].zero_()  # (individuals, n_neighbors, horizon)
                    Ec_values[i].zero_()  # (individuals, n_neighbors, horizon)
                    Y_values[i] = y  # (individuals, horizon)
                    Yc_values[i].zero_()  # (individuals, n_neighbors, horizon)
                    mu_xc_values[i].zero_()  # (individuals, n_neighbors)
                    sigma_xc_values[i].zero_()  # (individuals, n_neighbors)
                    w_x_xc_values[i].zero_()  # (individuals, n_neighbors)
                    same_individual_values[i].zero_()  # (individuals, n_neighbors)
                    delta_periods_values[i].zero_()  # (individuals, n_neighbors)
                    continue

                if online:
                    store_dates = get_store_dates_online(t)
                else:
                    store_dates = get_store_dates_fixed(store_start, store_end)

                if len(store_dates) == 0:
                    raise ValueError(f"No datastore windows available for {prefix} at query t={t}")

                x_features, x_windows = get_windows_representations(
                    data, np.array([t]), distance_space
                )
                # x_features: (individuals, D)
                # x_windows: (individuals, lags + horizon)

                store_features, store_windows = get_windows_representations(
                    data, store_dates, distance_space
                )
                # store_features: (individuals * S_store, D)
                # store_windows: (individuals * S_store, lags + horizon)

                store_index = fit_kNN(store_features, distance_space, distance_metric)

                distances, indices = predict_kNN(x_features, store_index, distance_metric, n_neighbors)
                # distances: (individuals, n_neighbors)
                # indices: (individuals, n_neighbors)

                x = x_windows[:, :lags].unsqueeze(1)  # (individuals, 1, lags)
                y = x_windows[:, lags:]  # (individuals, horizon)
                xy_c = store_windows[indices]  # (individuals, n_neighbors, lags + horizon)
                x_c = xy_c[:, :, :lags]  # (individuals, n_neighbors, lags)
                y_c = xy_c[:, :, lags:]  # (individuals, n_neighbors, horizon)

                pred = model(x.to(device)).detach().cpu().squeeze(1)  # (individuals, horizon)
                pred_c = model(x.to(device), xy_c.to(device)).detach().cpu().squeeze(1)  # (individuals, horizon)
                x_c_flat = x_c.reshape(-1, lags).unsqueeze(1)  # (individuals * n_neighbors, 1, lags)
                pred_neighbors = model(x_c_flat.to(device)).detach().cpu().squeeze(1)  # (individuals * n_neighbors, horizon)
                pred_neighbors = pred_neighbors.reshape(individuals, n_neighbors, horizon)  # (individuals, n_neighbors, horizon)

                if n_neighbors > 1:
                    pred_neighbors_context = torch.empty_like(pred_neighbors)  # (individuals, n_neighbors, horizon)
                    for j in range(n_neighbors):
                        mask = torch.ones(n_neighbors, dtype=torch.bool)
                        mask[j] = False
                        xy_c_context = xy_c[:, mask, :]  # (individuals, n_neighbors - 1, lags + horizon)
                        pred_neighbors_context[:, j] = (
                            model(x_c[:, j:j + 1, :].to(device), xy_c_context.to(device))
                            .detach()
                            .cpu()
                            .squeeze(1)
                        )
                else:
                    pred_neighbors_context = pred_neighbors.clone()  # (individuals, n_neighbors, horizon)

                weights, same_individual, delta_periods = distances_to_weights(
                    distances=distances,
                    indices=indices,
                    store_dates=store_dates,
                    query_t=t,
                    individuals=individuals,
                    period=period,
                )

                preds[i] = pred  # (individuals, horizon)
                preds_context[i] = pred_c  # (individuals, horizon)
                E_values[i] = y_c.cpu() - pred_neighbors  # (individuals, n_neighbors, horizon)
                Ec_values[i] = y_c.cpu() - pred_neighbors_context  # (individuals, n_neighbors, horizon)
                Y_values[i] = y.cpu()  # (individuals, horizon)
                Yc_values[i] = y_c.cpu()  # (individuals, n_neighbors, horizon)
                mu_xc_values[i] = x_c.mean(dim=-1).cpu()  # (individuals, n_neighbors)
                sigma_xc_values[i] = x_c.std(dim=-1, unbiased=False).cpu()  # (individuals, n_neighbors)
                w_x_xc_values[i] = weights  # (individuals, n_neighbors)
                same_individual_values[i] = same_individual  # (individuals, n_neighbors)
                delta_periods_values[i] = delta_periods  # (individuals, n_neighbors)

                if verbose and (i == 0 or (i + 1) % 50 == 0 or (i + 1) == n_eval):
                    logger.info(f"{prefix}: processed {i + 1}/{n_eval} eval dates")

            prediction_payload = {
                f"{prefix}_dates": torch.tensor(eval_dates, dtype=torch.long),  # (n_eval,)
                f"{prefix}_datetimes": [str(datetimes[t]) for t in eval_dates],
                f"{prefix}_preds": preds,  # (n_eval, individuals, horizon)
                f"{prefix}_preds_context": preds_context,  # (n_eval, individuals, horizon)
                f"{prefix}_E_values": E_values,  # (n_eval, individuals, n_neighbors, horizon)
                f"{prefix}_Ec_values": Ec_values,  # (n_eval, individuals, n_neighbors, horizon)
                f"{prefix}_Y_values": Y_values,  # (n_eval, individuals, horizon)
                f"{prefix}_Yc_values": Yc_values,  # (n_eval, individuals, n_neighbors, horizon)
                f"{prefix}_w_x_xc": w_x_xc_values,  # (n_eval, individuals, n_neighbors)
            }

            if is_context:
                features_payload = {
                    f"{prefix}_mu_x": mu_x_values,  # (n_eval, individuals)
                    f"{prefix}_sigma_x": sigma_x_values,  # (n_eval, individuals)
                    f"{prefix}_mu_xc_mean": mu_xc_values.mean(dim=-1),  # (n_eval, individuals)
                    f"{prefix}_sigma_xc_mean": sigma_xc_values.mean(dim=-1),  # (n_eval, individuals)
                    f"{prefix}_mu_xc_std": mu_xc_values.std(dim=-1, unbiased=False),  # (n_eval, individuals)
                    f"{prefix}_sigma_xc_std": sigma_xc_values.std(dim=-1, unbiased=False),  # (n_eval, individuals)
                    f"{prefix}_same_individual_mean": same_individual_values.mean(dim=-1),
                    f"{prefix}_delta_periods_mean": delta_periods_values.mean(dim=-1),
                    f"{prefix}_w_x_xc_max": w_x_xc_values.max(dim=-1).values,
                    f"{prefix}_w_x_xc_std": w_x_xc_values.std(dim=-1, unbiased=False),
                    f"{prefix}_loss_pred_pred_c": ((preds - preds_context) ** 2).mean(dim=-1),  # (n_eval, individuals)
                    f"{prefix}_loss_pred_yc_mean": ((preds.unsqueeze(2) - Yc_values) ** 2).mean(dim=-1).mean(dim=-1),  # (n_eval, individuals)
                    f"{prefix}_loss_neighbor_residual_mean": (E_values ** 2).mean(dim=-1).mean(dim=-1),  # (n_eval, individuals)
                    f"{prefix}_loss_neighbor_context_residual_mean": (Ec_values ** 2).mean(dim=-1).mean(dim=-1),  # (n_eval, individuals)
                    f"{prefix}_residual_var": weighted_residual_variance(E_values, w_x_xc_values),  # (n_eval, individuals)
                    f"{prefix}_context_residual_var": weighted_residual_variance(Ec_values, w_x_xc_values),  # (n_eval, individuals)
                }
            else:
                features_payload = {
                    f"{prefix}_mu_x": mu_x_values,  # (n_eval, individuals)
                    f"{prefix}_sigma_x": sigma_x_values,  # (n_eval, individuals)
                    f"{prefix}_loss_pred_pred_c": ((preds - preds_context) ** 2).mean(dim=-1),  # (n_eval, individuals)
                }

            logger.info(
                f"Prediction payload size: "
                f"{round(sum(v.numel() * v.element_size() for v in prediction_payload.values() if torch.is_tensor(v)) / 1024**3)} "
                f"(GB)"
            )
            logger.info(
                f"Features payload size: "
                f"{round(sum(v.numel() * v.element_size() for v in features_payload.values() if torch.is_tensor(v)) / 1024**3)} "
                f"(GB)"
            )

            prediction_save_path = save_dir / f"{prefix}_prediction_payload.pt"
            features_save_path = save_dir / f"{prefix}_features_payload.pt"
            torch.save(prediction_payload, prediction_save_path)
            torch.save(features_payload, features_save_path)

            logger.info(f"Saved prediction tensors to: {prediction_save_path}")
            logger.info(f"Saved feature tensors to: {features_save_path}")
            logger.info(f"{prefix}_preds shape: {tuple(preds.shape)}")
            logger.info(f"{prefix}_preds_context shape: {tuple(preds_context.shape)}")
            logger.info(f"{prefix}_E_values shape: {tuple(E_values.shape)}")
            logger.info(f"{prefix}_Ec_values shape: {tuple(Ec_values.shape)}")
            logger.info(f"{prefix}_Y_values shape: {tuple(Y_values.shape)}")
            logger.info(f"{prefix}_Yc_values shape: {tuple(Yc_values.shape)}")
            logger.info(f"{prefix}_w_x_xc shape: {tuple(w_x_xc_values.shape)}")
            logger.info(f"{prefix}_same_individual shape: {tuple(same_individual_values.shape)}")
            logger.info(f"{prefix}_delta_periods shape: {tuple(delta_periods_values.shape)}")
            logger.info(f"{prefix}_mu_x shape: {tuple(mu_x_values.shape)}")
            logger.info(f"{prefix}_sigma_x shape: {tuple(sigma_x_values.shape)}")
            logger.info(f"{prefix}_mu_xc shape: {tuple(mu_xc_values.shape)}")
            logger.info(f"{prefix}_sigma_xc shape: {tuple(sigma_xc_values.shape)}")

        ## Building query dates
        train_eval_dates = get_period_eval_dates(t0_end, t1_end, eval_stride)  # (n_train_eval,)
        eval_eval_dates = get_period_eval_dates(t1_end, t2_end, eval_stride)  # (n_eval_eval,)

        logger.info(
            f"Train/T1 query dates: "
            f"{datetimes[train_eval_dates[0]] if len(train_eval_dates) else 'empty'} "
            f"-> "
            f"{datetimes[train_eval_dates[-1]] if len(train_eval_dates) else 'empty'}"
        )
        logger.info(
            f"Eval/T2 query dates: "
            f"{datetimes[eval_eval_dates[0]] if len(eval_eval_dates) else 'empty'} "
            f"-> "
            f"{datetimes[eval_eval_dates[-1]] if len(eval_eval_dates) else 'empty'}"
        )

        ## Extracting train period
        extract_period(
            eval_dates=train_eval_dates,
            prefix="train",
            store_start=0,
            store_end=t0_end,
        )

        ## Extracting eval period
        extract_period(
            eval_dates=eval_eval_dates,
            prefix="eval",
            store_start=0,
            store_end=t1_end,
        )

        ## Plotting one random target and all its retrieved neighbors
        plot_random_neighbors(
            eval_dates=eval_eval_dates,
            store_start=0,
            store_end=t1_end,
        )

    logger.info("End of script")


if __name__ == "__main__":
    run()

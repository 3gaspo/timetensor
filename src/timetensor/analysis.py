import numpy as np
import torch
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.notebook import tqdm
import scipy.cluster.hierarchy as shc
from scipy.spatial.distance import squareform, pdist, cosine, cdist
from sklearn.manifold import TSNE
import ipywidgets as widgets
from IPython.display import display, clear_output


# --------- utils ---------

def set_seed(seed):
    """Sets RNG seeds when seed is not None."""
    if seed == "None":
        seed = None
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)


def symlog(x, linthresh=1):
    """Signed log transform with linear threshold."""
    return np.sign(x) * np.log1p(np.abs(x / linthresh)) * linthresh


def normalize(x, mean, std, eps=1e-8):
    """Normalizes x using mean/std with epsilon."""
    return (x - mean) / (std + eps)


def filter_df(df, mask):
    """Masks df entries where mask is True."""
    df = df.copy()
    df[mask] = pd.NA
    return df


def filter_dict(dico, keys):
    """Filters a dict by a list of keys."""
    return {key: dico[key] for key in keys}


def cte_mask(df, lookback):
    """Returns mask of constant rolling windows of length lookback."""
    stds = df.rolling(window=lookback).std()
    return stds == 0


def get_normal_stats(x):
    """Returns per-sample mean/std over last dimension.
    x: (B, dim, dates)
    means: (B, dim, 1)
    """
    mean = x.mean(dim=-1, keepdim=True).detach()
    std = x.std(dim=-1, keepdim=True).detach()
    return mean, std


def unroll_windows(dataloader, cap=None, normal=False, mean=None, std=None, seed=None):
    """Unrolls windows from a torch dataloader into tensors."""
    set_seed(seed)

    X, Y, C = [], [], []
    carry_on, total = True, 0
    while carry_on:
        for x, c, y, indiv, date in dataloader:
            total += x.shape[0]
            if normal:
                if mean is None and std is None:
                    mean, std = get_normal_stats(x)
                x, y = normalize(x, mean, std), normalize(y, mean, std)
            X.append(x)
            Y.append(y)
            C.append(c)
            if cap is not None and total + x.shape[0] > cap:
                carry_on = True
                break
        if cap is None or total >= cap:
            carry_on = False

    return torch.concat(X), torch.concat(Y), torch.concat(C)


def get_trend(df, window=1000):
    """Rolling mean trend."""
    return df.rolling(window=window).mean().iloc[window:]


def get_aggr(df, window=100):
    """Block-wise mean aggregation with block size window."""
    n = len(df)
    if n == 0:
        return df.copy()
    block_ids = np.arange(n) // window
    block_means = df.groupby(block_ids).mean()
    aggr_df = df.copy()
    for pos, idx in enumerate(df.index):
        aggr_df.loc[idx] = block_means.loc[block_ids[pos]]
    return aggr_df


def split_six_way(df, time_splits=(0.6, 0.4), indiv_split=1.0, seed=0):
    """Six-way split for dataframe."""
    set_seed(seed)

    if len(time_splits) not in (2, 3):
        raise ValueError("time_splits must have length 2 or 3")

    n = len(df)
    cols = list(df.columns)

    if len(time_splits) == 2:
        a, b = time_splits
        t1 = int(a * n)
        t2 = n
    else:
        a, b, c = time_splits
        t1 = int(a * n)
        t2 = int((a + b) * n)

    k = len(cols)
    k_primary = int(indiv_split * k)
    perm = np.random.permutation(cols)
    primary_cols = list(perm[:k_primary])
    secondary_cols = list(perm[k_primary:])

    df_primary = df[primary_cols] if primary_cols else df.iloc[:, :0]
    df_secondary = df[secondary_cols] if secondary_cols else df.iloc[:, :0]

    if len(time_splits) == 2:
        train = df_primary.iloc[:t1]
        valid1 = df_primary.iloc[:0]
        test1 = df_primary.iloc[t1:]

        valid2 = df_secondary.iloc[:t1]
        valid3 = df_secondary.iloc[:0]
        test2 = df_secondary.iloc[t1:]
    else:
        train = df_primary.iloc[:t1]
        valid1 = df_primary.iloc[t1:t2]
        test1 = df_primary.iloc[t2:]

        valid2 = df_secondary.iloc[:t1]
        valid3 = df_secondary.iloc[t1:t2]
        test2 = df_secondary.iloc[t2:]

    return {
        "train": train,
        "valid1": valid1,
        "test1": test1,
        "valid2": valid2,
        "valid3": valid3,
        "test2": test2,
    }


# --------- stats core ---------

def get_fourier(A, eps=1e-8):
    """Per-column FFT magnitude of standardized series."""
    return np.abs(np.fft.fft((A - np.mean(A, axis=1, keepdims=True)) / (np.std(A, axis=1, keepdims=True) + eps)))

def get_fourier_df(df, eps=1e-8):
    """Per-column FFT magnitude of standardized series."""
    return df.apply(lambda x: np.abs(np.fft.fft((x - x.mean()) / (x.std() + eps))))


def get_gammas(data, lookback, horizon, eps=1e-8):
    """Returns alpha/beta dataframes from rolling lookback/horizon stats."""
    lookback_means = data.rolling(window=lookback).mean().iloc[lookback:]
    lookback_stds = data.rolling(window=lookback).std().iloc[lookback:]
    horizon_means = data.rolling(window=horizon).mean().shift(-horizon).iloc[:-horizon]
    horizon_stds = data.rolling(window=horizon).std().shift(-horizon).iloc[:-horizon]
    alphas = horizon_stds.iloc[lookback:] / (lookback_stds.iloc[:-horizon] + eps)
    betas = (horizon_means.iloc[lookback:] - lookback_means.iloc[:-horizon]) / (lookback_stds.iloc[:-horizon] + eps)
    return alphas, betas


def get_gamma_df(df, lags, horizon, eps=1e-8):
    """Concatenates alpha and beta into a single dataframe."""
    alphas_df, betas_df = get_gammas(df, lags, horizon, eps=eps)
    gamma_df = pd.concat((alphas_df, betas_df))
    return gamma_df


def get_dataset_stats(df_dict, lags, horizon, sampling, save_path=None):
    """Computes dataset-wide mean/std and average alpha/beta for each split."""
    gammas_dict = {k: get_gammas(df_dict[k], lags, horizon) for k in df_dict}
    stats_dict = {}
    for key in df_dict:
        if (key == "train" and sampling["remove_train_cte"]) or (key != "train" and sampling["remove_eval_cte"]):
            mask = cte_mask(df_dict[key], lags)
            clean_df = filter_df(df_dict[key], mask)
            clean_alphas = filter_df(gammas_dict[key][0], mask)
            clean_betas = filter_df(gammas_dict[key][1], mask)
        else:
            clean_df, clean_alphas, clean_betas = df_dict[key], gammas_dict[key][0], gammas_dict[key][1]
        stats_dict[key] = {
            "shape": clean_df.shape,
            "mean": float(np.nanmean(clean_df.values)),
            "stds": float(np.nanmean(np.nanstd(clean_df.values, axis=0))),
            "std": float(np.nanstd(clean_df.values)),
            "alpha": float(np.nanmean(clean_alphas)),
            "beta": float(np.nanmean(clean_betas)),
        }

    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(stats_dict, f, indent=4)

    return stats_dict

def calculate_distances(data, metric='cosine', matrix=False):
    """Pairwise distances between columns."""
    if type(data) == pd.DataFrame:
        values = data.T.values
    else:
        values = data.T
    D = pdist(values, metric=metric)
    if matrix:
        return squareform(D)
    return D

def plot_distances(distances_matrix, show=True, path="", name="distances.pdf"):
    """Plots histogram of distances."""
    plt.figure(figsize=(10, 4))
    plt.hist(distances_matrix[np.triu_indices(distances_matrix.shape[0], k=1)], bins=100)
    plt.title("Distances histogram")
    plt.xlabel("Distances")
    plt.ylabel("Counts")
    if show:
        plt.show()
    else:
        plt.savefig(path + name)
    plt.close()


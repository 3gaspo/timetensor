import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json

from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import cosine

from .utils import filter_df

def get_gammas(data, lookback, horizon, eps=1e-8):
    """compute alpha and beta series. data must be pandas dataframe"""
    lookback_means = data.rolling(window=lookback).mean()[lookback:]
    lookback_stds = data.rolling(window=lookback).std()[lookback:]

    horizon_means = data.rolling(window=horizon).mean().shift(-horizon)[:-horizon]
    horizon_stds = data.rolling(window=horizon).std().shift(-horizon)[:-horizon]

    alphas = horizon_stds[lookback:] / (lookback_stds[:-horizon] + eps)
    betas = (horizon_means[lookback:] - lookback_means[:-horizon]) / (lookback_stds[:-horizon] + eps)

    return alphas, betas

def get_marginals(data, lookback, horizon, eps=1e-8):
    """compute alpha and beta series. data must be pandas dataframe"""
    lookback_means = data.rolling(window=lookback).mean()[lookback:]
    lookback_stds = data.rolling(window=lookback).std()[lookback:]

    horizon_means = data.rolling(window=horizon).mean().shift(-horizon)[:-horizon]
    horizon_stds = data.rolling(window=horizon).std().shift(-horizon)[:-horizon]

    alphas = horizon_stds[lookback:] / (lookback_stds[:-horizon] + eps)
    betas = horizon_means[lookback:] / (lookback_means[:-horizon]  + eps)

    return alphas, betas

def fourier(df):
    """transforms user series into their fft"""
    df = df.apply(lambda x: np.abs(np.fft.fft((x-np.mean(x))/np.std(x))))
    return df

def calculate_distances(df):
  """computes distance matrix of users"""
  num_individuals = df.shape[1]
  distances = np.zeros((num_individuals, num_individuals))
  for i in range(num_individuals):#tqdm(range(num_individuals)):
    for j in range(i + 1, num_individuals):
      distances[i, j] = distances[j, i] = cosine(df[i].values, df[j].values)
  return distances


def find_pairs(distances_matrix):
  """returns closest and furthest users"""
  size = distances_matrix.shape
  na, ma = np.unravel_index(np.argmin(distances_matrix + np.identity(size[0]), axis=None), size)
  nb, mb = np.unravel_index(np.argmax(distances_matrix, axis=None), size)
  return na, nb, ma, mb


def plot_distances(distances_matrix, show=False, path="", name="distances.pdf"):
    """plots distances distribution"""
    plt.hist(distances_matrix[np.triu_indices(distances_matrix.shape[0],k=1)], bins=100)
    if show:
        plt.show()
    else:
        plt.savefig(path + name)
    plt.close()

import scipy.cluster.hierarchy as shc
from scipy.spatial.distance import squareform

def init_clusters(df):
  distances_matrix = calculate_distances(df)
  Z = shc.linkage(squareform(distances_matrix), method='ward')
  return Z, distances_matrix

def plot_dendogram(Z, show=False, path="", name="dendogram.pdf"):
    plt.figure(figsize=(10, 7))
    #Z = shc.linkage(squareform(distances_matrix), method='ward')
    dend = shc.dendrogram(Z)
    if show:
        plt.show()
    else:
        plt.savefig(path + name)
    plt.close()


def get_clusters(Z, n_clusters):
    """returns n_clusters of df"""
    #distances_matrix = calculate_distances(df)
    #Z = shc.linkage(squareform(distances_matrix), method='ward')
    labels = fcluster(Z, n_clusters, criterion='maxclust')
    cluster_indices = [np.where(labels == i)[0] for i in range(1, n_clusters + 1)]
    return labels, cluster_indices

def plot_clusters(df, cluster_indices, n_examples, show=False, path=""):
    """prints n_examples of each cluster"""
    for i, indices in enumerate(cluster_indices):
        print(f"Cluster {i+1}:")
        for j in range(min(n_examples, len(indices))):
            sample_index = indices[j]
            print(f"  Sample index: {sample_index}")
            plt.figure(figsize=(20,3))
            plt.plot(df[sample_index], c=f"C{i}")
            plt.title(f'Sample {sample_index} from cluster {i+1}')
            if show:
                plt.show()
            else:
                plt.savefig(path + f"cluster{i+1}_id{sample_index}.pdf")
            plt.close()

def get_centroids(df, cluster_indices):
  centroids = []
  for indices in cluster_indices:
      cluster_data = df.iloc[:, indices]
      centroid = cluster_data.mean(axis=1)
      centroids.append(centroid)
  return centroids

def get_cluster_dicts(df, cluster_indices):
  clusters = {}
  for i, indices in enumerate(cluster_indices):
      clusters[f'cluster_{i}'] = df.iloc[:, indices]
  return clusters

def get_cluster_distances(df, cluster_indices, centroids):
  """returns intra et inter distances of clusters"""
  intra_distances = {}
  inter_distances = {}

  for i, cluster_1_indices in enumerate(cluster_indices):
    #intra
    if len(cluster_1_indices) > 1:
      cluster_distances = []
      for j in range(len(cluster_1_indices)):
        for k in range(j + 1, len(cluster_1_indices)):
          cluster_distances.append(cosine(df[cluster_1_indices[j]].values, df[cluster_1_indices[k]].values))
      intra_distances[i] = np.mean(cluster_distances)
    else:
      intra_distances[i] = 0

    #inter
    for j in range(i + 1, len(cluster_indices)):
      inter_distances[(i, j)] = cosine(centroids[i], centroids[j])

  return intra_distances, inter_distances

def get_cluster_heterogeneity(df, cluster_indices, centroids):
    intra_distances, inter_distances = get_cluster_distances(df, cluster_indices, centroids)
    intra_distances, inter_distances = list(intra_distances.values()), list(inter_distances.values())
    if len(inter_distances)>0:
        return np.mean(intra_distances) / (np.mean(inter_distances) + 1)
    else:
        return np.mean(intra_distances)

def plot_centroids(centroids, show=False, path="", name="centroids.pdf"):
    plt.figure(figsize=(10, 6))
    for i, centroid in enumerate(centroids):
        plt.plot(centroid, label=f'Cluster {i+1}')
    plt.title('Centroids of clusters')
    plt.xlabel('Time')
    plt.ylabel('Load')
    plt.legend()
    if show:
        plt.show()
    else:
        plt.savefig(path + name)
    plt.close()


def plot_heterogeneity(df, show=False, path="", name="heterogeneity.pdf"):
    heterogeneities = []
    N_clusters = [1, 2, 3, 4, 5, 10, 20, df.shape[1]//10, df.shape[1]//5, df.shape[1]//2, df.shape[1]]
    N_clusters = np.sort(N_clusters)
    Z, distances_matrix = init_clusters(df)
    for n_clusters in tqdm(N_clusters):
        labels, cluster_indices = get_clusters(Z, n_clusters)
        centroids = get_centroids(df, cluster_indices)
        heterogeneity = get_cluster_heterogeneity(df, cluster_indices, centroids)
        heterogeneities.append(heterogeneity)
    plt.plot(N_clusters, heterogeneities)
    plt.xlabel("Number of clusters")
    plt.ylabel("Heterogeneity")
    if show:
        plt.show()
    else:  
        plt.savefig(path + name)
    plt.close()


def identify_cte(df, lookback, save_path=None):
    stds = df.rolling(window=lookback).std()

    #counts
    cte_idxs = np.where(stds==0)
    counts = {}
    for user in enumerate(cte_idxs[1]):
        if counts.get(user) is None:
            counts[user] = 0
        counts[user] += 1

    #plots
    if (save_path is not None) and (len(counts)>0):
        plt.clf()
        fig = plt.figure(figsize=(10,5))
        plt.hist(np.log(list(counts.values())), bins=100)
        plt.yscale("log")
        plt.title("Constant windows per individual")
        plt.xlabel("Individuals")
        plt.ylabel("Constant windows counts")
        plt.savefig(save_path + "constants_hist.pdf")
        plt.close()
    
    mask = stds==0
    return mask, counts

def valid_for_kde(sub, keyx, keyy):
    a = sub[keyy].to_numpy()
    b = sub[keyx].to_numpy()
    na = np.sum(~np.isnan(a))
    nb = np.sum(~np.isnan(b))
    if na < 2 or nb < 2:
        return False
    if np.nanmin(a) == np.nanmax(a) or np.nanmin(b) == np.nanmax(b):
        return False
    return len(sub) >= 3

def plot_gamma(data, path="", name="stats.pdf", per_user=True, lookback=336, horizon=48, samples=1000, title=None, remove_cte=True, log=False, show=False):
    """plots means and stds. data must be pandas dataframe or dict of df"""
    if type(data) != dict:
        data = {"data":data}

    keys, alpha_means_list, beta_means_list = [], [], []
    for key, df in data.items():
        alphas, betas = get_gammas(df, lookback, horizon)
        if per_user:
            clean_alphas, clean_betas = alphas.copy(), betas.copy()
            if remove_cte:
                cte_mask, _ = identify_cte(df.iloc[lookback:-horizon], lookback)
                clean_alphas[cte_mask] = pd.NA
                clean_betas[cte_mask] = pd.NA
            alpha_means = clean_alphas.mean(axis=0)
            beta_means = clean_betas.mean(axis=0)
        else:
            alpha_means = alphas.stack()
            beta_means = betas.stack()
            stds = df.rolling(window=lookback).std()[lookback:].stack()
            if samples < len(alpha_means):
                sampled_idx = np.random.choice(len(alpha_means), size=samples, replace=False)
                alpha_means = alpha_means.iloc[sampled_idx]
                beta_means = beta_means.iloc[sampled_idx]
                stds = stds.iloc[sampled_idx]
            if remove_cte:
                keep_idx = np.where(stds>0)[0]
                alpha_means, beta_means = alpha_means.iloc[keep_idx], beta_means.iloc[keep_idx]

        keys += [key + f" (alpha: {alpha_means.mean():.2f} | beta: {beta_means.mean():.2f})" for _ in range(len(alpha_means))]
        if log:
            alpha_means_list += np.log(np.where(alpha_means>0, alpha_means, 1e-8)).tolist()
            beta_means_list += np.log(np.where(beta_means>0, beta_means, 1e-8)).tolist()
        else:
            alpha_means_list += alpha_means.tolist()
            beta_means_list += beta_means.tolist()

    stats_df = pd.DataFrame({
        'key': keys,
        'beta': beta_means_list,
        'alpha': alpha_means_list})

    g = sns.jointplot(
        data=stats_df,
        x='beta',
        y='alpha',
        hue='key',
        kind='scatter',
        palette='Set1',
        marginal_kws=dict(common_norm=False, fill=True, alpha=0.5)
    )

    # g.plot_joint(sns.kdeplot, hue='key', fill=False, alpha=0.3)
    ax = g.ax_joint
    hue_order = list(dict.fromkeys(stats_df["key"]))  # preserves first-seen order
    palette = sns.color_palette("Set1", n_colors=len(hue_order))
    color_for = dict(zip(hue_order, palette))
    for key, sub in stats_df.groupby("key"):
        if not valid_for_kde(sub, "beta", "alpha"):
            continue
        try:
            sns.kdeplot(
                data=sub,
                x="beta", y="alpha",
                ax=ax,
                color=color_for[key],   # match scatter color
                fill=False, alpha=0.3,
                levels=10,              # strictly increasing
                thresh=1e-6,
                bw_adjust=1.2,
                warn_singular=False,
                common_norm=False,
                legend=False,           # avoid legend duplication
            )
        except ValueError: # If a group still blows up, just skip its KDE
            pass

    if title is None:
        plt.suptitle("Statistics distribution")#, y=1.02)
    else:
        plt.suptitle(title)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(path+name)
    plt.close()

def plot_marginals(data, path="", name="stats.pdf", per_user=True, lookback=336, horizon=48, samples=1000, title=None, remove_cte=True, log=False, show=False):
    """plots means and stds. data must be pandas dataframe or dict of df"""
    if type(data) != dict:
        data = {"data":data}

    keys, alpha_means_list, beta_means_list = [], [], []
    for key, df in data.items():
        alphas, betas = get_marginals(df, lookback, horizon)
        if per_user:
            clean_alphas, clean_betas = alphas.copy(), betas.copy()
            if remove_cte:
                cte_mask, _ = identify_cte(df.iloc[lookback:-horizon], lookback)
                clean_alphas[cte_mask] = pd.NA
                clean_betas[cte_mask] = pd.NA
            alpha_means = clean_alphas.mean(axis=0)
            beta_means = clean_betas.mean(axis=0)
        else:
            alpha_means = alphas.stack()
            beta_means = betas.stack()
            stds = df.rolling(window=lookback).std()[lookback:].stack()
            if samples < len(alpha_means):
                sampled_idx = np.random.choice(len(alpha_means), size=samples, replace=False)
                alpha_means = alpha_means.iloc[sampled_idx]
                beta_means = beta_means.iloc[sampled_idx]
                stds = stds.iloc[sampled_idx]
            if remove_cte:
                keep_idx = np.where(stds>0)[0]
                alpha_means, beta_means = alpha_means.iloc[keep_idx], beta_means.iloc[keep_idx]

        keys += [key + f" (stds ratio: {alpha_means.mean():.2f} | means ratio: {beta_means.mean():.2f})" for _ in range(len(alpha_means))]
        if log:
            alpha_means_list += np.log(np.where(alpha_means>0, alpha_means, 1e-8)).tolist()
            beta_means_list += np.log(np.where(beta_means>0, beta_means, 1e-8)).tolist()
        else:
            alpha_means_list += alpha_means.tolist()
            beta_means_list += beta_means.tolist()

    stats_df = pd.DataFrame({
        'key': keys,
        'beta': beta_means_list,
        'alpha': alpha_means_list})

    g = sns.jointplot(
        data=stats_df,
        x='beta',
        y='alpha',
        hue='key',
        kind='scatter',
        palette='Set1',
        marginal_kws=dict(common_norm=False, fill=True, alpha=0.5)
    )

    # g.plot_joint(sns.kdeplot, hue='key', fill=False, alpha=0.3)
    ax = g.ax_joint
    hue_order = list(dict.fromkeys(stats_df["key"]))  # preserves first-seen order
    palette = sns.color_palette("Set1", n_colors=len(hue_order))
    color_for = dict(zip(hue_order, palette))
    for key, sub in stats_df.groupby("key"):
        if not valid_for_kde(sub, "beta", "alpha"):
            continue
        try:
            sns.kdeplot(
                data=sub,
                x="beta", y="alpha",
                ax=ax,
                color=color_for[key],   # match scatter color
                fill=False, alpha=0.3,
                levels=5,              # strictly increasing
                thresh=1e-6,
                bw_adjust=1.2,
                warn_singular=False,
                common_norm=False,
                legend=False,           # avoid legend duplication
            )
        except ValueError: # If a group still blows up, just skip its KDE
            pass

    if title is None:
        plt.suptitle("Statistics distribution")#, y=1.02)
    else:
        plt.suptitle(title)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(path+name)
    plt.close()

def get_dataset_stats(df_dict, lags, horizon, remove_train_cte=True, remove_eval_cte=True, save_path=None):
    """produces and dictionary of dataset stats (raw and splits)"""
    gammas_dict = {key: get_gammas(df_dict[key], lags, horizon) for key in df_dict}
    stats_dict = {}
    for key in df_dict:
        if (key == "train" and remove_train_cte) or (key != "train" and remove_eval_cte):
            cte_mask, _ = identify_cte(df_dict[key], lags)
            clean_df, clean_alphas, clean_betas = filter_df(df_dict[key], cte_mask), filter_df(gammas_dict[key][0], cte_mask), filter_df(gammas_dict[key][1], cte_mask)
        else:
            clean_df, clean_alphas, clean_betas = df_dict[key], gammas_dict[key][0], gammas_dict[key][1]
        stats_dict[key] = {
            "mean": float(np.nanmean(clean_df.values)),
            "stds": float(np.nanmean(np.nanstd(clean_df.values, axis=0))),
            "std": float(np.nanstd(clean_df.values)),
            "alpha": float(np.nanmean(clean_alphas)),
            "beta": float(np.nanmean(clean_betas))
        }
        # for key_, value in stats_dict[key].items():
        #     if np.isnan(value) or value==0:
        #         print("debug", clean_df.shape)
        #         print("debug", float(np.nanmean(np.nanstd(clean_df.values, axis=1))))
        #         print("debug", float(np.nanmean(np.nanstd(clean_df.values, axis=0))))
        #         print("debug", np.nanstd(clean_df.values))
        #         raise ValueError(f"nan in get stats, {key} {key_} {value}")
            

    if save_path is not None:
        with open(save_path, "w") as file:
            json.dump(stats_dict, file, indent=4)
    
    return stats_dict
    
    

def get_spatial_distance(df, normalize=True, multiplier=1e14):
    """returns spatial distance of dataset"""
    df_ = df.copy()
    if normalize:
        df_ = (df - df.mean()) / df.std()
    means = df_.mean(axis=0)
    stds = df_.std(axis=0)

    points = np.stack((means.values, stds.values), axis=1)

    dist_matrix = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2))
    max_dist = dist_matrix.max() * multiplier

    return max_dist 


def get_temporal_distance(df, t1, t2, normalize=True, multiplier=1e1):
    """returns spatial distance of dataset"""
    df_ = df.copy()
    if normalize:
        df_ = (df - df.mean()) / df.std()

    train_data = df_.iloc[:t1]
    test_data = df_.iloc[t2:]

    train_mean = train_data.values.mean()
    train_std = test_data.values.std()

    test_mean = test_data.values.mean()
    test_std = test_data.values.std()

    dist = np.sqrt((train_mean-test_mean)**2 + (train_std-test_std)**2) * multiplier

    return dist


def get_modulation_distance(df, lags=168, horizon=24, normalize=True, multiplier=1e-7):
    df_ = df.copy()
    if normalize:
      df_ = (df - df.mean()) / df.std()

    alphas_df, betas_df = get_gammas(df_, lags, horizon)

    delta_range = alphas_df.max() - alphas_df.min()
    lambda_range = betas_df.max() - betas_df.min()
    total_range = delta_range + lambda_range
    user_max = total_range.idxmax()
    max_ = total_range[user_max] * multiplier
    return max_

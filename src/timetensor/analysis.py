import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json

from .utils import filter_df, unroll_windows, symlog


##Clustering
import scipy.cluster.hierarchy as shc
from scipy.spatial.distance import squareform
from scipy.spatial.distance import cosine
from scipy.cluster.hierarchy import fcluster
from scripy.cluster.distance import pdist

def fourier(df, eps=1e-8):
    """transforms user series into their fft"""
    df = df.apply(lambda x: np.abs(np.fft.fft((x - x.mean()) / (x.std() + eps))))
    return df

# def calculate_distances(df):
#   """computes distance matrix of users"""
#   num_individuals = df.shape[1]
#   distances = np.zeros((num_individuals, num_individuals))
#   for i in range(num_individuals):#tqdm(range(num_individuals)):
#     for j in range(i + 1, num_individuals):
#       distances[i, j] = distances[j, i] = cosine(df.iloc[:,i].values, df.iloc[:,j].values)
#   return distances

def find_pairs(distances_matrix):
  """returns closest and furthest users"""
  size = distances_matrix.shape
  na, ma = np.unravel_index(np.argmin(distances_matrix + np.identity(size[0]), axis=None), size)
  nb, mb = np.unravel_index(np.argmax(distances_matrix, axis=None), size)
  return na, nb, ma, mb

def plot_distances(distances_matrix, show=False, path="", name="distances.pdf"):
    """plots distances distribution"""
    plt.figure(figsize=(10, 7))
    plt.hist(distances_matrix[np.triu_indices(distances_matrix.shape[0],k=1)], bins=100)
    if show:
        plt.show()
    else:
        plt.savefig(path + name)
    plt.close()

def init_clusters(df):
#   distances_matrix = calculate_distances(df)
#   Z = shc.linkage(squareform(distances_matrix), method='ward')
  Z = shc.linkage(df.T.values, method='ward', metric='euclidean') #propal GPT
  return Z

def plot_dendogram(Z, show=False, path="", name="dendogram.pdf"):
    plt.figure(figsize=(10, 7))
    dend = shc.dendrogram(Z)
    if show:
        plt.show()
    else:
        plt.savefig(path + name)
    plt.close()

def get_clusters(Z, n_clusters):
    """returns n_clusters of df"""
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
            plt.plot(df.iloc[:,sample_index], c=f"C{i}")
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
          cluster_distances.append(cosine(df.iloc[:,cluster_1_indices[j]].values, df.iloc[:,cluster_1_indices[k]].values))
      intra_distances[i] = np.mean(cluster_distances)
    else:
      intra_distances[i] = np.nan

    #inter
    for j in range(i + 1, len(cluster_indices)):
      inter_distances[(i, j)] = cosine(centroids[i].values, centroids[j].values)

  return intra_distances, inter_distances

def get_cluster_heterogeneity(df, cluster_indices, centroids):
    intra_distances, inter_distances = get_cluster_distances(df, cluster_indices, centroids)
    intra_distances, inter_distances = list(intra_distances.values()), list(inter_distances.values())
    if len(inter_distances)>0:
        return np.nanmean(intra_distances) / (np.mean(inter_distances) + 1)
    else:
        return np.nanmean(intra_distances)

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
    Z = init_clusters(df)
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


## Constant windows

def identify_cte(df, lookback, save_path=None):
    """returns mask and counts where windows are constant"""
    stds = df.rolling(window=lookback).std()

    #counts
    cte_idxs = np.where(stds==0)
    counts = {}
    for user in cte_idxs[1]:
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


## Modulations

def get_gammas(data, lookback, horizon, eps=1e-8):
    """compute alpha and beta series. data must be pandas dataframe"""
    lookback_means = data.rolling(window=lookback).mean().iloc[lookback:]
    lookback_stds = data.rolling(window=lookback).std().iloc[lookback:]

    horizon_means = data.rolling(window=horizon).mean().shift(-horizon).iloc[:-horizon]
    horizon_stds = data.rolling(window=horizon).std().shift(-horizon).iloc[:-horizon]

    alphas = horizon_stds.iloc[lookback:] / (lookback_stds.iloc[:-horizon] + eps)
    betas = (horizon_means.iloc[lookback:] - lookback_means.iloc[:-horizon]) / (lookback_stds.iloc[:-horizon] + eps)

    return alphas, betas

def plot_gamma(data, path="", name="stats.pdf", per_user=True, lookback=336, horizon=48, samples=2000, title=None, remove_cte=True, log=False, show=False):
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
            # alpha_means_list += np.log(np.where(alpha_means>0, alpha_means, 1e-8)).tolist()
            # beta_means_list += np.log(np.where(beta_means>0, beta_means, 1e-8)).tolist()
            alpha_means_list += symlog(alpha_means).tolist()
            beta_means_list += symlog(beta_means).tolist()
            xlbl, ylbl = "symlog(delta)", "symlog(gamma)"

        else:
            alpha_means_list += alpha_means.tolist()
            beta_means_list += beta_means.tolist()
            xlbl, ylbl = "delta", "gamma"

    stats_df = pd.DataFrame({
        'key': keys,
        xlbl: beta_means_list,
        ylbl: alpha_means_list})

    plt.figure(figsize=(10, 7))
    g = sns.jointplot(
        data=stats_df,
        x=xlbl,
        y=ylbl,
        hue='key',
        kind='scatter',
        palette='Set1',
        marginal_kws=dict(common_norm=False, fill=True, alpha=0.5)
    )

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

def plot_betas(data, path="", name="stats.pdf", per_user=True, lookback=336, horizon=48, samples=2000, title=None, remove_cte=True, log=False, show=False):
    """plots betas histogram. data must be pandas dataframe or dict of df"""

    if type(data) != dict:
        data = {"data":data}

    keys, beta_means_list = [], []
    for key, df in data.items():
        alphas, betas = get_gammas(df, lookback, horizon)
        if per_user:
            clean_betas = betas.copy()
            if remove_cte:
                cte_mask, _ = identify_cte(df.iloc[lookback:-horizon], lookback)
                clean_betas[cte_mask] = pd.NA
            beta_means = clean_betas.mean(axis=0)
        else:
            beta_means = df.rolling(window=lookback).mean()[lookback:].stack()#.sample(samples)
            stds = df.rolling(window=lookback).std()[lookback:].stack()
            if samples<len(beta_means):
                sampled_idx = np.random.choice(len(beta_means), size=samples, replace=False)
                beta_means = beta_means.iloc[sampled_idx]
                stds = stds.iloc[sampled_idx]
            if remove_cte:
                keep_idx = np.where(stds>0)[0]
                beta_means = beta_means.iloc[keep_idx]

        keys += [key + f" (mean: {beta_means.mean():.2f})" for _ in range(len(beta_means))]
        if log:
            # beta_means_list += np.log(np.where(beta_means>0, beta_means, 1e-8)).tolist()
            beta_means_list += symlog(beta_means).tolist()
            xlbl = "symlog(delta)"
        else:
            beta_means_list += beta_means.tolist()
            xlbl = "delta"

    betas_df = pd.DataFrame({
        'key': keys,
        xlbl: beta_means_list,})

    plt.figure(figsize=(10, 7))
    sns.kdeplot(betas_df, x=xlbl, hue="key", fill=True, common_norm=False)#, log_scale=False), #label=f"{key} (avg:{means.mean():.2f})")

    if title is None:
        plt.title(f"Delta distribution")
    else:
        plt.title(title)
    plt.xlabel("Values")
    plt.ylabel("Density")
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

    if save_path is not None:
        with open(save_path, "w") as file:
            json.dump(stats_dict, file, indent=4)
    
    return stats_dict
    

## Distances

def get_spatial_distance(df, normalize=True, multiplier=1e14, eps=1e-8):
    """returns spatial distance of dataset"""
    df_ = df.copy()
    if normalize:
        df_ = (df - df.mean()) / (df.std() + eps)
    means = df_.mean(axis=0)
    stds = df_.std(axis=0)

    points = np.stack((means.values, stds.values), axis=1)

    dist_matrix = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2))
    max_dist = dist_matrix.max() * multiplier

    return max_dist 


def get_temporal_distance(df, t1, t2, normalize=True, multiplier=1e1, eps=1e-8):
    """returns temporal distance of dataset"""
    df_ = df.copy()
    if normalize:
        df_ = (df - df.mean()) / (df.std() + eps)

    train_data = df_.iloc[:t1]
    test_data = df_.iloc[t2:]
    train_mean = train_data.values.mean()
    train_std = train_data.values.std()
    test_mean = test_data.values.mean()
    test_std = test_data.values.std()

    dist = np.sqrt((train_mean-test_mean)**2 + (train_std-test_std)**2) * multiplier

    return dist


def get_modulation_distance(df, lags=168, horizon=24, normalize=True, multiplier=1e-7, eps=1e-8):
    """returns modulations distance of dataset"""
    df_ = df.copy()
    if normalize:
      df_ = (df - df.mean()) / (df.std() + eps)

    alphas_df, betas_df = get_gammas(df_, lags, horizon)

    delta_range = alphas_df.max() - alphas_df.min()
    lambda_range = betas_df.max() - betas_df.min()
    total_range = delta_range + lambda_range
    user_max = total_range.idxmax()
    max_ = total_range[user_max] * multiplier
    return max_



from sklearn.metrics.pairwise import rbf_kernel
def mmd(X_, Y_, gamma=None, dim=0, multiplier=1e2):
    """Compute the unbiased estimate of MMD^2 between X and Y."""
    X, Y = X_[:, dim, :], Y_[:, dim, :]
    if gamma is None:
        pairwise_dists = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=-1)
        median = np.median(pairwise_dists)
        gamma = 1 / (2 * median**2 + 1e-8)

    Kxx = rbf_kernel(X, X, gamma=gamma)
    Kyy = rbf_kernel(Y, Y, gamma=gamma)
    Kxy = rbf_kernel(X, Y, gamma=gamma)
    
    n, m = len(X), len(Y)
    mmd2 = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1)) \
         + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1)) \
         - 2 * Kxy.mean()
    return multiplier * np.sqrt(max(mmd2, 0))

from scipy.stats import energy_distance
def energy(X, Y, dim=0, multiplier=1e0):
    return multiplier * np.mean([energy_distance(X[:, dim, i], Y[:, dim, i]) for i in range(X.shape[-1])])

from scipy.linalg import sqrtm
def frechet(X_, Y_, eps=1e-6, dim=0, multiplier=1e0):
    """Compute Fréchet (FID-like) distance between two sets of feature vectors."""
    X, Y = X_[:, dim, :], Y_[:, dim, :]
    mu1, mu2 = X.mean(axis=0), Y.mean(axis=0)
    sigma1, sigma2 = np.cov(X, rowvar=False), np.cov(Y, rowvar=False)
    diff = mu1 - mu2

    # Regularize covariances slightly for numerical stability
    sigma1 += eps * np.eye(sigma1.shape[0])
    sigma2 += eps * np.eye(sigma2.shape[0])

    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return multiplier * float(fid)


def get_discrepency(loaders_dict, split_kwargs, samples=3000, mean=None, std=None, alpha=1, beta=0, seed=None, multiplier=1, normal=False):
    """returns dict of distances between disjoint distributions before and after normalization"""
    temp_distances_dict = {}
    spatial_distances_dict = {}

    indiv_split = float(split_kwargs["indiv_split"])
    date_splits = split_kwargs["date_splits"]
    date_splits = [float(split) for split in date_splits.split(";")]

    X1, Y1, C1 = unroll_windows(loaders_dict["train"], cap=samples, shuffle=False, normal=normal, do_context=True, mean=mean, std=std, alpha=alpha, beta=beta,seed=seed)
    X2, Y2, C2 = unroll_windows(loaders_dict["valid2"], cap=int(indiv_split*samples), shuffle=False, normal=normal, do_context=True, mean=mean, std=std, alpha=alpha, beta=beta,seed=seed)
    X3, Y3, C3 = unroll_windows(loaders_dict["test1"], cap=int((date_splits[2]/date_splits[0])*samples), shuffle=False, normal=normal, do_context=True, mean=mean, std=std, alpha=alpha, beta=beta,seed=seed)

    distances = {"Eng": lambda x,y : energy(x,y), "FID": lambda x,y: frechet(x,y)}
    for (key, distance) in distances.items():

        indiv_dist = multiplier * distance(X1, X2)
        temp_dist = multiplier * distance(X1, X3)

        temp_distances_dict["Temporal " + key] = temp_dist
        spatial_distances_dict["Spatial " + key] = indiv_dist

    return temp_distances_dict, spatial_distances_dict

def analyze_discrepency(loaders_dict, split_kwargs, stats_dict, samples=3000, seed=None):
    """returns discrepency distances for different normalization methods"""
    none_temp_distances_dict, none_spatial_distances_dict = get_discrepency(loaders_dict, split_kwargs, samples, seed=seed)
    std_temp_distances_dict, std_spatial_distances_dict = get_discrepency(loaders_dict, split_kwargs, samples, normal=True, mean=stats_dict["train"]["mean"], std=stats_dict["train"]["std"],seed=seed)
    in_temp_distances_dict, in_spatial_distances_dict = get_discrepency(loaders_dict, split_kwargs, samples, normal=True, seed=seed)
    temp_df = pd.DataFrame({"None": none_temp_distances_dict, "Standard":std_temp_distances_dict, "IN":in_temp_distances_dict})
    spatial_df = pd.DataFrame({"None": none_spatial_distances_dict, "Standard":std_spatial_distances_dict, "IN":in_spatial_distances_dict})
    return temp_df, spatial_df
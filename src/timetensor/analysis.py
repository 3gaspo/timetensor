import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scipy.cluster.hierarchy import fcluster


def get_gammas(data, lookback, horizon, eps=1e-6):
    """compute alpha and beta series. data must be pandas dataframe"""
    lookback_means = data.rolling(window=lookback).mean()[lookback:]
    lookback_stds = data.rolling(window=lookback).std()[lookback:]

    horizon_means = data.rolling(window=horizon).mean().shift(-horizon)[:-horizon]
    horizon_stds = data.rolling(window=horizon).std().shift(-horizon)[:-horizon]

    alphas = horizon_stds[lookback:] / (lookback_stds[:-horizon] + eps)
    betas = (horizon_means[lookback:] - lookback_means[:-horizon]) / (lookback_stds[:-horizon] + eps)

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
  return np.mean(list(inter_distances.values())) / (np.mean(list(intra_distances.values())) + 1)

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


def plot_heterogeneity(df, show=False, save_path="", name="heterogeneity.pdf"):
    heterogeneities = []
    N_clusters = [1, 2, 3, 4, 5, 10, 20, df.shape[1]//10, df.shape[1]//5, df.shape[1]//2, df.shape[1]]
    N_clusters = np.sort(N_clusters)
    Z, distances_matrix = init_clusters(df)
    for n_clusters in N_clusters:
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
        plt.savefig(save_path + name)
    plt.close()



def plot_gamma(df, save_path="", save_name="stats.pdf", show=False, per_user=True, lookback=336, horizon=48, samples=1000, title=None):
    """plots means and stds. data must be pandas dataframe or dict of df"""
    if type(df) != dict:
        alphas_df, betas_df = get_gammas(df, lookback, horizon)
        data = {"data":(alphas_df, betas_df)}
    else:
       data = {}
       for key, split_df in df.items():
            alphas_df, betas_df = get_gammas(split_df, lookback, horizon)
            data[key] = (alphas_df, betas_df)

    keys, alpha_means_list, beta_means_list = [], [], []
    for key, (alphas, betas) in data.items():
        if per_user:
            alpha_means = alphas.median(axis=0)
            beta_means = betas.median(axis=0)
        else:
            alpha_means = alphas.stack().sample(samples)
            beta_means = betas.stack().sample(samples)

        keys += [key + f" (alpha: {alpha_means.median():.2f} | beta: {beta_means.median():.2f})" for k in range(len(alpha_means))]
        alpha_means_list += alpha_means.tolist()
        beta_means_list += beta_means.tolist()

    stats_df = pd.DataFrame({
        'key': keys,
        'beta': beta_means_list,
        'alpha': alpha_means_list})

    sns.set_theme(style="white")

    g = sns.jointplot(
        data=stats_df,
        x='beta',
        y='alpha',
        hue='key',
        kind='scatter',
        palette='Set1',
        marginal_kws=dict(common_norm=False, fill=True, alpha=0.5)
    )

    g.plot_joint(sns.kdeplot, hue='key', fill=False, alpha=0.3)

    if title is None:
        plt.suptitle("Statistics distribution", y=1.02)
    else:
        plt.suptitle(title)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(save_path+save_name)
    plt.close()
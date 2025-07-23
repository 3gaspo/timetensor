import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

def get_betas(data, lookback, horizon, eps=1e-6):
    """data must be pandas dataframe"""
    lookback_means = data.rolling(window=lookback).mean()[lookback:]
    lookback_stds = data.rolling(window=lookback).std()[lookback:]

    horizon_means = data.rolling(window=horizon).mean().shift(-horizon)[:-horizon]
    horizon_stds = data.rolling(window=horizon).std().shift(-horizon)[:-horizon]

    alphas = horizon_stds[lookback:] / (lookback_stds[:-horizon] + eps)
    betas = (horizon_means[lookback:] - lookback_means[:-horizon]) / (lookback_stds[:-horizon] + eps)

    return alphas, betas


def calculate_distances(df):
  num_individuals = df.shape[1]
  distances = np.zeros((num_individuals, num_individuals))
  for i in tqdm(range(num_individuals)):
    for j in range(i + 1, num_individuals):
      distances[i, j] = distances[j, i] = cosine(df[i].values, df[j].values)
  return distances

def find_pairs(distances_matrix):
  size = distances_matrix.shape
  na, ma = np.unravel_index(np.argmin(distances_matrix + np.identity(size[0]), axis=None), size)
  nb, mb = np.unravel_index(np.argmax(distances_matrix, axis=None), size)
  return na, nb, ma, mb

def show_distance(df,i,j):
  dist = cosine(df[i].values, df[j].values)
  ni, nj = cosine(df[i].values, df[i].values), cosine(df[j].values, df[j].values)
  fig = plt.figure(figsize=(20,4))
  plt.plot(df[i].values/(ni+1e-6))
  plt.plot(df[j].values/(nj+1e-6))
  plt.title(f"Distance : {dist}")
  plt.show()

import scipy.cluster.hierarchy as shc
from scipy.spatial.distance import squareform

def plot_distances(distances_matrix):
    plt.hist(distances_matrix[np.triu_indices(distances_matrix.shape[0],k=1)], bins=100)
    plt.show()
    plt.close()

def plot_dendogram(distances_matrix):
    plt.figure(figsize=(10, 7))
    Z = shc.linkage(squareform(distances_matrix), method='ward')
    dend = shc.dendrogram(Z)
    plt.show()
    plt.close()


from scipy.cluster.hierarchy import fcluster

def get_clusters(df, n_clusters):
    distances_matrix = calculate_distances(df)
    Z = shc.linkage(squareform(distances_matrix), method='ward')
    n_clusters = 3
    labels = fcluster(Z, n_clusters, criterion='maxclust')
    cluster_indices = [np.where(labels == i)[0] for i in range(1, n_clusters + 1)]
    return cluster_indices, labels

def plot_clusters(data, cluster_indices):
    for i, indices in enumerate(cluster_indices):
        print(f"Cluster {i+1}:")
        for j in range(min(3, len(indices))):
            sample_index = indices[j]
            print(f"  Sample index: {sample_index}")
            plt.figure(figsize=(20,3))
            plt.plot(data[sample_index], c=f"C{i}")
            plt.title(f'Sample {sample_index} from Cluster {i+1}')
            plt.show()

def plot_centroids(data, cluster_indices):
    centroids = []
    for i, indices in enumerate(cluster_indices):
        cluster_data = data.iloc[:, indices]
        centroid = cluster_data.mean(axis=1)
        centroids.append(centroid)

    plt.figure(figsize=(10, 6))
    for i, centroid in enumerate(centroids):
        plt.plot(centroid, label=f'Cluster {i+1}')
    plt.title('Centroids of clusters')
    plt.xlabel('Time')
    plt.ylabel('Load')
    plt.legend()
    plt.show()
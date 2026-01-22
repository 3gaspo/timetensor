import numpy as np
import pandas as pd
import torch
import os

## data generation

def sinusoid(x, period, offset=0, amplitude=1, phase=0):
    return offset + amplitude * np.sin((x-phase) * 2 * np.pi / period)

def linear_trend(x, a, b):
    return b + a*x

def noise(std, size, mean=0):
    return np.random.normal(mean, std, size=size)

def series(T, period, offset, shift, std):
    cyclic = sinusoid(T, period)
    trend = linear_trend(T, shift, offset)
    eps = noise(std, len(T))
    return trend + cyclic + eps


class user:
    def __init__(self, period, std, shift, offset):
        self.period = period
        self.std = std
        self.shift = shift
        self.offset = offset
        self.gen_data = lambda T: series(T, self.period, self.offset, self.shift, self.std)

    def get_data(self, T):
        """returns series for times t in T"""
        return self.gen_data(T)

class cluster:
    def __init__(self, cluster_centroid, cluster_std):
        self.cluster_centroid = cluster_centroid
        self.cluster_std = cluster_std

    def get_user(self):
        """return a random user with cluster parameters"""
        user_dict = {k: v + np.random.normal(0, self.cluster_std[k]) for k, v in self.cluster_centroid.items()}
        return user(**user_dict)

class bipartite_population:
    def __init__(self, cluster1, cluster2, n_cluster1, n_cluster2, r_mix1, r_mix2):
        self.n_cluster1, self.n_cluster2 = n_cluster1, n_cluster2
        self.r_mix1, self.r_mix2 = r_mix1, r_mix2
        self.n_cluster1_out = int(self.r_mix1 * self.n_cluster1)
        self.n_cluster2_out = int(self.r_mix2 * self.n_cluster2)
        self.n_cluster1_in = self.n_cluster1 - self.n_cluster1_out
        self.n_cluster2_in = self.n_cluster2 - self.n_cluster2_out
        self.cluster1 = cluster1
        self.cluster2 = cluster2
        self.build_population()

    def build_population(self):
        users = {"cluster1": [], "cluster2": []}
        for _ in range(self.n_cluster1_in):
            users["cluster1"].append(self.cluster1.get_user())
        for _ in range(self.n_cluster1_out):
            users["cluster1"].append(self.cluster2.get_user())
        for _ in range(self.n_cluster2_in):
            users["cluster2"].append(self.cluster2.get_user())
        for _ in range(self.n_cluster2_out):
            users["cluster2"].append(self.cluster1.get_user())
        self.users = users

    def get_dataset(self, dates):
        """returns dataset of size (n_cluster1 + n_cluster2) x T """
        T = np.linspace(0, dates, dates)
        data = np.array([user.get_data(T) for user in self.users["cluster1"]] + [user.get_data(T) for user in self.users["cluster2"]])
        df = pd.DataFrame(data.T)
        df.columns = [f"user_{k}" for k in range(df.shape[1])]
        return df


def build_dataset(data_path, clusters=None, n1=100, n2=100, r1=0, r2=0, dates=2000, seed=None):
    """builds a synthetic dataset of two clusters.
    r1 : proportion of users from cluster2 in cluster1
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    if clusters is None:
        cluster1 = cluster(
            cluster_centroid = {"period":10, "offset":10, "shift":1e-2, "std":5e-2},
            cluster_std = {"period":0,"offset":1, "shift":0, "std":0})

        cluster2 = cluster(
            cluster_centroid = {"period":10, "offset":100, "shift":-1e-2, "std":5e-2},
            cluster_std = {"period":0,"offset":10, "shift":0, "std":0})
        clusters = [cluster1, cluster2]

    population = bipartite_population(cluster1, cluster2, n1, n2, r1, r2)    
    values_df = population.get_dataset(dates)
    values_df.to_csv(data_path + "synthetic.csv")
    datetimes = list(values_df.index)
    #tensors
    context_pt =  torch.tensor([0 for _ in range(n1)] + [1 for _ in range(n2)]).unsqueeze(dim=1).unsqueeze(dim=1) #.repeat(1, 1, dates)
    values_pt = torch.tensor(values_df.values.T, dtype=torch.float32).unsqueeze(1)
    #save
    torch.save(values_pt, data_path + "values.pt")
    torch.save(context_pt, data_path + "context.pt")
    torch.save(datetimes, data_path+ "datetimes.pt")
    if not os.path.exists(data_path + "clusters/"):
        os.makedirs(data_path + "clusters/")
    torch.save(list(range(0,100)), data_path + "clusters/" + "node0.pt")
    torch.save(list(range(100,200)), data_path + "clusters/" + "node1.pt")

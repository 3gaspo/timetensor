from .dataset import load_data, train_test_split
import numpy as np
import os
import torch

class Client:
    def __init__(self, dataloaders, model=None, id=None, params={}):
        self.dataloaders = dataloaders
        self.model = model
        self.params = params
        self.id = id
        self.available = True
    
    def set_unavailable(self):
        self.available = False
    def set_available(self):
        self.available = True


def client_split(values, context, datetimes, splits, shuffle=True, replace=False, seed=None, context_by_individuals=False, path=""):
    """splits individuals according to splits"""
    
    N = len(splits)
    if type(splits[0]) == str:
        indices_list = []
        for split_path in splits:
            indices = torch.load(split_path, weights_only=False)
            indices_list.append(indices)
    else:
        if seed is not None:
            np.random.seed(seed)
        individuals = values.shape[0]

        remaining = list(range(individuals))
        indices_list = []
        for k in range(N):
            n = int(splits[k]*individuals) #local number of individuals
            if shuffle:
                indices = np.random.choice(remaining, n, replace=False)
                if replace is False:
                    remaining = [k for k in remaining if k not in indices]
            else:
                indices = remaining[:n]
                remaining = remaining[n:]
            indices_list.append(indices)
            torch.save(indices, path + f"node_{k}_indices.pt")

    if context_by_individuals:
        if context is None:
            return {f"node_{i}":(values[indices_list[i], :, :], None, datetimes) for i in range(N)}
        else:
            return {f"node_{i}":(values[indices_list[i], :, :], context[indices_list[i], :, :], datetimes) for i in range(N)}
    else:
        if context is None:
            return {f"node_{i}":(values[indices_list[i], :, :], None, datetimes) for i in range(N)}
        else:
            return {f"node_{i}":(values[indices_list[i], :, :], context, datetimes) for i in range(N)}


#def build_split_datasets(path, splits, shuffle=True, replace=False, seed=None, context_by_individuals=False):
def get_client_splits(path, splits, shuffle=True, replace=False, seed=None, context_by_individuals=False, save=False):
    """splits is a dict with keys the splits for nodes (or path) and for each node the indiv and date splits (or paths)"""
    #data_dict = load_datasets(path)
    values, context, datetimes = load_data(path)
    node_dict =  client_split(values, context, datetimes, list(splits.keys()), shuffle, replace, seed, context_by_individuals, path)

    node_split_dict = {}
    for k, (split, (indiv_split, date_split)) in enumerate(splits.items()):
        values, context, datetimes = node_dict[f"node_{k}"]
        subpath = path + f"node_{k}/"
        if not os.path.exists(subpath):
            os.makedirs(subpath)
        node_split_dict[f"node_{k}"] = train_test_split(values, context, datetimes, indiv_split, date_split, seed, context_by_individuals, subpath)
        
        if save:
            for key, (values, context, datetimes) in node_split_dict[f"node_{k}"].items():
                torch.save(values, subpath + key + "_values.pt")
                if context is not None:
                    torch.save(context, subpath + key + "_context.pt")
                torch.save(datetimes, subpath + key + "_datetimes.pt")
    return node_split_dict
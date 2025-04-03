from .dataset import load_datasets
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


def client_split(values, context, datetimes, splits, shuffle=True, replace=False, seed=None, context_by_individuals=False):
    """splits individuals according to splits"""
    if seed is not None:
        np.random.seed(seed)
    individuals = values.shape[0]
    N = len(splits)

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


def build_split_datasets(path, splits, shuffle=True, replace=False, seed=None, context_by_individuals=False):
    data_dict = load_datasets(path)
    N = len(splits)
    fed_data_dict = {f"node_{i}": {} for i in range(N)}
    for key, key_dict in data_dict.items():
        subset_split = client_split(key_dict["values"], key_dict.get("context"), key_dict["datetimes"], splits, shuffle, replace, seed, context_by_individuals)
        
        for node, (values, context, datetimes) in subset_split.items():
            subpath = path + f"{node}/"
            if not os.path.exists(subpath):
                os.makedirs(subpath)
            torch.save(values, subpath + key + "_values.pt")
            if context is not None:
                torch.save(context, subpath + key + "_context.pt")
            torch.save(datetimes, subpath + key + "_datetimes.pt")
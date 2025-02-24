import numpy as np
import torch
import os

from .dataset import load_example

def get_temporal_features(date):
    """returns list of size context(=3) for a given date"""
    features = []
    hour, weekday, posan = date.hour, date.weekday(), date.timetuple().tm_yday

    features.append(np.cos((2*np.pi / 23) * hour)) # cos(hour)
    features.append(np.sin((2*np.pi / 23) * hour)) # sin
    features.append(np.cos((2*np.pi / 6)* weekday)) # cos(weekday)
    features.append(np.sin((2*np.pi / 6)* weekday)) # sin
    features.append(np.cos((2*np.pi / 365) * posan)) # cos(position in year)
    features.append(np.sin((2*np.pi / 365) * posan)) # sin

    return features


def set_random_data(path="datasets/", prefix="", lag=168, horizon=24, name="rand", context_by_individual=False):
    """gets a random individual and random window from dataset"""
    if prefix is None:
        prefix = ""
    if prefix != "":
        prefix = prefix + "_"
    name = name + "_"
    values = torch.load(path + prefix + "values.pt")
    if os.path.exists(path + prefix + "context.pt"):
        context = torch.load(path + prefix + "context.pt")
    else:
        context = None
    datetimes = torch.load(path + prefix + "datetimes.pt", weights_only=False)

    individuals, dim, dates = values.shape
    rand_indiv = np.random.randint(individuals)
    rand_date = np.random.randint(dates - (lag + horizon))

    inputs = values[rand_indiv, :, rand_date : rand_date+lag]
    target = values[rand_indiv, :, rand_date+lag : rand_date+lag+horizon]
    if context is not None:
        if context_by_individual:
            context = context[rand_indiv, :, rand_date : rand_date+lag+horizon]
        else:
            context = context[:, :, rand_date : rand_date+lag+horizon]

    torch.save(inputs, path + name + "input.pt")
    if context is not None:
        torch.save(inputs, path + name + "context.pt")
    torch.save(target, path + name + "target.pt")
    torch.save((rand_indiv, datetimes[rand_date]), path + name + "indivdate.pt")


def fetch_example_data(path="datasets/", names=None):
    """fetches example data"""
    if type(names) == list:
        dico = {}
        for name in names:
            dico[name] = load_example(path, name)
        return dico
    else:
        if names is None:
            name = "rand"
        else:
            name = names
        return load_example(path, name)


def rename_example_data(path, name, new_dir, old_name="rand"):
    """renames example data to new name (to create new examples)"""
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    os.rename(path + old_name + "_example.pdf", f"{new_dir}{name}_example.pdf")
    os.rename(path + old_name + "_normal_example.pdf", f"{new_dir}{name}_normal_example.pdf")
    os.rename(path + old_name + "_input.pt", f"{new_dir}{name}_input.pt")
    if os.path.exists(path + old_name + "_context.pt"):
        os.rename(path + old_name + "_context.pt", f"{new_dir}{name}_context.pt")
    os.rename(path + old_name + "_target.pt", f"{new_dir}{name}_target.pt")
    os.rename(path + old_name + "_indivdate.pt", f"{new_dir}{name}_indivdate.pt")


def get_stats(values, stat, dim=0):
    """returns tensor of given stats for a loader
    values (Nindiv, Ndim, Ndates)
    """
    if stat == "mean":
        values_stat = values.mean(axis=-1) #(Nindiv, Ndim)
        total_stat = values.mean()
    elif stat == "max":
        values_stat, _ = values.max(axis=-1) #(Nindiv, Ndim)
        total_stat = values.max()
    else:
        raise ValueError("Unrecognized stat name")
    if dim is not None:
        values_stat = values_stat[:, dim]
    return values_stat, total_stat #(Nindiv), (1)



def normalize(X, return_stats=False):
  """
  X: tensor (B, dim, features)
  normalize for each B
  """
  mean = X.mean(dim=-1, keepdim=True)
  std =  X.std(dim=-1, keepdim=True)
  std = torch.where(std != 0, std, 1)
  
  X_normalized = (X - mean) / std

  if return_stats:
    return X_normalized, mean, std
  return X_normalized

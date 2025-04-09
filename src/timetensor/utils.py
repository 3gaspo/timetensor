import numpy as np
import torch
import os
import json

from .dataset import load_example, load_data

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


def set_random_data(path="datasets/", lag=168, horizon=24, name="rand", context_by_individual=False, prefix=""):
    """gets a random individual and random window from dataset"""
    
    values, context, datetimes = load_data(path, prefix)

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
    
    ex_dir = path + "examples/" + name + "/"
    if not os.path.exists(ex_dir):
        os.makedirs(ex_dir)
    torch.save(inputs, ex_dir + "input.pt")
    if context is not None:
        torch.save(inputs, ex_dir + "context.pt")
    torch.save(target, ex_dir + "target.pt")
    torch.save((rand_indiv, datetimes[rand_date]), ex_dir + "indivdate.pt")


def fetch_example_data(path="datasets/examples", names="rand"):
    """fetches example data"""
    if type(names) == list:
        dico = {}
        for name in names:
            dico[name] = load_example(path + name + "/")
        return dico
    else:
        return load_example(path + names + "/")


def unroll_windows(dataloader, cap=None, shuffle=False):
    """unrolls (x,y) examples of dataloaders (typically individuals*dates examples)"""
    X = []
    Y = []
    i = 0
    for x, c, y in dataloader:
        X.append(x)
        Y.append(y)
        if cap is not None and i == cap and not shuffle:
            break
        i+=1
    if shuffle:
        X, Y = np.random.shuffle(X), np.random.shuffle(Y)
    return torch.concat(X), torch.concat(Y)



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
    elif stat == "std":
        values_stat = values.std(axis=-1) #(Nindiv, Ndim)
        total_stat = values.std()
    else:
        raise ValueError("Unrecognized stat name")
    if dim is not None:
        values_stat = values_stat[:, dim]
    return values_stat, total_stat #(Nindiv), (1)


def get_normal_stats(x, std_cst=1):
  """
  X: tensor (B, dim, features)
  normalize for each B
  """
  mean = x.mean(dim=-1, keepdim=True).detach()
  std =  x.std(dim=-1, keepdim=True).detach()
  std = torch.where(std != 0, std, std_cst)
  
  return mean, std


def save_results(value, path, name, model_name, metric_name):
    """adds accuracy result to pandas file"""
    file_path = path + name
    if os.path.exists(file_path):
      with open(file_path, "r") as file:
        dico = json.load(file)
    else:
      dico = {}
    
    model_dico = dico.get(model_name, {})
    model_dico[metric_name] = float(value)
    dico[model_name] = model_dico
    with open(file_path, "w") as file:
        try:
            json.dump(dico, file, indent=4)
        except:
            print(dico)
            
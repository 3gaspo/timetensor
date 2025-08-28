import numpy as np
import torch
import os
import json
import hydra
import pandas as pd


def get_dirs(output_dir, save_name, model_name, normalization=None, criterion_name=None, subsets=None):
    
    get_training = ("revin" in normalization) or (normalization=="mIN") or (model_name not in ["persistence", "repeat", "lookback", "expected"])
    if subsets is not None:
        subset = float(subsets.split(";")[0])
    else:
        subset=None
    if save_name is None:
        save_name = model_name
        if get_training and (normalization is not None):
            if normalization != "None":
                save_name = save_name + "_" + normalization
        if get_training and (criterion_name is not None) and ("sklinear" not in model_name):
            if criterion_name != "MSE":
                save_name = save_name + "_" + criterion_name
        if get_training and subset is not None:
            if subset != 1:
                save_name = save_name + "_" + str(subset)
    save_dir = output_dir + save_name + "/" #current experiment dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    hydra_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir #hydra logs
    with open(save_dir + f'hydra_dir.txt', 'w') as file: 
        file.write(f"{hydra_dir}") #save path of hydra logs to experiment dir
    if not os.path.exists(save_dir + "examples/"): #dir for example predictions
        os.makedirs(save_dir + "examples/")
    if not os.path.exists(save_dir + "plots/"): #dir for example predictions
        os.makedirs(save_dir + "plots/")
    return save_name, save_dir


# def get_temporal_features(date):
#     """returns list of size context(=3) for a given date"""
#     features = []
#     hour, weekday, posan = date.hour, date.weekday(), date.timetuple().tm_yday

#     features.append(np.cos((2*np.pi / 23) * hour)) # cos(hour)
#     features.append(np.sin((2*np.pi / 23) * hour)) # sin
#     features.append(np.cos((2*np.pi / 6)* weekday)) # cos(weekday)
#     features.append(np.sin((2*np.pi / 6)* weekday)) # sin
#     features.append(np.cos((2*np.pi / 365) * posan)) # cos(position in year)
#     features.append(np.sin((2*np.pi / 365) * posan)) # sin

#     return features



def unroll_windows(dataloader, cap=None, shuffle=False, normal=False, alpha=1, beta=0, mIN=False):#, std_cst=1e-6):
    """unrolls (x,y) examples of dataloaders (typically individuals*dates examples)"""
    ###TODO: remove std=0 windows for sklearn fitting
    X = []
    Y = []
    i = 0
    for x, c, y in dataloader:
        if normal:
            mean, std = get_normal_stats(x)
            nx = normalize(x, mean, std)
            ny = normalize(y, mean, std)
            if mIN:
                nx = beta*nx + alpha
                ny = beta*ny + alpha
            X.append(nx)
            Y.append(ny)
        else:
            X.append(x)
            Y.append(y)
        if cap is not None and i == cap and not shuffle:
            break
        i+=1
    if shuffle:
        idx = np.random.permutation(len(X))
        X, Y = [X[i] for i in idx], [Y[i] for i in idx]
        if cap:
            X, Y = X[:cap], Y[:cap]
    return torch.concat(X), torch.concat(Y)


# def get_stats(values, stat, dim=0):
#     """returns tensor of given stats for a loader
#     values (Nindiv, Ndim, Ndates)
#     """
#     if stat == "mean":
#         values_stat = values.mean(axis=-1) #(Nindiv, Ndim)
#         total_stat = values.mean()
#     elif stat == "max":
#         values_stat, _ = values.max(axis=-1) #(Nindiv, Ndim)
#         total_stat = values.max()
#     elif stat == "std":
#         values_stat = values.std(axis=-1) #(Nindiv, Ndim)
#         total_stat = values.std()
#     else:
#         raise ValueError("Unrecognized stat name")
#     if dim is not None:
#         values_stat = values_stat[:, dim]
#     return values_stat, total_stat #(Nindiv), (1)


def get_normal_stats(x):#, std_cst=1e-6):
  """
  X: tensor (B, dim, features)
  normalize for each B
  """
  mean = x.mean(dim=-1, keepdim=True).detach()
  std =  x.std(dim=-1, keepdim=True).detach()
  #std = torch.where(std != 0, std, std_cst)

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


def normalize(x, mean, std, eps=1e-6):
    return (x - mean) / (std + eps)


def average_loss(eval_losses):
    """averages the losses inside dictionnary"""
    mean_losses = {}
    for loss_name, losses in eval_losses.items():
        mean_losses[loss_name] = losses.mean().item()
    return mean_losses
            

def append_in_dict(dico1, dico2):
    for key, value in dico2.items():
        if key not in dico1:
            dico1[key] = []
        if type(value) == list:
            dico1[key] += value
        elif type(value) == torch.tensor and len(value.shape)==0:
            dico1[key] += value.item()
        else:
            dico1[key].append(value)


def filter_dict(dico, keys):
    return {key: dico[key] for key in keys}

def filter_df(df, mask):
    clean_df = df.copy()
    clean_df[mask] = pd.NA
    return clean_df
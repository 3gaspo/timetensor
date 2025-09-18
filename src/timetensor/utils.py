import numpy as np
import torch
import os
import json
import hydra
import pandas as pd


def get_dirs(output_dir, save_name, model_name, norm_name=None, criterion_name=None, subsets=None):
    
    get_training = ((norm_name is not None) and (("revin" in norm_name) or ("mIN" in norm_name))) or (model_name not in ["persistence", "repeat", "lookback", "expected"])
    if subsets is not None:
        subset = float(subsets.split(";")[0])
    else:
        subset=None
    if save_name is None:
        save_name = model_name
        if get_training and (norm_name is not None):
            save_name = save_name + "_" + norm_name
        if get_training and (criterion_name is not None) and (criterion_name != "MSE"):# and ("sklinear" not in model_name):
            save_name = save_name + "_" + criterion_name
        if get_training and (subset is not None) and (subset != 1):
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

def unroll_windows(dataloader, cap=None, shuffle=False, normal=False, alpha=1, beta=0, mIN=False):
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


def get_normal_stats(x): #(B, dim, T)
  mean = x.mean(dim=-1, keepdim=True).detach() #(B, dim, 1)
  std =  x.std(dim=-1, keepdim=True).detach() #(B, dim, 1)
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


def normalize(x, mean, std, eps=1e-8):
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


def is_cte(x, dim=-1):
    """checks if x is constant along dim"""
    return (x.min(dim=dim).values == x.max(dim=dim).values).all()
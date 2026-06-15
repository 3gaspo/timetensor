import numpy as np
import torch
import os
import json
import hydra
import pandas as pd


def get_dirs(output_dir, save_name, model_name, norm_name=None, criterion_name=None, subsets=None, make_dirs=True):
    
    get_training = ((norm_name is not None) and (("revin" in norm_name) or ("mIN" in norm_name))) or (model_name not in ["persistence", "repeat", "lookback", "expected"])
    if subsets is not None:
        subset = float(subsets.sizes["train"])
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
    os.makedirs(save_dir, exist_ok=True)

    hydra_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir #hydra logs
    with open(save_dir + f'hydra_dir.txt', 'w') as file: 
        file.write(f"{hydra_dir}") #save path of hydra logs to experiment dir
    
    if make_dirs:
        os.makedirs(save_dir + "examples/", exist_ok=True)
        os.makedirs(save_dir + "plots/", exist_ok=True)

    return save_name, save_dir

def unroll_windows(dataloader, cap=None, normal=False, alpha=1, beta=0, mean=None, std=None, do_context=False, seed=None):
    """unrolls (x,y) examples of dataloaders (typically individuals*dates examples)"""
    X = []
    Y = []
    C = []
    i = 0
    if seed is not None:
        set_seed(seed)

    carry_on = True
    while carry_on:
        for x, c, y, indiv, date in dataloader:
            i+=x.shape[0]
            if normal:
                if mean is None and std is None:
                    mean, std = get_normal_stats(x)
                nx = normalize(x, mean, std)
                ny = normalize(y, mean, std)
                nx = alpha*nx + beta
                ny = alpha*ny + beta
                X.append(nx)
                Y.append(ny)
            else:
                X.append(x)
                Y.append(y)
            C.append(c)

            if cap is not None and i+x.shape[0] > cap:
                carry_on = True
                break
        if cap is None or i>= cap:
            carry_on = False

    if do_context:
        return torch.concat(X), torch.concat(Y), torch.concat(C)

    else:
        return torch.concat(X), torch.concat(Y)


def symlog(x, linthresh=1):
    return np.sign(x) * np.log1p(np.abs(x / linthresh)) * linthresh

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
            

# def append_in_dict(dico1, dico2):
#     for key, value in dico2.items():
#         if key not in dico1:
#             dico1[key] = []
#         if type(value) == list:
#             dico1[key] += value
#         elif type(value) == torch.tensor and len(value.shape)==0:
#             dico1[key] += value.item()
#         else:
#             dico1[key].append(value)


def filter_dict(dico, keys):
    return {key: dico[key] for key in keys}

def filter_df(df, mask):
    clean_df = df.copy()
    clean_df[mask] = pd.NA
    return clean_df


def is_cte(x, dim=-1):
    """checks if x is constant along dim"""
    return (x.min(dim=dim).values == x.max(dim=dim).values).all()

def text_list(L):
    if type(L) == list:
        return L
    elif type(L) == str:
        return L.split(";")
    else:
        return [L]
    
def set_seed(seed):
    if seed == "None":
        seed =  None
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

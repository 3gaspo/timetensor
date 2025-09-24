import numpy as np
import pandas as pd
import torch
import os

## data generation

def filter(x, c=10, K=100, std=1e-3, p=0.01):
    if x <= K-c:
        return x + c + np.random.normal(0, std)
    elif x == K:
        if np.random.binomial(n=1, p=p, size=1)[0]:
            return int(K/2)
        else:
            return K
    else:
        return K

def series(T, x0=0, c=1, K=100, std=1):
    X = [x0]
    for _ in range(1,T):
        X.append(max(filter(X[-1], c, K, std),0))
    return X


def build_dataset(data_path, dates=200, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
    values = series(dates)
    values_df = pd.DataFrame(np.array(values).T)
    values_pt = torch.tensor(values, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    #save
    values_df.to_csv(data_path + "saturation.csv")
    torch.save(values_pt, data_path + "values.pt")

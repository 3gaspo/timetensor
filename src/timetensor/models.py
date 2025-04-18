import torch
import torch.nn as nn

# from .patchtst import PatchTST
from .dlinear import DLinear
from .utils import get_normal_stats
from sklearn.linear_model import LinearRegression


class RevIN(nn.Module):
    def __init__(self, model, dim, eps=1):
        """
        RevIN: Reversible Instance Normalization for Time Series Forecasting
        """
        super(RevIN, self).__init__()
        self.dim, self.eps = dim, eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1))  #scale
        self.beta = nn.Parameter(torch.zeros(1, dim, 1))  #shift
        self.model = model

    def norm(self, x):
        self.mu, self.std = get_normal_stats(x, std_cst=self.eps)
        x = (x - self.mu) / self.std # (B, dim, lags)
        x = x * self.gamma + self.beta
        return x
    def denorm(self, y):
        y = (y - self.beta) / torch.where(self.gamma != 0, self.gamma, self.eps) #(B, dim, horizon)
        y = y * self.std + self.mu
        return y
    
    def forward(self, x, c=None): #(B, dim, lags)
        x  = self.norm(x) #(B, dim, lags)
        pred = self.model(x, c) #(B, dim, horizon)
        output = self.denorm(pred) #(B, dim, horizon)
        return output


class Normalized(nn.Module):
    def __init__(self, model, mean, std):
        """
        Normalizes input before predictions and denormalizes prediction
        """
        super(Normalized, self).__init__()
        self.model = model
        self.mean, self.std = mean, std

    def norm(self, x):
        x = (x - self.mean) / self.std # (B, dim, lags)
        return x
    def denorm(self, y):
        y = y * self.std + self.mean
        return y
    
    def forward(self, x, c=None): #(B, dim, lags)
        x  = self.norm(x) #(B, dim, lags)
        pred = self.model(x, c) #(B, dim, horizon)
        output = self.denorm(pred) #(B, dim, horizon)
        return output

class InstanceNormalized(nn.Module):
    def __init__(self, model, eps=1):
        """
        Normalizes input before predictions and denormalizes prediction
        """
        super(InstanceNormalized, self).__init__()
        self.eps = eps
        self.model = model

    def norm(self, x):
        self.mean, self.std = get_normal_stats(x, std_cst=self.eps)
        x = (x - self.mean) / self.std # (B, dim, lags)
        return x
    def denorm(self, y):
        y = y * self.std + self.mean
        return y
    
    def forward(self, x, c=None): #(B, dim, lags)
        x  = self.norm(x) #(B, dim, lags)
        pred = self.model(x, c) #(B, dim, horizon)
        output = self.denorm(pred) #(B, dim, horizon)
        return output


    

class persistence(nn.Module):
    """repeats single last value"""
    def __init__(self, horizon):
        super(persistence, self).__init__()
        self.horizon = horizon
    def forward(self, x, context=None):
        past_values = x[:, :, -1].unsqueeze(2) # (B, dim, 1)
        output = past_values.repeat(1, 1, self.horizon) # (B, dim, horizon)
        return output
    
class repeat(nn.Module):
    """returns last horizon values"""
    def __init__(self, horizon):
        super(repeat, self).__init__()
        self.horizon = horizon
    def forward(self, x, context=None):
        output = x[:, :, -self.horizon:] # (B, dim, horizon)
        return output
    
class lookback(nn.Module):
    """repeats past horizon (at idx)"""
    def __init__(self, idx, horizon):
        super(lookback, self).__init__()
        self.idx  = idx
        self.horizon = horizon
    def forward(self, x, context=None):
        output = x[:, :, self.idx:self.idx+self.horizon] # (B, dim, horizon)
        return output

class linear(nn.Module):
    """linear layer on lags"""
    def __init__(self, lags, horizon, dim):
        super(linear, self).__init__()
        self.lags, self.dim, self.horizon  = lags, dim, horizon
        self.fc = nn.Linear(lags * dim, horizon * dim)
    def forward(self, x, context=None):
        batch_size = x.shape[0]
        input = x.view(batch_size, self.lags * self.dim) # (B, lag*dim)
        output = self.fc(input) # (B, horizon*dim)
        output = output.view(batch_size, self.dim, self.horizon) # (B, dim, horizon)
        return output



class sklinear():
    """linear layer on lags"""
    def __init__(self):
        self.reg = LinearRegression()

    def fit(self, Xtrain, ytrain):
        self.reg.fit(Xtrain, ytrain)

    def __call__(self, X):
        return self.reg.predict(X)


def load_model(model_name, shape, normalization=1, **kwargs):
    """loads models from str model name
    normalization:
        0/False
        1: by provided mean and std
        2: by instance
        3: revin
    """
    dim, lags, horizon = shape[0], shape[1], shape[2]
    if model_name == "persistence":
        model = persistence(horizon)
    elif model_name == "repeat":
        model = repeat(horizon)
    elif model_name == "lookback":
        idx = kwargs.get("lookback_idx")
        if idx is None:
            raise ValueError("Please provide lookback_idx for lookback model")
        model = lookback(idx, horizon)
    elif model_name == "linear":
        model = linear(lags, horizon, dim)
    elif model_name == "DLinear":
        model = DLinear(lags, horizon, dim, kwargs.get("kernel_size",25))
    elif model_name == "sklinear":
        model = sklinear()
    elif model_name == "patch_tst":
        model = PatchTST(lags, horizon)
    else:
        raise ValueError(f"Model name not recognized : {model_name}")
    
    if normalization==1:
        mean, std = kwargs.get("mean"), kwargs.get("std")
        if mean is None or std is None:
            mean = 2500
            std = 15000
        return Normalized(model, mean, std)
    elif normalization==2:
        return InstanceNormalized(model, kwargs.get("std_cst", 1))
    elif normalization==3:
        return RevIN(model, dim, kwargs.get("std_cst", 1))
    else:
        return model
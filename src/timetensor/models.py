import torch
import torch.nn as nn
import numpy as np

from .sota.patchtst.patch_tst import PatchTST
from .sota.dlinear import DLinear
from .utils import get_normal_stats
from sklearn.linear_model import LinearRegression


## Models

class ConstantModel(nn.Module):
    """Wrapper for model, repeats value in case of constant window"""
    def __init__(self, model, horizon):
        super().__init__()
        self.model = model
        self.horizon = horizon

    def forward(self, x, c=None): #x : (B, dim, lags)
        std = x.std(dim=-1) # (B, dim)
        non_cte_mask = (std > 0).any(dim=1) # (B)
        last_values = x[:, :, -1].unsqueeze(2) # (B, dim, 1)
        y = last_values.repeat(1, 1, self.horizon) # (B, dim, horizon)
        if non_cte_mask.any():
            x_nc = x[non_cte_mask]
            if c is not None:
                c_nc = c[non_cte_mask]
            else:
                c_nc = None
            y_nc = self.model(x_nc, c=c_nc)
            y[non_cte_mask] = y_nc
        return y
    
    def __getattr__(self, name): # only called if attribute not found normally
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        if hasattr(self.model, name):
            return getattr(self.model, name)
        else:
            raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

class Persistence(nn.Module):
    """Repeats last value"""
    def __init__(self, horizon):
        super().__init__()
        self.name = "persistence"
        self.horizon = horizon
    def forward(self, x, context=None):
        #past_values = x[:, :, -1].unsqueeze(2) # (B, dim, 1)
        past_values = x[:, :, -1:] # (B, dim, 1)
        output = past_values.repeat(1, 1, self.horizon) # (B, dim, horizon)
        return output

class Expected(nn.Module):
    """Repeats lookback mean"""
    def __init__(self, horizon):
        super().__init__()
        self.name = "expected"
        self.horizon = horizon
    def forward(self, x, context=None):
        mean = x.mean(dim=-1, keepdim=True).detach()
        output = mean.repeat(1, 1, self.horizon) # (B, dim, horizon)
        return output

class Repeat(nn.Module):
    """Repeats last segment of horizon size"""
    def __init__(self, horizon):
        super().__init__()
        self.name = "repeat"
        self.horizon = horizon
    def forward(self, x, context=None):
        output = x[:, :, -self.horizon:] # (B, dim, horizon)
        return output
    
class Lookback(nn.Module):
    """Repeats segment of horizon size starting at idx"""
    def __init__(self, horizon, idx):
        super().__init__()
        self.name = "lookback"
        self.horizon = horizon
        self.idx  = idx
    def forward(self, x, context=None):
        output = x[:, :, self.idx:self.idx+self.horizon] # (B, dim, horizon)
        return output

class Linear(nn.Module):
    """Linear layer over lookback"""
    def __init__(self, lags, dim, horizon):
        super().__init__()
        self.name = "linear"
        self.lags, self.dim, self.horizon  = lags, dim, horizon
        self.fc = nn.Linear(lags * dim, horizon * dim)
    def forward(self, x, context=None):
        batch_size = x.shape[0]
        inpt = x.view(batch_size, self.lags * self.dim) # (B, lag*dim)
        output = self.fc(inpt) # (B, horizon*dim)
        output = output.view(batch_size, self.dim, self.horizon) # (B, dim, horizon)
        return output

class Weekly(nn.Module):
    """Linear layer over subset indexes of lookback windows"""
    def __init__(self, lags, dim, horizon):
        super().__init__()
        self.name = "weekly"
        self.dim, self.horizon  = dim, horizon
        indexes=list(range(horizon))
        idx = 7*24
        while idx+horizon<=lags:
            indexes += [idx + k for k in range(horizon)]
            idx += 7*24
        indexes += list(range(lags-1, lags-horizon-1, -1))
        self.indexes = np.unique(indexes)
        self.fc = nn.Linear(len(indexes) * dim, horizon * dim)
    def forward(self, x, context=None): # (B, dim, lag)
        batch_size = x.shape[0]
        subx = x[:, :, self.indexes]
        inpt = subx.view(batch_size, self.indexes * self.dim) # (B, lag*dim)
        output = self.fc(inpt) # (B, horizon*dim)
        output = output.view(batch_size, self.dim, self.horizon) # (B, dim, horizon)
        return output

class Sklinear():
    """Scikit learn closed-form linear regression"""
    def __init__(self, norm_name=False, dim=0, eps=1e-8, **kwargs):
        self.name = "sklinear"
        self.reg = LinearRegression()
        self.norm_name = norm_name
        self.dim = dim
        self.eps = eps
    def norm(self, X, mean, std):
        if self.norm_name == "instance":
            X = (X - mean) / (std+self.eps)
        elif self.norm_name == "relative":
            mean = torch.abs(mean)
            X = X / (mean + self.eps)
        if len(X.shape)==3:
            X = X[:, self.dim, :]
        return X
    def denorm(self, X, mean, std):
        if self.norm_name == "instance":
            X = X * (std+self.eps) + mean
        elif self.norm_name == "relative":
            mean = torch.abs(mean)
            X = X * (mean + self.eps)
        if len(X.shape)==2:
            X = X.unsqueeze(dim=1)
        return X
    def fit(self, Xtrain, ytrain):
        mean, std = get_normal_stats(Xtrain)
        Xtrain, ytrain = self.norm(Xtrain, mean, std), self.norm(ytrain, mean, std)
        self.reg.fit(Xtrain, ytrain)
    def __call__(self, X, context=None):
        mean, std = get_normal_stats(X)
        X = self.norm(X, mean, std)
        pred = torch.tensor(self.reg.predict(X.cpu()))
        pred = pred.unsqueeze(dim=1)
        pred = self.denorm(pred, mean.cpu(), std.cpu())
        return pred



## Normalizations

class DefaultNorm(nn.Module):
    def __init__(self, model, latent=False):
        super().__init__()
        self.model = model
        self.norm_name = "default"
        self.latent = latent
    def norm(self, x):
        return x
    def denorm(self, y):
        return y
    def forward(self, x, c=None): #(B, dim, lags)
        x  = self.norm(x) #(B, dim, lags)
        pred = self.model(x, c) #(B, dim, horizon)
        if self.latent:
            output = pred
        else:
            output = self.denorm(pred) #(B, dim, horizon)
        return output
    
    def __getattr__(self, name): # only called if attribute not found normally
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        if hasattr(self.model, name):
            return getattr(self.model, name)
        else:
            raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")
        
class StandardNorm(DefaultNorm):
    def __init__(self, model, mean, std, eps=1e-8, latent=False, **kwargs):
        """Z-normalizes using global mean and std"""
        super().__init__(model, latent)
        self.norm_name = "standard"
        self.mean, self.std = mean, std
        self.eps = eps
        assert std >= 0
    def norm(self, x):
        x = (x - self.mean) / (self.std+self.eps) # (B, dim, lags)
        return x
    def denorm(self, y):
        y = y * self.std + self.mean
        return y

class MinMax(DefaultNorm):
    def __init__(self, model, min, max, latent=False, **kwargs):
        """Normalizes in range [0,1]"""
        super().__init__(model, latent)
        self.norm_name = "minmax"
        self.min, self.max = min, max
        assert min != max
    def norm(self, x):
        x = (x - self.min) / (self.max - self.min) # (B, dim, lags)
        return x
    def denorm(self, y):
        y = y * (self.max - self.min) + self.min
        return y

class InstanceNorm(DefaultNorm):
    def __init__(self, model, eps=1e-8, latent=False, specific=False, last=False, **kwargs):
        """Z-normalizes using instance lookback mean and std"""
        super().__init__(model, latent)
        self.norm_name = "instance"
        self.eps, self.last, self.specific = eps, last, specific
    def norm(self, x):
        if self.last: #last value
            self.mu = x[:, :, -1].unsqueeze(2).detach()
        else: #mean value
            self.mu = x.mean(dim=-1, keepdim=True).detach() #(B, dim, 1)
        self.std =  x.std(dim=-1, keepdim=True).detach() #(B, dim, 1)
        if self.specific:
            self.scale = torch.where(self.std != 0, self.std, self.eps) #(B, dim, horizon)
        else:
            self.scale = self.std + self.eps
        x = (x - self.mu) / self.scale # (B, dim, lags)
        return x
    def denorm(self, y):
        y = y * self.scale + self.mu
        return y

class RevIN(DefaultNorm):
    def __init__(self, model, dim, eps=1e-8, latent=False, **kwargs):
        """RevIN: Reversible Instance Normalization for Time Series Forecasting"""
        super().__init__(model, latent)
        self.norm_name = "revin"
        self.dim, self.eps = dim, eps
        self.alpha = nn.Parameter(torch.ones(1, dim, 1))  #scale
        self.beta = nn.Parameter(torch.zeros(1, dim, 1))  #shift

    def norm(self, x):
        self.mu, self.std = get_normal_stats(x)
        x = (x - self.mu) / (self.std+self.eps) # (B, dim, lags)
        x = x * self.alpha + self.beta
        return x
    def denorm(self, y):
        y = (y - self.beta) / self.alpha 
        if self.latent:
            return y
        else:
            y = y * (self.std+self.eps) + self.mu
            return y
    def forward(self, x, c=None): #(B, dim, lags)
        x  = self.norm(x) #(B, dim, lags)
        pred = self.model(x, c) #(B, dim, horizon)
        output = self.denorm(pred) #(B, dim, horizon)
        return output
    
class FlexRevIN(DefaultNorm):
    def __init__(self, model, dim, eps=1e-8, start=True, latent=False, **kwargs):
        """Flexible RevIn module"""
        super().__init__(model, latent)
        self.norm_name = "flexrevin"
        self.dim, self.eps = dim, eps

        #flex in
        if start:
            self.nu = nn.Parameter(torch.ones(1, dim, 1))  #scale
            self.eta = nn.Parameter(torch.ones(1, dim, 1))  #shift
        else:
            self.nu = nn.Parameter(torch.zeros(1, dim, 1))  #scale
            self.eta = nn.Parameter(torch.zeros(1, dim, 1))  #shift
        #input modulations
        self.gamma = nn.Parameter(torch.ones(1, dim, 1)) #scale
        self.omega = nn.Parameter(torch.zeros(1, dim, 1)) #shift

        #output modulations
        self.alpha = nn.Parameter(torch.ones(1, dim, 1))  #scale
        self.beta = nn.Parameter(torch.zeros(1, dim, 1))  #shift

    def norm(self, x):
        self.mu, self.std = get_normal_stats(x)
        self.offset = self.nu*self.mu
        self.scale = 1 + self.eta*( 1/(self.std+self.eps) - 1)
        x = (x-self.offset) * self.scale # (B, dim, lags)
        x = x * self.gamma + self.omega
        return x
    def denorm(self, y):
        y = (y - self.omega) / self.gamma 
        y = y / self.scale + self.offset
        y = y * self.alpha + self.beta
        return y
    def forward(self, x, c=None): #(B, dim, lags)
        x  = self.norm(x) #(B, dim, lags)
        pred = self.model(x, c) #(B, dim, horizon)
        output = self.denorm(pred) #(B, dim, horizon)
        return output
    

class mIN(DefaultNorm):
    def __init__(self, model, dim, eps=1e-8, init_alpha=False, init_beta=False, fixed_alpha=False, fixed_beta=False, use_gamma=False, inverse_gamma=False, latent=False,**kwargs):
        """mIN: Modulated Instance Normalization"""
        super().__init__(model, latent)
        self.norm_name = "mIN"
        self.dim, self.eps = dim, eps

        if fixed_alpha:
            if init_alpha:
                self.register_buffer("alpha", torch.full((1, self.dim, 1), init_alpha))
            else:
                self.register_buffer("alpha", torch.ones(1, self.dim, 1))
        else:
            if init_alpha:
                self.alpha = nn.Parameter(torch.full((1, self.dim, 1), init_alpha))
            else:
                self.alpha = nn.Parameter(torch.ones(1, self.dim, 1))  #scale
        if fixed_beta:
            if init_beta:
                self.register_buffer("beta", torch.full((1, self.dim, 1), init_beta))
            else:
                self.register_buffer("beta", torch.zeros((1, self.dim, 1)))
        else:
            if init_beta:
                self.beta = nn.Parameter(torch.full((1, self.dim, 1), init_beta))
            else:
                self.beta = nn.Parameter(torch.zeros(1, self.dim, 1))  #shift

        self.gamma =  nn.Parameter(torch.ones(1, self.dim, 1))
        self.omega = nn.Parameter(torch.zeros(1, self.dim, 1))
        self.use_gamma, self.inverse_gamma = use_gamma, inverse_gamma
        if self.inverse_gamma:
            assert self.use_gamma
    
    def norm(self, x):
        self.mu, self.std = get_normal_stats(x)
        x = (x - self.mu) / (self.std+self.eps)
        if self.use_gamma:
            x = self.gamma * x + self.omega
        return x
    def denorm(self, y):
        if self.inverse_gamma:
            y = (y - self.omega) / self.gamma 
        y = y * self.alpha + self.beta
        if self.latent:
            return y
        y = y * (self.std+self.eps) + self.mu 
        return y
    def forward(self, x, c=None): #(B, dim, lags)
        x  = self.norm(x) #(B, dim, lags)
        pred = self.model(x, c) #(B, dim, horizon)
        output = self.denorm(pred) #(B, dim, horizon)
        return output

class cmIN(mIN):
    def __init__(self, model, dim, n_clusters=2, eps=1e-8, init_alpha=False, init_beta=False, fixed_alpha=False, fixed_beta=False, use_gamma=False, inverse_gamma=False, latent=False, **kwargs):
        """clustered mIN"""
        super().__init__(model, dim, eps, use_gamma=use_gamma, inverse_gamma=inverse_gamma, latent=latent, **kwargs)
        self.norm_name="cmIN"
        assert n_clusters is not None

        if init_alpha:
            self.init_alphas = [float(value) for value in init_alpha]
        else:
            self.init_alphas = [1.0 for _ in range(n_clusters)]
        if init_beta:
            self.init_betas = [float(value) for value in init_beta]
        else:
            self.init_betas = [0.0 for _ in range(n_clusters)]

        self.fixed_alpha = fixed_alpha
        if self.fixed_alpha:
            for k in range(len(self.init_alphas)):
                self.register_buffer(f"alpha_{k}", torch.full((1, self.dim, 1), self.init_alphas[k]))
        else:
            self.alphas = nn.ParameterList([nn.Parameter(torch.full((1, self.dim, 1), self.init_alphas[k])) for k in range(len(self.init_alphas))])
        
        self.fixed_beta = fixed_beta
        if self.fixed_beta:
            for k in range(len(self.init_betas)):
                self.register_buffer(f"beta_{k}", torch.full((1, self.dim, 1), self.init_betas[k]))
        else:
            self.betas = nn.ParameterList([nn.Parameter(torch.full((1, self.dim, 1), self.init_betas[k])) for k in range(len(self.init_betas))])

    def get_alpha_beta(self, cluster):
        if self.fixed_alpha:
            alpha = torch.cat([getattr(self, f"alpha_{int(k)}") for k in cluster])
        else:
            alpha = torch.cat([self.alphas[int(k)] for k in cluster])
        if self.fixed_beta:
            beta  = torch.cat([getattr(self, f"beta_{int(k)}") for k in cluster])
        else:
            beta  = torch.cat([self.betas[int(k)] for k in cluster])
        return alpha, beta

    def denorm(self, y, cluster):
        if self.inverse_gamma:
            y = (y - self.omega) / self.gamma 
        alpha,beta = self.get_alpha_beta(cluster)
        y = y * alpha + beta
        if self.latent:
            return y
        y = y * (self.std+self.eps) + self.mu 
        return y
    def forward(self, x, c=None): #(B, dim, lags)
        assert c is not None
        x  = self.norm(x) #(B, dim, lags)
        pred = self.model(x, c) #(B, dim, horizon)
        output = self.denorm(pred, c[:, 0, 0]) #(B, dim, horizon)
        return output


class cRevIN(RevIN):
    def __init__(self, model, dim, n_clusters=2, eps=1e-8, latent=False, **kwargs):
        """clustered RevIN"""
        super().__init__(model, dim, eps, latent=latent, **kwargs)
        self.norm_name="crevin"
        assert n_clusters is not None
        self.alphas = nn.ParameterList([nn.Parameter(torch.ones((1, self.dim, 1))) for _ in range(n_clusters)])
        self.betas = nn.ParameterList([nn.Parameter(torch.zeros((1, self.dim, 1))) for _ in range(n_clusters)])

    def get_alpha_beta(self, cluster):
        alpha = torch.cat([self.alphas[int(k)] for k in cluster])
        beta  = torch.cat([self.betas[int(k)] for k in cluster])
        return alpha, beta

    def norm(self, x, cluster):
        self.mu, self.std = get_normal_stats(x)
        x = (x - self.mu) / (self.std+self.eps) # (B, dim, lags)
        alpha, beta = self.get_alpha_beta(cluster)
        x = x * alpha + beta
        return x
    def denorm(self, y, cluster):
        alpha, beta = self.get_alpha_beta(cluster)
        y = (y - beta) / alpha 
        if self.latent:
            return y
        else:
            y = y * (self.std+self.eps) + self.mu
            return y
    def forward(self, x, c=None): #(B, dim, lags)
        assert c is not None
        x  = self.norm(x, c[:, 0, 0]) #(B, dim, lags)
        pred = self.model(x, c) #(B, dim, horizon)
        output = self.denorm(pred, c[:, 0, 0]) #(B, dim, horizon)
        return output


class cflexRevIN(DefaultNorm):
    def __init__(self, model, dim, n_clusters=2, latent=False, start=True, **kwargs):
        """clustered RevIN"""
        super().__init__(model, latent, **kwargs)
        self.norm_name="cflexrevin"
        assert n_clusters is not None

        #flex in
        if start:
            self.nus = nn.ParameterList([nn.Parameter(torch.ones(1, dim, 1)) for _ in range(n_clusters)])  #scale
            self.etas = nn.ParameterList([nn.Parameter(torch.ones(1, dim, 1)) for _ in range(n_clusters)])  #shift
        else:
            self.nus = nn.ParameterList([nn.Parameter(torch.zeros(1, dim, 1)) for _ in range(n_clusters)])  #scale
            self.etas = nn.ParameterList([nn.Parameter(torch.zeros(1, dim, 1)) for _ in range(n_clusters)])  #shift
        #input modulations
        self.gammas = nn.ParameterList([nn.Parameter(torch.ones(1, dim, 1)) for _ in range(n_clusters)]) #scale
        self.omegas = nn.ParameterList([nn.Parameter(torch.zeros(1, dim, 1)) for _ in range(n_clusters)]) #shift
        #output modulations
        self.alphas = nn.ParameterList([nn.Parameter(torch.ones(1, dim, 1)) for _ in range(n_clusters)])  #scale
        self.betas = nn.ParameterList([nn.Parameter(torch.zeros(1, dim, 1)) for _ in range(n_clusters)])  #shift

    def get_alpha_beta(self, cluster):
        alpha = torch.cat([self.alphas[int(k)] for k in cluster])
        beta  = torch.cat([self.betas[int(k)] for k in cluster])
        eta = torch.cat([self.etas[int(k)] for k in cluster])
        nu = torch.cat([self.nus[int(k)] for k in cluster])
        gamma = torch.cat([self.gammas[int(k)] for k in cluster])
        omega = torch.cat([self.omegas[int(k)] for k in cluster])
        return alpha, beta, eta, nu, gamma, omega

    def norm(self, x, cluster):
        self.mu, self.std = get_normal_stats(x)
        alpha, beta, eta, nu, gamma, omega = self.get_alpha_beta(cluster)
        self.offset = self.nu*self.mu
        self.scale = 1 + self.eta*( 1/(self.std+self.eps) - 1)
        x = (x-self.offset) * self.scale # (B, dim, lags)
        x = x * self.gamma + self.omega
        return x
    def denorm(self, y, cluster):
        alpha, beta, eta, nu, gamma, omega = self.get_alpha_beta(cluster)
        y = (y - omega) / gamma 
        y = y / self.scale + self.offset
        y = y * self.alpha + self.beta
        return y
    def forward(self, x, c=None): #(B, dim, lags)
        assert c is not None
        x  = self.norm(x, c[:, 0, 0]) #(B, dim, lags)
        pred = self.model(x, c) #(B, dim, horizon)
        output = self.denorm(pred, c[:, 0, 0]) #(B, dim, horizon)
        return output


######

def load_model(model_name, shape, norm_name=None, init_path=None, freeze_core=False, constants=True, **kwargs):
    """loads models from str model name"""
    lags, dim, horizon = shape[0], shape[1], shape[2]
    
    #model
    if model_name == "persistence":
        model = Persistence(horizon)
    elif model_name == "repeat":
        model = Repeat(horizon)
    elif model_name == "lookback":
        model = Lookback(horizon, kwargs.get("lookback_idx",0))
    elif model_name == "expected":
        model = Expected(horizon)
    elif model_name == "linear":
        model = Linear(lags, dim, horizon)
    elif model_name == "weekly":
        model = Weekly(kwargs.get("indexes"), dim, horizon)
    elif model_name == "DLinear":
        model = DLinear(lags, dim, horizon, kwargs.get("kernel_size",25))
        model.name = "DLinear"
    elif model_name == "sklinear":
        model = Sklinear(norm_name, **kwargs)
    elif model_name == "PatchTST":
        model = PatchTST(lags, horizon)
        model.name = "PatchTST"
    else:
        raise ValueError(f"Model name not recognized : {model_name}")
    
    #normalization
    get_training = ((norm_name is not None) and (("mIN" in norm_name) or ("revin" in norm_name))) or (model_name not in ["persistence", "repeat", "lookback", "expected"])
    if get_training and (norm_name is not None) and ("sk" not in model_name): #and norm_name != "None" 
        if norm_name == "standard":
            model = StandardNorm(model, **kwargs)
        elif norm_name == "instance":
            model = InstanceNorm(model, **kwargs)
        elif norm_name == "revin":
            model = RevIN(model, dim, **kwargs)
        elif norm_name == "flexrevin":
            model = FlexRevIN(model, dim, **kwargs)
        elif norm_name == "crevin":
            model = cRevIN(model, dim, **kwargs)
        elif norm_name == "mIN":
            model = mIN(model, dim, **kwargs)
        elif norm_name == "cmIN":
            model = cmIN(model, dim, **kwargs)
        elif norm_name == "cflexrevin":
            model = cflexRevIN(model, dim, **kwargs)
        else:
            ValueError(f"Normalization not recognized : {norm_name}")
    elif ("sk" not in model_name):
        model = DefaultNorm(model)

    #constants
    if constants and ("sk" not in model_name):
        model = ConstantModel(model, horizon)

    #init
    if init_path is not None and ("sk" not in model_name):
        weights = torch.load(init_path)
        model.load_state_dict(weights)
    if freeze_core and ("sk" not in model_name):
        for param in model.parameters():
            if "alpha" not in param and "beta" not in param:
                param.requires_grad = False

    return model
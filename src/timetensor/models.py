import torch
import torch.nn as nn

from .sota.patchtst.patch_tst import PatchTST
from .sota.dlinear import DLinear
from .utils import get_normal_stats
from sklearn.linear_model import LinearRegression

#TODO: un module qui renvoie une constante quand il y a une constante en entrée, plutot que l'output du modele

class DefaultNorm(nn.Module):
    def __init__(self, model, latent=False):
        super().__init__()
        self.model = model
        self.name = self.model.name
        self.normalization = "None"
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

class GlobalNorm(DefaultNorm):
    def __init__(self, model, mean, std, latent=False):
        """Norm/Denorm using fixed mean and std"""
        super().__init__(model, latent=latent)
        self.normalization = "global"
        self.mean, self.std = mean, std

    def norm(self, x):
        x = (x - self.mean) / self.std # (B, dim, lags)
        return x
    def denorm(self, y):
        y = y * self.std + self.mean
        return y


class InstanceNorm(DefaultNorm):
    def __init__(self, model, eps=1e-6, last=False, latent=True, **kwargs):
        """Norm/Denorm using per instance mean and std"""
        super().__init__(model, latent=latent)
        self.normalization = "instance"
        self.eps, self.last = eps, last
    def norm(self, x):
        self.mean, self.std = get_normal_stats(x)#, std_cst=self.eps)
        x = (x - self.mean) / (self.std+self.eps) # (B, dim, lags)
        return x
    def denorm(self, y):
        y = y * (self.std+self.eps) + self.mean
        return y

class RevIN(DefaultNorm):
    def __init__(self, model, dim, eps=1e-6, latent=False, **kwargs):
        """RevIN: Reversible Instance Normalization for Time Series Forecasting"""
        super().__init__(model, latent=latent)
        self.normalization = "revin"
        self.dim, self.eps = dim, eps
        self.alpha = nn.Parameter(torch.ones(1, dim, 1))  #scale
        self.beta = nn.Parameter(torch.zeros(1, dim, 1))  #shift

    def norm(self, x):
        self.mu, self.std = get_normal_stats(x)#, std_cst=self.eps)
        x = (x - self.mu) / (self.std+self.eps) # (B, dim, lags)
        x = x * self.alpha + self.beta
        return x
    def denorm(self, y):
        y = (y - self.beta) / self.alpha #torch.where(self.alpha != 0, self.alpha, self.eps) #(B, dim, horizon)
        if self.latent:
            return y
        y = y * (self.std+self.eps) + self.mu
        return y    
    def forward(self, x, c=None): #(B, dim, lags)
        x  = self.norm(x) #(B, dim, lags)
        pred = self.model(x, c) #(B, dim, horizon)
        output = self.denorm(pred) #(B, dim, horizon)
        return output
    

class mIN(DefaultNorm):
    def __init__(self, model, dim, eps=1e-6, last=False, init_alpha=False, init_beta=False, fixed_alpha=False, fixed_beta=False, use_gamma=False, inverse_gamma=False, mode=0,  latent=False, **kwargs):
        """mIN: Modulated Instance Normalization"""
        super().__init__(model, latent=latent)
        self.normalization = "mIN"
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
        self.mode = mode
        self.last, self.latent = last, latent

        self.model = model

    def set_modules(self, alpha=None, beta=None):
        if alpha is not None:
            self.register_buffer("alpha", torch.full((1, self.dim, 1), alpha))
        if beta is not None:
            self.register_buffer("beta", torch.full((1, self.dim, 1), beta))

    def norm(self, x):
        self.mu, self.std = get_normal_stats(x)#, std_cst=self.eps)
        if self.last:
            self.mu = x[:, :, -1].unsqueeze(2).detach()
        x = (x - self.mu) / (self.std+self.eps)
        if self.use_gamma:
            x = self.gamma * x + self.omega
        return x
    
    def denorm(self, y):
        if self.inverse_gamma:
            y = (y - self.omega) / self.gamma 
        if self.mode == 0:
            y = y * self.alpha + self.beta
            if self.latent:
                return y
            y = (self.std+self.eps)  * y + self.mu 
        elif self.mode == 1:
            y = ((self.std+self.eps) * self.alpha) * y + (self.mu + self.beta) 
        elif self.mode == 2:
            y = (self.std+self.eps) * y + self.mu 
            y = self.alpha * y + self.beta
        return y
    
    def forward(self, x, c=None): #(B, dim, lags)
        x  = self.norm(x) #(B, dim, lags)
        pred = self.model(x, c) #(B, dim, horizon)
        output = self.denorm(pred) #(B, dim, horizon)
        return output


class cmIN(mIN):
    def __init__(self, model, dim, n_clusters=2, eps=1e-6, last=False, init_alpha=False, init_beta=False, fixed_alpha=False, fixed_beta=False, use_gamma=False, inverse_gamma=False, mode=0,  latent=False, **kwargs):
        """mIN: Modulated Instance Normalization"""
        super().__init__(model, dim, eps, last, use_gamma=use_gamma, inverse_gamma=inverse_gamma, mode=mode,  latent=latent, **kwargs)
        self.normalization="cmIN"
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

    def denorm(self, y, cluster):
        if self.inverse_gamma:
            y = (y - self.omega) / self.gamma 
            
        if self.fixed_alpha:
            alpha = torch.cat([getattr(self, f"alpha_{int(k)}") for k in cluster]) 
            beta  = torch.cat([getattr(self, f"beta_{int(k)}") for k in cluster])
        else:
            alpha = torch.cat([self.alphas[int(k)] for k in cluster])
            beta  = torch.cat([self.betas[int(k)] for k in cluster])
        
        if self.mode == 0:
            y = y * alpha + beta
            if self.latent:
                return y
            y = (self.std+self.eps)  * y + self.mu 
        elif self.mode == 1:
            y = ((self.std+self.eps) * alpha) * y + (self.mu + beta) 
        elif self.mode == 2:
            y = (self.std+self.eps) * y + self.mu 
            y = alpha * y + beta
        return y
    
    def forward(self, x, c=None): #(B, dim, lags)
        x  = self.norm(x) #(B, dim, lags)
        pred = self.model(x, c) #(B, dim, horizon)
        output = self.denorm(pred, c[:, 0, 0]) #(B, dim, horizon)
        return output


class persistence(nn.Module):
    """repeats single last value"""
    def __init__(self, horizon):
        super().__init__()
        self.name = "persistence"
        self.normalization = None
        self.horizon = horizon
    def forward(self, x, context=None):
        past_values = x[:, :, -1].unsqueeze(2) # (B, dim, 1)
        output = past_values.repeat(1, 1, self.horizon) # (B, dim, horizon)
        return output

class expected(nn.Module):
    """repeats single last value"""
    def __init__(self, horizon):
        super().__init__()
        self.name = "expected"
        self.normalization = None
        self.horizon = horizon
    def forward(self, x, context=None):
        mean, _ = get_normal_stats(x)
        output = mean.repeat(1, 1, self.horizon) # (B, dim, horizon)
        return output

class repeat(nn.Module):
    """returns last horizon values"""
    def __init__(self, horizon):
        super().__init__()
        self.name = "repeat"
        self.normalization = None
        self.horizon = horizon
    def forward(self, x, context=None):
        output = x[:, :, -self.horizon:] # (B, dim, horizon)
        return output
    
class lookback(nn.Module):
    """repeats past horizon (at idx)"""
    def __init__(self, idx, horizon):
        super().__init__()
        self.name = "lookback"
        self.normalization = None
        self.idx  = idx
        self.horizon = horizon
    def forward(self, x, context=None):
        output = x[:, :, self.idx:self.idx+self.horizon] # (B, dim, horizon)
        return output

class linear(nn.Module):
    """linear layer on lags"""
    def __init__(self, lags, dim, horizon):
        super().__init__()
        self.name = "linear"
        self.normalization = None
        self.lags, self.dim, self.horizon  = lags, dim, horizon
        self.fc = nn.Linear(lags * dim, horizon * dim)
    def forward(self, x, context=None):
        batch_size = x.shape[0]
        inpt = x.view(batch_size, self.lags * self.dim) # (B, lag*dim)
        output = self.fc(inpt) # (B, horizon*dim)
        output = output.view(batch_size, self.dim, self.horizon) # (B, dim, horizon)
        return output



class sklinear():
    """linear layer on lags"""
    def __init__(self, normalization=False, dim=0, eps=1e-6, **kwargs):
        self.name = "sklinear"
        self.reg = LinearRegression()
        self.normalization = normalization
        self.dim = dim
        self.eps = eps
        
    def norm(self, X, mean, std):
        if self.normalization == "instance":
            X = (X - mean) / (std+self.eps)
        elif self.normalization == "relative":
            mean = torch.abs(mean)
            X = X / (mean + self.eps) 
        if len(X.shape)==3:
            X = X[:, self.dim, :]
        return X
    def denorm(self, X, mean, std):
        if self.normalization == "instance":
            X = X * (std+self.eps) + mean
        elif self.normalization == "relative":
            mean = torch.abs(mean)
            X = X * (mean + self.eps)
        if len(X.shape)==2:
            X = X.unsqueeze(dim=1)
        return X
    
    def fit(self, Xtrain, ytrain):
        mean, std = get_normal_stats(Xtrain)#, std_cst=self.eps)
        Xtrain, ytrain = self.norm(Xtrain, mean, std), self.norm(ytrain, mean, std)
        self.reg.fit(Xtrain, ytrain)

    def __call__(self, X, context=None):
        mean, std = get_normal_stats(X)
        X = self.norm(X, mean, std)
        pred = torch.tensor(self.reg.predict(X.cpu()))
        pred = pred.unsqueeze(dim=1)
        pred = self.denorm(pred, mean.cpu(), std.cpu())
        return pred


def load_model(model_name, shape, normalization, init_path=None, freeze_core=False, logger=None, **kwargs):
    """loads models from str model name"""
    lags, dim, horizon = shape[0], shape[1], shape[2]
    if model_name == "persistence":
        model = persistence(horizon)
    elif model_name == "repeat":
        model = repeat(horizon)
    elif model_name == "lookback":
        model = lookback(kwargs.get("lookback_idx",0), horizon)
    elif model_name == "expected":
        model = expected(horizon)
    elif model_name == "linear":
        model = linear(lags, dim, horizon)
    elif model_name == "DLinear":
        model = DLinear(lags, dim, horizon, kwargs.get("kernel_size",25))
        model.name = "DLinear"
    elif model_name == "sklinear":
        model = sklinear(normalization, **kwargs)
    elif model_name == "PatchTST":
        model = PatchTST(lags, horizon)
        model.name = "PatchTST"
    else:
        raise ValueError(f"Model name not recognized : {model_name}")
    
    get_training = normalization in ["mIN", "revin"] or (model_name not in ["persistence", "repeat", "lookback", "expected"])

    if normalization is not None and normalization != "None" and get_training and model_name != "sklinear":
        if normalization == "None":
            model = DefaultNorm(model)
        if normalization == "global":
            mean, std = kwargs.get("mean", 2500), kwargs.get("std", 15000)
            model = GlobalNorm(model, mean, std)
        elif normalization == "instance":
            model = InstanceNorm(model, **kwargs)
        elif normalization == "revin":
            model = RevIN(model, dim,  **kwargs)
        elif normalization == "mIN":
            model = mIN(model, dim, **kwargs)
        elif normalization == "cmIN":
            model = cmIN(model, dim, **kwargs)
        else:
            ValueError(f"Normalization not recognized : {normalization}")
    else:
        model = DefaultNorm(model)

    if init_path is not None:
        weights = torch.load(init_path)
        model.load_state_dict(weights)
        if logger is not None:
            logger.info("Loaded previous state dict")
    if freeze_core:
        for param in model.model.parameters():
            param.requires_grad = False
        trainable_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
        if logger is not None:
            logger.info(f"Froze parameters. Trainable: {trainable_params}")

    return model
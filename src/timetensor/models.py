import torch
import torch.nn as nn
import numpy as np

from .sota.patchtst.patch_tst import PatchTST
from .sota.dlinear import DLinear
from .sota.chronos2.chronos import Chronos
from .utils import get_normal_stats
from sklearn.linear_model import LinearRegression


## Models

class ConstantModel(nn.Module):
    """Wrapper for model, repeats value in case of constant window"""
    def __init__(self, model, horizon):
        super().__init__()
        self.model = model
        self.does_constant = True
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

class ResidualModel(nn.Module):
    """Wrapper for model, sums linear(mu,std) and residual=model(x)"""
    def __init__(self, model, dim, horizon):
        super().__init__()
        self.model = model
        self.does_residual = True

        self.dim, self.horizon = dim, horizon
        self.fc = nn.Linear((2+self.horizon) * self.dim, self.horizon * self.dim)
        # --- custom init ---
        with torch.no_grad():
            # Zero all weights and biases
            self.fc.weight.zero_()
            self.fc.bias.zero_()
            # Identity mapping for latent part
            # For each (dim, horizon), connect latent -> pred directly
            for d in range(dim):
                for h in range(horizon):
                    out_idx = d * horizon + h
                    in_idx = d * (2 + horizon) + 2 + h  # skip mu,std
                    self.fc.weight[out_idx, in_idx] = 1.0

    def forward(self, x, c=None): #x : (B, dim, lags)
        batch_size = x.shape[0]
        mu = x.mean(dim=-1, keepdim=True).detach() #(B, dim, 1)
        std =  x.std(dim=-1, keepdim=True).detach() #(B, dim, 1)

        latent = self.model(x, c) #(B, dim, horizon)
        # stats = torch.cat((mu,std), dim=-1) # (B, dim, 2)
        # features = torch.cat((stats,latent), dim=-1) # (B, dim, 2+horizon)
        features = torch.cat((mu, std, latent), dim=-1)   # (B, dim, 2+horizon)
        features = features.reshape(batch_size, self.dim * (2 + self.horizon)) 
        pred = self.fc(features) # (B, dim * horizon)
        pred = pred.view(batch_size, self.dim, self.horizon) #(B, dim, horizon)

        return pred
     
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

# class Weekly(nn.Module):
#     """Linear layer over subset indexes of lookback windows"""
#     def __init__(self, lags, dim, horizon):
#         super().__init__()
#         self.name = "weekly"
#         self.dim, self.horizon  = dim, horizon
#         indexes=list(range(horizon))
#         idx = 7*24
#         while idx+horizon<=lags:
#             indexes += [idx + k for k in range(horizon)]
#             idx += 7*24
#         indexes += list(range(lags-1, lags-horizon-1, -1))
#         self.indexes = np.unique(indexes)
#         self.fc = nn.Linear(len(indexes) * dim, horizon * dim)
#     def forward(self, x, context=None): # (B, dim, lag)
#         batch_size = x.shape[0]
#         subx = x[:, :, self.indexes]
#         inpt = subx.view(batch_size, self.indexes * self.dim) # (B, lag*dim)
#         output = self.fc(inpt) # (B, horizon*dim)
#         output = output.view(batch_size, self.dim, self.horizon) # (B, dim, horizon)
#         return output

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
        assert torch.all(self.std >= 0)
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
        assert torch.all((self.max - self.min) >= 0)
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
    
class SoftmIN(DefaultNorm):
    def __init__(self, model, dim, eps=1e-8, start=True, **kwargs):
        """Flexible RevIn module"""
        super().__init__(model)
        self.norm_name = "softmin"
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

        #activations
        self.sig1 = nn.Sigmoid()
        self.sig2 = nn.Sigmoid()

    def norm(self, x):
        self.mu, self.std = get_normal_stats(x)
        self.offset = self.sig1(self.nu)*self.mu
        self.scale = 1 + self.sig2(self.eta)*( 1/(self.std+self.eps) - 1)
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
    def __init__(self, model, dim, eps=1e-8, init_alpha=True, init_beta=True, fixed_alpha=False, fixed_beta=False, use_gamma=True, inverse_gamma=True, latent=False,**kwargs):
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

class cmIN(DefaultNorm):
    def __init__(self, model, dim, n_clusters=None, eps=1e-8, init_alpha=False, init_beta=False, fixed_alpha=False, fixed_beta=False, use_gamma=False, inverse_gamma=False, latent=False, **kwargs):
        """clustered mIN"""
        # super().__init__(model, dim, eps, use_gamma=use_gamma, inverse_gamma=inverse_gamma, latent=latent, **kwargs)
        super().__init__(model, latent)
        self.norm_name="cmIN"
        self.dim, self.eps = dim, eps
        if n_clusters is None:
            self.n_clusters = len(init_alpha)
        else:
            self.n_clusters = n_clusters

        if init_alpha is not False and init_alpha is not None:
            self.init_alphas = [float(value) for value in init_alpha]
        else:
            self.init_alphas = [1.0 for _ in range(n_clusters)]
        if init_beta is not False and init_beta is not None:
            self.init_betas = [float(value) for value in init_beta]
        else:
            self.init_betas = [0.0 for _ in range(n_clusters)]

        self.fixed_alpha = fixed_alpha
        self.register_buffer("alpha_out", torch.ones(1, self.dim, 1))
        if self.fixed_alpha:
            for k in range(len(self.init_alphas)):
                self.register_buffer(f"alpha_{k}", torch.full((1, self.dim, 1), self.init_alphas[k]))
        else:
            self.alphas = nn.ParameterList([nn.Parameter(torch.full((1, self.dim, 1), self.init_alphas[k])) for k in range(len(self.init_alphas))])
        
        self.fixed_beta = fixed_beta
        self.register_buffer("beta_out", torch.zeros(1, self.dim, 1))
        if self.fixed_beta:
            for k in range(len(self.init_betas)):
                self.register_buffer(f"beta_{k}", torch.full((1, self.dim, 1), self.init_betas[k]))
        else:
            self.betas = nn.ParameterList([nn.Parameter(torch.full((1, self.dim, 1), self.init_betas[k])) for k in range(len(self.init_betas))])

        self.register_buffer("gamma_out", torch.ones(1, self.dim, 1))
        self.gammas = nn.ParameterList([nn.Parameter(torch.ones(1, self.dim, 1)) for _ in range(n_clusters)])
        # self.gamma =  nn.Parameter(torch.ones(1, self.dim, 1))
        self.register_buffer("omega_out", torch.zeros(1, self.dim, 1))
        self.omegas = nn.ParameterList([nn.Parameter(torch.zeros(1, self.dim, 1)) for _ in range(n_clusters)])
        # self.omega = nn.Parameter(torch.zeros(1, self.dim, 1))
        self.use_gamma, self.inverse_gamma = use_gamma, inverse_gamma
        if self.inverse_gamma:
            assert self.use_gamma

    def get_modulations(self, cluster):
        alpha, beta, gamma, omega = [], [], [], []
        for k in cluster:
            idx = int(k.item())
            if k >= self.n_clusters:
                alpha.append(getattr(self, f"alpha_out"))
                beta.append(getattr(self, f"beta_out"))
                gamma.append(getattr(self, f"gamma_out"))
                omega.append(getattr(self, f"omega_out"))
            else:
                if self.fixed_alpha:
                    alpha.append(getattr(self, f"alpha_{idx}"))
                else:
                    alpha.append(self.alphas[idx])
                if self.fixed_beta:
                    beta.append(getattr(self, f"beta_{idx}"))
                else:
                    beta.append(self.betas[idx])
                gamma.append(self.gammas[idx])
                omega.append(self.omegas[idx])
        alpha, beta, gamma, omega = torch.cat(alpha), torch.cat(beta), torch.cat(gamma), torch.cat(omega)
        return alpha, beta, gamma, omega

    def norm(self, x, cluster):
        self.mu, self.std = get_normal_stats(x)
        alpha, beta, gamma, omega = self.get_modulations(cluster)
        x = (x - self.mu) / (self.std+self.eps)
        if self.use_gamma:
            # x = self.gamma * x + self.omega
            x = gamma * x + omega
        return x
    def denorm(self, y, cluster):
        alpha, beta, gamma, omega = self.get_modulations(cluster)
        if self.inverse_gamma:
            # y = (y - self.omega) / self.gamma 
            y = (y - omega) / gamma 
        y = y * alpha + beta
        if self.latent:
            return y
        y = y * (self.std+self.eps) + self.mu 
        return y
    def forward(self, x, c=None): #(B, dim, lags)
        assert c is not None
        x  = self.norm(x, c[:, 0, 0]) #(B, dim, lags)
        pred = self.model(x, c) #(B, dim, horizon)
        output = self.denorm(pred, c[:, 0, 0]) #(B, dim, horizon)
        return output

######

def load_model(model_name, shape, norm_name=None, init_path=None, freeze_core=False, constants=True, residuals=False, stats_dict=None, nodes_stats_dict=None, cpu=False, logger=None, **kwargs):
    """loads models from str model name"""
    lags, dim, horizon = shape[0], shape[1], shape[2]
    
    if (norm_name is not None) and ("mIN" in norm_name):
        format_min_kwargs(kwargs, norm_name, nodes_stats_dict, stats_dict, logger)

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
    # elif model_name == "weekly":
    #     model = Weekly(kwargs.get("indexes"), dim, horizon)
    elif model_name == "DLinear":
        model = DLinear(lags, dim, horizon, kwargs.get("kernel_size",25))
        model.name = "DLinear"
    elif model_name == "sklinear":
        model = Sklinear(norm_name, **kwargs)
    elif model_name == "PatchTST":
        model = PatchTST(lags, horizon)
        model.name = "PatchTST"
    elif model_name == "chronos":
        model = Chronos(horizon)
        model.name = "chronos"
    else:
        raise ValueError(f"Model name not recognized : {model_name}")
    logger.info(f"Loaded {model.name}")

    #normalization
    get_training = ((norm_name is not None) and (("mIN" in norm_name) or ("revin" in norm_name))) or (model_name not in ["persistence", "repeat", "lookback", "expected"])
    if get_training and (norm_name is not None) and ("sk" not in model_name):
        if norm_name == "standard":
            model = StandardNorm(model, **kwargs)
        elif norm_name == "instance":
            model = InstanceNorm(model, **kwargs)
        elif norm_name == "revin":
            model = RevIN(model, dim, **kwargs)
        elif norm_name == "softmin":
            model = SoftmIN(model, dim, **kwargs)
        elif norm_name == "mIN":
            model = mIN(model, dim, **kwargs)
        elif norm_name == "cmIN":
            model = cmIN(model, dim, **kwargs)
        else:
            raise ValueError(f"Normalization not recognized : {norm_name}")
    elif ("sk" not in model_name):
        model = DefaultNorm(model)

    #constants
    if constants and ("sk" not in model_name):
        logger.info(f"Added constant wrapper to model")
        model = ConstantModel(model, horizon)
    else:
        model.does_constant = False
    #residuals
    if residuals and ("sk" not in model_name):
        logger.info(f"Added residuals to model")
        model = ResidualModel(model, dim, horizon)
    else:
        model.does_residual = False
        
    #init
    if init_path is not None and ("sk" not in model_name):
        logger.info(f"Loaded model at {init_path}")
        if cpu:
            weights = torch.load(init_path, map_location=torch.device('cpu'))
        else:
            weights = torch.load(init_path)
        model.load_state_dict(weights)
    if freeze_core and ("sk" not in model_name):
        logger.info(f"Froze core model")
        for name, param in model.named_parameters():
            if "alpha" not in name and "beta" not in name:
                param.requires_grad = False

    return model


def format_min_kwargs(kwargs, norm_name, nodes_stats_dict, stats_dict, logger):
    """utils methods to format model kwargs"""
    if (norm_name is not None and "cmIN" in norm_name) and kwargs.get("n_clusters") is None and not (kwargs.get("init_alpha") or kwargs.get("init_beta")):
        kwargs["n_clusters"] = len(nodes_stats_dict)

    if kwargs.get("init_alpha") is True:
        if "cmIN" in norm_name:
            kwargs["init_alpha"] = [nodes_stats_dict[node]["train"]["alpha"] for node in nodes_stats_dict]
            if len(kwargs["init_alpha"])<10:
                logger.info(f"Loaded init_alphas: {kwargs['init_alpha']}")
        else:
            kwargs["init_alpha"] = stats_dict["train"]["alpha"]
            logger.info(f"Loaded init_alphas: {kwargs['init_alpha']}")
    if kwargs.get("init_beta") is True:
        if "cmIN" in norm_name:
            kwargs["init_beta"] = [nodes_stats_dict[node]["train"]["beta"] for node in nodes_stats_dict]
            if len(kwargs["init_beta"])<10:
                logger.info(f"Loaded init_betas: {kwargs['init_beta']}")
        else:
            kwargs["init_beta"] = stats_dict["train"]["beta"]
            logger.info(f"Loaded init_betas: {kwargs['init_beta']}")

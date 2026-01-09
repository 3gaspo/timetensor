import torch
import torch.nn as nn
import numpy as np

from .utils import get_normal_stats

from sklearn.linear_model import LinearRegression
from .sota.patchtst.patch_tst import PatchTST
from .sota.dlinear import DLinear
from .sota.chronos2.chronos import Chronos

## Baselines

class Persistence(nn.Module):
    """Repeats last value"""
    def __init__(self, horizon):
        super().__init__()
        self.model_name, self.model_type = "persistence", "pytorch"
        self.horizon = horizon
    def forward(self, x):
        past_values = x[:, :, -1:] # (B, dim, 1)
        output = past_values.repeat(1, 1, self.horizon) # (B, dim, horizon)
        return output

class Expected(nn.Module):
    """Repeats lookback mean"""
    def __init__(self, horizon):
        super().__init__()
        self.model_name, self.model_type = "expected", "pytorch"
        self.horizon = horizon
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True).detach()
        output = mean.repeat(1, 1, self.horizon) # (B, dim, horizon)
        return output

class Repeat(nn.Module):
    """Repeats last segment of horizon size"""
    def __init__(self, horizon):
        super().__init__()
        self.model_name, self.model_type = "repeat", "pytorch"
        self.horizon = horizon
    def forward(self, x):
        output = x[:, :, -self.horizon:] # (B, dim, horizon)
        return output
    
class Lookback(nn.Module):
    """Repeats segment of horizon size starting at idx"""
    def __init__(self, horizon, idx):
        super().__init__()
        self.model_name, self.model_type = "lookback", "pytorch"
        self.horizon = horizon
        self.idx  = idx
    def forward(self, x):
        output = x[:, :, self.idx:self.idx+self.horizon] # (B, dim, horizon)
        return output

class Linear(nn.Module):
    """Linear layer over lookback"""
    def __init__(self, lags, dim, horizon):
        super().__init__()
        self.model_name, self.model_type = "linear", "pytorch"
        self.lags, self.dim, self.horizon  = lags, dim, horizon
        self.fc = nn.Linear(lags * dim, horizon * dim)
    def forward(self, x):
        batch_size = x.shape[0]
        inpt = x.view(batch_size, self.lags * self.dim) # (B, lag*dim)
        output = self.fc(inpt) # (B, horizon*dim)
        output = output.view(batch_size, self.dim, self.horizon) # (B, dim, horizon)
        return output

class LinearPeriod(nn.Module):
    """Linear layer over specific period"""
    def __init__(self, lags, dim, horizon, period):
        super().__init__()
        self.model_name, self.model_type = "period", "pytorch"
        self.lags, self.dim, self.horizon  = lags, dim, horizon
        self.period = period

        self.idx = [t for t in range(lags) if (t % period) in range(self.horizon)]
        self.fc = nn.Linear(len(idx) * dim, horizon * dim)

    def forward(self, x): #(B, dim, lag)
        batch_size = x.shape[0]
        subx = x[:, :, self.idx]
        inpt = subx.view(batch_size, len(self.idx) * self.dim) # (B, idxs*dim)
        output = self.fc(inpt) # (B, horizon*dim)
        output = output.view(batch_size, self.dim, self.horizon) # (B, dim, horizon)
        return output

class Sklinear():
    """Scikit learn closed-form linear regression"""
    def __init__(self, norm_name=False, dim=0, eps=1e-8, **kwargs):
        self.model_name, self.model_type = "sklinear", "scikit-learn"
        self.reg = LinearRegression()
        self.norm_name = norm_name
        self.dim = dim
        self.eps = eps

    def norm(self, X, mean, std):
        if self.norm_name == "instance":
            X = (X - mean) / (std + self.eps)
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
    def __call__(self, X):
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
        self.norm_name = "min_max"
        self.min, self.max = min, max
        assert torch.all((self.max - self.min) > 0)
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
    

## Models wrappers

class ContextModel(nn.Module):
    """Wrapper for model to enable optional context in forward"""
    def __init__(self, model, do_context=False):
        super().__init__()
        self.model = model
        self.do_context = do_context

    def forward(self, x, c=None): #x : (B, dim, lags)
        if self.do_context and c is not None:
            return self.model(x, c)
        else:
            return self.model(x)
        return
    
    def __getattr__(self, name): # only called if attribute not found normally
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        if hasattr(self.model, name):
            return getattr(self.model, name)
        else:
            raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

class ConstantModel(nn.Module):
    """Wrapper for model, repeats value in case of constant window"""
    def __init__(self, model, horizon, do_constant=False):
        super().__init__()
        self.model = model
        self.do_constant = do_constant
        self.horizon = horizon

    def forward(self, x, c=None): #x : (B, dim, lags)
        if self.do_constant:
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
        else:
            y = self.model(x, c)
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


## Loading model

def model_selector(model_name, lags, dim, horizon, **kwargs):
    do_context = False
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
    elif model_name == "period":
         model = LinearPeriod(lags, dim, horizon, kwargs.get("period", horizon))
    elif model_name == "DLinear":
        model = DLinear(lags, dim, horizon, kwargs.get("kernel_size",25))
        model.model_name = "DLinear"
        model.model_type = "pytorch"
    elif model_name == "sklinear":
        model = Sklinear(**kwargs)
    elif model_name == "PatchTST":
        model = PatchTST(lags, horizon)
        model.model_name = "PatchTST"
        model.model_type = "pytorch"
    elif model_name == "chronos":
        model = Chronos(horizon)
        model.model_name = "chronos"
        model.model_type = "pytorch"
        do_context = True
    else:
        raise ValueError(f"Model name not recognized : {model_name}")
    return model, do_context

def normalization_selector(model, norm_name, dim, **kwargs):
    if norm_name is None or norm_name == "None":
        model = DefaultNorm(model)
    elif norm_name == "standard":
        model = StandardNorm(model, **kwargs)
    elif norm_name == "instance":
        model = InstanceNorm(model, **kwargs)
    elif norm_name == "IN":
        model = GRevIN.build_in(model, dim)
        model.norm_name = "instance"
    elif norm_name == "revin":
        model = RevIN(model, dim, **kwargs)
    elif norm_name == "RevIN":
        model = GRevIN.build_revin(model, dim)
        model.norm_name = "revin"
    elif norm_name == "cmIN":
        model = GRevIN.build_cmin(model, dim)
        model.norm_name = "cmIN"
    elif norm_name == "PRevIN":
        model = GRevIN.build_ersonalized_revin(model, dim)
        model.norm_name = "revin"
    elif norm_name == "GRevIN":
        model = GRevIN(model, dim, **kwargs)
    else:
        raise ValueError(f"Normalization not recognized : {norm_name}")
    return model

def load_model(model_name, shape, norm_name=None, init_path=None, do_constants=True, cpu=False, **kwargs):
    """loads models from str model name"""
    lags, dim, horizon = shape[0], shape[1], shape[2]
    
    model, do_context = model_selector(model_name, lags, dim, horizon, **kwargs) #model architecture
    if model.model_type == "pytorch": 
        model = ContextModel(model, do_context) #allow context in call input
        model = normalization_selector(model, norm_name, dim, **kwargs)
        model = ConstantModel(model, horizon, kwargs.get(do_constants))

    #init
    if init_path is not None and model.model_type == "pytorch":
        if cpu:
            weights = torch.load(init_path, map_location=torch.device('cpu'))
        else:
            weights = torch.load(init_path)
        model.load_state_dict(weights)

    return model


# def format_min_kwargs(kwargs, norm_name, nodes_stats_dict, stats_dict, logger):
#     """utils methods to format model kwargs"""
#     if (norm_name is not None and "cmIN" in norm_name) and kwargs.get("n_clusters") is None and not (kwargs.get("init_alpha") or kwargs.get("init_beta")):
#         kwargs["n_clusters"] = len(nodes_stats_dict)

#     if kwargs.get("init_alpha") is True:
#         if "cmIN" in norm_name:
#             kwargs["init_alpha"] = [nodes_stats_dict[node]["train"]["alpha"] for node in nodes_stats_dict]
#             if len(kwargs["init_alpha"])<10:
#                 logger.info(f"Loaded init_alphas: {kwargs['init_alpha']}")
#         else:
#             if stats_dict is not None:
#                 kwargs["init_alpha"] = stats_dict["train"]["alpha"]
#             else:
#                 kwargs["init_alpha"] = None
#             logger.info(f"Loaded init_alphas: {kwargs['init_alpha']}")
#     if kwargs.get("init_beta") is True:
#         if "cmIN" in norm_name:
#             kwargs["init_beta"] = [nodes_stats_dict[node]["train"]["beta"] for node in nodes_stats_dict]
#             if len(kwargs["init_beta"])<10:
#                 logger.info(f"Loaded init_betas: {kwargs['init_beta']}")
#         else:
#             if stats_dict is not None:
#                 kwargs["init_beta"] = stats_dict["train"]["beta"]
#             else:
#                 kwargs["init_beta"] = None
#             logger.info(f"Loaded init_betas: {kwargs['init_beta']}")


## Experimental

class GRevIN(DefaultNorm):
    """
    Generalized RevIN-style normalization with:
      x~      = h_{a,b}(x) = (x - mu_a) * inv_sigma_b
      x_mod   = gamma * x~ + nu
      y_aff   = alpha * f_theta(x_mod) + beta
      y       = h^{-1}_{c,d}(y_aff) = (1/inv_sigma_d) * y_aff + mu_c

    mu_a = a*mu
    inv_sigma_b = 1 + b*(1/(std+eps) - 1)

    Changes vs your original:
    - a,b,c,d are free Parameters and clamped to [0,1] at forward-time (no sigmoid gating).
    - Freeze utilities for parameter groups (shared + personalized variants).
    - Classmethods to build classical configurations using only init + freezes.
    - Inherits DefaultNorm (keeps attribute forwarding to underlying model).
    """

    def __init__(
        self,
        model: nn.Module,
        dim: int,
        eps: float = 1e-8,
        n_clusters: int | None = None,
        personalize: str = "none",  # "none", "affine", "all"
        unknown_cluster_id: int | None = None,
        start_in: bool = True,      # init (a,b,c,d) to 1 (IN) if True else 0 (None)
        tie_revin: bool = False,    # enforce symmetric RevIN inverse + output modulation invert
        clamp_gamma_eps: float = 1e-6,
        latent: str = "none", # "none", "model", "affine"
    ):
        super().__init__(model, latent=False)
        assert personalize in {"none", "affine", "all"}
        self.norm_name = "grevin"
        self.dim = dim
        self.eps = eps
        self.n_clusters = n_clusters
        self.personalize = personalize
        self.unknown_cluster_id = unknown_cluster_id
        self.tie_revin = tie_revin
        self.clamp_gamma_eps = clamp_gamma_eps

        def free_gate_param(init_one: bool):
            return nn.Parameter(torch.full((1, dim, 1), 1.0 if init_one else 0.0))

        # Shared (fallback / default) parameters
        self.a = free_gate_param(start_in)
        self.b = free_gate_param(start_in)
        self.c = None if tie_revin else free_gate_param(start_in)
        self.d = None if tie_revin else free_gate_param(start_in)

        self.gamma = nn.Parameter(torch.ones(1, dim, 1))
        self.nu = nn.Parameter(torch.zeros(1, dim, 1))

        self.alpha = nn.Parameter(torch.ones(1, dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1))

        # Optional cluster parameters
        self._has_clusters = n_clusters is not None and int(n_clusters) > 0
        if self._has_clusters and personalize != "none":
            def make_list(init_tensor: torch.Tensor):
                return nn.ParameterList([nn.Parameter(init_tensor.clone()) for _ in range(int(n_clusters))])

            if personalize in {"affine", "all"}:
                self.gamma_k = make_list(torch.ones(1, dim, 1))
                self.nu_k = make_list(torch.zeros(1, dim, 1))
                self.alpha_k = make_list(torch.ones(1, dim, 1))
                self.beta_k = make_list(torch.zeros(1, dim, 1))

            if personalize == "all":
                self.a_k = make_list(self.a.data)
                self.b_k = make_list(self.b.data)
                if not tie_revin:
                    self.c_k = make_list(self.c.data)
                    self.d_k = make_list(self.d.data)

        # forward-time cache (per batch)
        self.mu = None
        self.std = None

    # ---------- helpers ----------

    def _parse_cluster(self, c):
        if c is None:
            return None
        if isinstance(c, torch.Tensor):
            if c.ndim == 1:
                return c
            if c.ndim == 2:
                return c[:, 0]
            return c[:, 0, 0]
        return None

    def _is_unknown(self, k: int):
        if self.unknown_cluster_id is not None and k == int(self.unknown_cluster_id):
            return True
        if self.n_clusters is None:
            return True
        return k < 0 or k >= int(self.n_clusters)

    def _clamp01(self, t: torch.Tensor):
        return t.clamp(0.0, 1.0)

    def _select_params(self, cluster: torch.Tensor | None):
        # returns tensors shaped (B, C, 1)
        B = 1 if cluster is None else int(cluster.shape[0])
        device = self.gamma.device

        def expand_shared(t: torch.Tensor):
            return t.expand(B, -1, -1)

        # shared
        a = expand_shared(self._clamp01(self.a))
        b = expand_shared(self._clamp01(self.b))
        if self.tie_revin:
            c = a
            d = b
        else:
            c = expand_shared(self._clamp01(self.c))
            d = expand_shared(self._clamp01(self.d))

        gamma = expand_shared(self.gamma)
        nu = expand_shared(self.nu)
        alpha = expand_shared(self.alpha)
        beta = expand_shared(self.beta)

        if not (self._has_clusters and self.personalize != "none" and cluster is not None):
            return a, b, c, d, gamma, nu, alpha, beta

        a_out, b_out, c_out, d_out = [], [], [], []
        gamma_out, nu_out, alpha_out, beta_out = [], [], [], []

        for ki in cluster.tolist():
            k = int(ki)

            if self._is_unknown(k):
                a_out.append(a[:1])
                b_out.append(b[:1])
                c_out.append(c[:1])
                d_out.append(d[:1])
                gamma_out.append(gamma[:1])
                nu_out.append(nu[:1])
                alpha_out.append(alpha[:1])
                beta_out.append(beta[:1])
                continue

            # affine personalization
            if self.personalize in {"affine", "all"}:
                gamma_out.append(self.gamma_k[k])
                nu_out.append(self.nu_k[k])
                alpha_out.append(self.alpha_k[k])
                beta_out.append(self.beta_k[k])
            else:
                gamma_out.append(gamma[:1])
                nu_out.append(nu[:1])
                alpha_out.append(alpha[:1])
                beta_out.append(beta[:1])

            # full personalization
            if self.personalize == "all":
                a_out.append(self._clamp01(self.a_k[k]))
                b_out.append(self._clamp01(self.b_k[k]))
                if self.tie_revin:
                    c_out.append(self._clamp01(self.a_k[k]))
                    d_out.append(self._clamp01(self.b_k[k]))
                else:
                    c_out.append(self._clamp01(self.c_k[k]))
                    d_out.append(self._clamp01(self.d_k[k]))
            else:
                a_out.append(a[:1])
                b_out.append(b[:1])
                c_out.append(c[:1])
                d_out.append(d[:1])

        a = torch.cat(a_out, dim=0).to(device)
        b = torch.cat(b_out, dim=0).to(device)
        c = torch.cat(c_out, dim=0).to(device)
        d = torch.cat(d_out, dim=0).to(device)
        gamma = torch.cat(gamma_out, dim=0).to(device)
        nu = torch.cat(nu_out, dim=0).to(device)
        alpha = torch.cat(alpha_out, dim=0).to(device)
        beta = torch.cat(beta_out, dim=0).to(device)

        return a, b, c, d, gamma, nu, alpha, beta

    # ---------- freeze API ----------

    def freeze(self, groups: str | list[str], freeze: bool = True):
        """
        groups can be a string or list of strings in:
          "ab", "cd", "gamma_nu", "alpha_beta"
        This freezes BOTH shared and personalized variants (if present).
        """
        if isinstance(groups, str):
            groups = [groups]

        def set_req_grad(p, rg: bool):
            if p is None:
                return
            if isinstance(p, nn.Parameter):
                p.requires_grad_(rg)
            elif isinstance(p, (nn.ParameterList, list, tuple)):
                for pp in p:
                    if pp is not None:
                        pp.requires_grad_(rg)

        rg = not freeze

        for g in groups:
            if g == "ab":
                set_req_grad(self.a, rg)
                set_req_grad(self.b, rg)
                if hasattr(self, "a_k"):
                    set_req_grad(self.a_k, rg)
                if hasattr(self, "b_k"):
                    set_req_grad(self.b_k, rg)

            elif g == "cd":
                if not self.tie_revin:
                    set_req_grad(self.c, rg)
                    set_req_grad(self.d, rg)
                    if hasattr(self, "c_k"):
                        set_req_grad(self.c_k, rg)
                    if hasattr(self, "d_k"):
                        set_req_grad(self.d_k, rg)

            elif g == "gamma_nu":
                set_req_grad(self.gamma, rg)
                set_req_grad(self.nu, rg)
                if hasattr(self, "gamma_k"):
                    set_req_grad(self.gamma_k, rg)
                if hasattr(self, "nu_k"):
                    set_req_grad(self.nu_k, rg)

            elif g == "alpha_beta":
                set_req_grad(self.alpha, rg)
                set_req_grad(self.beta, rg)
                if hasattr(self, "alpha_k"):
                    set_req_grad(self.alpha_k, rg)
                if hasattr(self, "beta_k"):
                    set_req_grad(self.beta_k, rg)

            else:
                raise ValueError(f"Unknown freeze group: {g!r}")

        return self

    def get_params(self, c: torch.Tensor | int | None = None, clamp: bool = True):
        """
        Return effective parameters (a,b,c,d, alpha,beta,gamma,nu).

        - If c is None: returns shared parameters shaped (1, dim, 1)
        - If c is an int: returns that cluster's parameters shaped (1, dim, 1)
        - If c is a tensor: returns per-sample parameters shaped (B, dim, 1)

        Notes:
        - a,b,c,d are clamped to [0,1] if clamp=True (matches forward-time behavior)
        - if tie_revin=True, returns tied (c=a, d=b) and effective (alpha,beta) implied by (gamma,nu)
        """
        if c is None:
            cluster = None
        elif isinstance(c, int):
            cluster = torch.tensor([c], device=self.gamma.device, dtype=torch.long)
        else:
            cluster = self._parse_cluster(c)

        a, b, cc, d, gamma, nu, alpha, beta = self._select_params(cluster)

        if not clamp:
            # reconstruct unclamped versions for the shared case only
            if cluster is None:
                a = self.a.expand(1, -1, -1)
                b = self.b.expand(1, -1, -1)
                if self.tie_revin:
                    cc, d = a, b
                else:
                    cc = self.c.expand(1, -1, -1)
                    d = self.d.expand(1, -1, -1)

        if self.tie_revin:
            gamma_safe = gamma.clamp_min(self.clamp_gamma_eps)
            alpha = 1.0 / gamma_safe
            beta = -nu / gamma_safe
            cc, d = a, b

        return {
            "a": a, "b": b, "c": cc, "d": d,
            "alpha": alpha, "beta": beta,
            "gamma": gamma, "nu": nu,
        }


    # ---------- norm / denorm ----------

    def norm(self, x: torch.Tensor, cluster: torch.Tensor | None = None):
        # x: (B, C, T)
        mu, std = get_normal_stats(x)
        self.mu, self.std = mu, std

        a, b, _, _, gamma, nu, _, _ = self._select_params(cluster)
        mu_a = a * mu
        inv_sigma_b = 1.0 + b * (1.0 / (std + self.eps) - 1.0)
        x_hat = (x - mu_a) * inv_sigma_b
        x_mod = gamma * x_hat + nu
        return x_mod

    def denorm(self, y: torch.Tensor, cluster: torch.Tensor | None = None, latent=False):
        assert self.mu is not None and self.std is not None, "Call norm() before denorm()."
        mu, std = self.mu, self.std

        a, b, c, d, gamma, nu, alpha, beta = self._select_params(cluster)

        if self.tie_revin:
            gamma_safe = gamma.clamp_min(self.clamp_gamma_eps)
            alpha = 1.0 / gamma_safe
            beta = -nu / gamma_safe
            c, d = a, b

        y_aff = alpha * y + beta
        if latent:
            return y_aff
        
        mu_c = c * mu
        inv_sigma_d = 1.0 + d * (1.0 / (std + self.eps) - 1.0)
        y_out = y_aff / inv_sigma_d + mu_c
        return y_out

    def forward(self, x: torch.Tensor, c=None):
        cluster = self._parse_cluster(c)
        x_mod = self.norm(x, cluster)
        pred = self.model(x_mod, c)
        if self.latent == "model":
            return pred
        elif self.latent == "affine":
            return self.denorm(pred, cluster, latent=True)
        else:
            return self.denorm(pred, cluster)

    # ---------- presets (only init + freezes) ----------

    @classmethod
    def build_in(
        cls,
        model: nn.Module,
        dim: int,
        eps: float = 1e-8,
        latent: bool = False,
        **kwargs,
    ):
        """
        Classical Instance Normalization (IN):
        - a=b=c=d=1  (pure normalization + exact inverse)
        - gamma=1, nu=0
        - alpha=1, beta=0
        - everything frozen
        """
        m = cls(
            model=model,
            dim=dim,
            eps=eps,
            tie_revin=False,
            start_in=True,
            personalize="none",
            latent=latent,
            **kwargs,
        )
        m.freeze(["ab", "cd", "gamma_nu", "alpha_beta"], freeze=True)
        return m


    @classmethod
    def build_revin(
        cls, 
        model: nn.Module, 
        dim: int, 
        eps: float = 1e-8, 
        latent: bool = False, 
        **kwargs
    ):
        """
        Classical RevIN:
          - tie_revin=True
          - init a=b=c=d=1 (pure IN), gamma=1, nu=0
          - freeze everything except gamma,nu
        """
        m = cls(
            model=model,
            dim=dim,
            eps=eps,
            tie_revin=True,
            start_in=True,
            personalize="none",
            latent=latent,
            **kwargs,
        )
        m.freeze(["ab", "cd", "alpha_beta"], freeze=True)
        m.freeze(["gamma_nu"], freeze=False)
        return m

    @classmethod
    def build_personalized_revin(
        cls,
        model: nn.Module,
        dim: int,
        n_clusters: int,
        eps: float = 1e-8,
        unknown_cluster_id: int | None = None,
        latent: bool = False,
        **kwargs,
    ):
        """
        Personalized RevIN:
          - tie_revin=True
          - init a=b=c=d=1
          - personalize affine (gamma,nu per cluster; alpha/beta exist but are tied away at runtime)
          - freeze everything except gamma/nu (shared + per-cluster)
        """
        m = cls(
            model=model,
            dim=dim,
            eps=eps,
            n_clusters=n_clusters,
            personalize="affine",
            unknown_cluster_id=unknown_cluster_id,
            tie_revin=True,
            start_in=True,
            latent=latent,
            **kwargs,
        )
        m.freeze(["ab", "cd", "alpha_beta"], freeze=True)
        m.freeze(["gamma_nu"], freeze=False)
        return m

    @classmethod
    def build_cmin(
        cls,
        model: nn.Module,
        dim: int,
        n_clusters: int | None = None,
        eps: float = 1e-8,
        unknown_cluster_id: int | None = None,
        latent: bool = False,
        **kwargs,
    ):
        """
        "cmIN" style (as requested): freeze everything except (alpha,beta).
        Default init is pure IN: a=b=c=d=1 and gamma=1,nu=0.
        If n_clusters is provided, alpha/beta personalization is enabled via personalize="affine",
        but gamma/nu are frozen (shared + per-cluster).
        """
        personalize = "affine" if (n_clusters is not None and int(n_clusters) > 0) else "none"
        m = cls(
            model=model,
            dim=dim,
            eps=eps,
            n_clusters=n_clusters,
            personalize=personalize,
            unknown_cluster_id=unknown_cluster_id,
            tie_revin=False,
            start_in=True,
            latent=latent,
            **kwargs,
        )
        m.freeze(["ab", "cd", "gamma_nu"], freeze=True)
        m.freeze(["alpha_beta"], freeze=False)
        return m

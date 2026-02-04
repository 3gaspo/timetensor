## Adapted from https://github.com/PriorLabs/TabPFN and https://github.com/PriorLabs/tabpfn-time-series

import torch
import numpy as np
from tabpfn import TabPFNRegressor

from hydra.utils import to_absolute_path
import torch
import numpy as np
import torch.nn.functional as F


class TabPFN:
    def __init__(self, lags, horizon, context_mode="past_only", seasonal_periods=None,
        cross_learning=False, dimension_encoding="ordinal",
        device="cuda", weights_path="src/timetensor/sota/tabpfnts/weights/tabpfn-v2.5-regressor-v2.5_default.ckpt"
    ):
        self.lags = lags
        self.horizon = horizon
        self.context_mode = context_mode # "past_only" | "future" | "any"
        self.device = device
        self.cross_learning = cross_learning
        self.dimension_encoding = dimension_encoding # "ordinal" | "one-hot" | "categorical"

        ## seasonality
        if seasonal_periods:
            self.seasonal_periods = seasonal_periods
        else: # assumes hourly data
            self.seasonal_periods = []
            if lags > 24: self.seasonal_periods.append(24)#lags//24)
            if lags > 168: self.seasonal_periods.append(168)#lags//168)
        
        local_model_dir = to_absolute_path(weights_path)
        self.model = TabPFNRegressor(device=device, model_path=local_model_dir)
    
    def _generate_time_features(self, input_len, window_len, device, dtype):
        """returns tensor of time index features"""
        t_idx = torch.arange(window_len, device=device, dtype=dtype)
        norm_idx = t_idx / input_len
        features = [norm_idx.unsqueeze(1)]

        for p in self.seasonal_periods:
            omega = 2 * np.pi / p
            features.append(torch.sin(omega * t_idx).unsqueeze(1))
            features.append(torch.cos(omega * t_idx).unsqueeze(1))
            
        return torch.cat(features, dim=1) # (window_len, n_t)

    def _split_context(self, c):
        """returns (past_only, future_included) contexts from c"""
        if c is None:
            return None, None

        if self.context_mode == "past_only":
            assert c.shape[-1] >= self.lags, f"Wrong context shape: {c.shape}"
            return c[:, :, :self.lags], None

        if self.context_mode == "future":
            assert c.shape[-1] == self.lags+self.horizon, f"Wrong context shape: {c.shape}"
            return None, c

        if self.context_mode == "any":
            assert type(c) == tuple and len(c) == 2, f"Wrong context shape: {type(c)}, {len(c)}"
            return c
        
        return None, None

    def _create_tabular_block(self, values, time_features, start_idx=0, has_context=False, channel_id=0):
        bs, dim, length = values.shape
        device = values.device
        
        tf_subset = time_features[start_idx: start_idx+length].unsqueeze(0).unsqueeze(0)  # (1, 1, length, n_t)
        X = tf_subset.expand(bs, dim, length, -1)  # (bs, dim, length, n_t)
        
        if has_context and self.dimension_encoding == "ordinal":
            chan_idx = torch.full((bs, dim, length, 1), float(channel_id), device=device, dtype=time_features.dtype)
            X = torch.cat([X, chan_idx], dim=-1)

        if dim > 1:
            if self.dimension_encoding == "ordinal":
                d_enc = torch.arange(dim, device=device, dtype=time_features.dtype).view(1, dim, 1, 1).expand(bs, dim, length, 1)
            elif self.dimension_encoding == "one-hot":
                d_enc = F.one_hot(torch.arange(dim, device=device), num_classes=dim).float()
                d_enc = d_enc.view(1, dim, 1, -1).expand(bs, dim, length, -1)
            elif self.dimension_encoding == "categorical":
                raise ValueError("TODO")
            X = torch.cat([X, d_enc], dim=-1)
            
        if bs > 1:
            if self.dimension_encoding == "ordinal":
                b_enc = torch.arange(bs, device=device, dtype=time_features.dtype).view(bs, 1, 1, 1).expand(bs, dim, length, 1)
            elif self.dimension_encoding == "one-hot":
                b_enc = F.one_hot(torch.arange(bs, device=device), num_classes=bs).float()
                b_enc = b_enc.view(bs, 1, 1, -1).expand(bs, dim, length, -1)
            elif self.dimension_encoding == "categorical":
                raise ValueError("TODO")
            X = torch.cat([X, b_enc], dim=-1)

        X_flat = X.reshape(-1, X.shape[-1]) # (bs * dim * length, n_features)
        y_flat = values.reshape(-1) # (bs * dim * length)
        
        return X_flat, y_flat

    def _prepare_matrix(self, x, time_features, past_context, future_context):
        bs, dim, lags = x.shape
        horizon = self.horizon
        has_context = (past_context is not None) or (future_context is not None)
        X_train_blocks, y_train_blocks = [], []

        X_t, y_t = self._create_tabular_block(x, time_features, channel_id=0, has_context=has_context)
        X_train_blocks.append(X_t)
        y_train_blocks.append(y_t)
        
        if past_context is not None:
            X_c, y_c = self._create_tabular_block(past_context, time_features, channel_id=1, has_context=True)
            X_train_blocks.append(X_c)
            y_train_blocks.append(y_c)
            
        if future_context is not None:
            X_cf, y_cf = self._create_tabular_block(future_context, time_features, channel_id=1, has_context=True)
            X_train_blocks.append(X_cf)
            y_train_blocks.append(y_cf)

        dummy = torch.zeros((bs, dim, horizon), device=x.device)
        X_test, _ = self._create_tabular_block(dummy, time_features, start_idx=lags, channel_id=0)

        X_train = torch.cat(X_train_blocks, dim=0)
        y_train = torch.cat(y_train_blocks, dim=0)
        
        return X_train.cpu().numpy(), y_train.cpu().numpy(), X_test.cpu().numpy()

    def __call__(self, x, c=None): # x (bs, dim, lags)
        assert x.shape[-1] == self.lags
        bs, dim, lags = x.shape
        horizon = self.horizon

        past_only, future_included = self._split_context(c)
        time_features = self._generate_time_features(lags, lags + horizon, x.device, x.dtype)

        if self.cross_learning: #process all batch
            X_train, y_train, X_test = self._prepare_matrix(x, time_features, past_only, future_included)
            self.model.fit(X_train, y_train)
            preds_flat = self.model.predict(X_test)
            preds = torch.from_numpy(preds_flat).to(x.device).reshape(bs, dim, horizon)

        else: #process per sample
            preds_list = []
            for i in range(bs):
                x_i = x[i].unsqueeze(0)
                past_only_i, future_included_i = past_only, future_included
                if past_only is not None and past_only.shape[0] == bs:
                    past_only_i = past_only[i].unsqueeze(0)
                if future_included and future_included.shape[0] == bs:
                    future_included_i = future_included[i].unsqueeze(0)
                X_train, y_train, X_test = self._prepare_matrix(x_i, time_features, past_only_i, future_included_i)
                
                self.model.fit(X_train, y_train)
                p_flat = self.model.predict(X_test)
                p_reshaped = torch.from_numpy(p_flat).to(x.device).reshape(1, dim, horizon)
                preds_list.append(p_reshaped)
            preds = torch.cat(preds_list, dim=0)

        return preds # (bs, dim, horizon)
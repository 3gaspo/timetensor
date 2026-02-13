## Adapted from https://github.com/PriorLabs/TabPFN and https://github.com/PriorLabs/tabpfn-time-series

import torch
import numpy as np
import torch.nn.functional as F

from tabpfn import TabPFNRegressor

from hydra.utils import to_absolute_path


class TabPFN:
    """Wrapper for TabPFN for time series forecasting with covariates"""
    def __init__(self, lags, horizon, context_mode="past_only", seasonal_periods=None,
        cross_learning=False, dimension_encoding="ordinal", context_as_features=True,
        device="cuda", weights_path="src/timetensor/sota/tabpfnts/weights/tabpfn-v2.5-regressor-v2.5_default.ckpt"
    ):
        self.lags = lags
        self.horizon = horizon
        self.context_mode = context_mode # "past_only" | "future" | "any" (decides if horizon is included in context)
        self.device = device
        self.cross_learning = cross_learning
        self.dimension_encoding = dimension_encoding # "ordinal" | "one-hot" | "categorical"
        self.context_as_features = context_as_features

        ## seasonality for time index embeddings
        if seasonal_periods:
            self.seasonal_periods = seasonal_periods
        else: # assumes hourly data
            self.seasonal_periods = []
            if lags > 24: self.seasonal_periods.append(24)
            if lags > 168: self.seasonal_periods.append(168)
        
        local_model_dir = to_absolute_path(weights_path)
        self.model = TabPFNRegressor(device=device, model_path=local_model_dir)
    
    def _generate_time_features(self, lookback, window_length, device, dtype):
        """returns tensor of time index features"""
        t_idx = torch.arange(window_length, device=device, dtype=dtype)
        norm_idx = t_idx / lookback
        features = [norm_idx.unsqueeze(1)]

        for p in self.seasonal_periods:
            omega = 2 * np.pi / p
            features.append(torch.sin(omega * t_idx).unsqueeze(1))
            features.append(torch.cos(omega * t_idx).unsqueeze(1))
            
        return torch.cat(features, dim=1) # (window_len, n_t)  n_t = 1 + 2*len(seasonal_periods)

    def _split_context(self, c): #WARNING: future_included is of size L+H, not H
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
            assert c[0].shape[-1] == self.lags and c[1].shape[-1] == self.lags+self.horizon, f"Wrong context shapes: {c[0].shape}, {c[1].shape}"
            return c
        
        return None, None

    def _create_tabular_block(self, values, time_features, context_values=None,
        start_idx=0, bs_offset=0, d_offset=0, total_bs_classes=1, total_dim_classes=1):
        """creates X,y matrices from values, time features and optional context features"""

        bs, dim, length = values.shape
        device, dtype = values.device, time_features.dtype

        tf_subset = time_features[start_idx:start_idx + length].view(1, 1, length, -1)
        X = tf_subset.expand(bs, dim, length, -1)  # (bs, dim, length, n_t)

        if self.context_as_features:
            if context_values is not None:
                c_feat = context_values.permute(0, 2, 1).unsqueeze(1) # (bs, 1, length, c_dim)
                c_feat = c_feat.to(dtype).expand(bs, dim, length, -1)
                X = torch.cat([X, c_feat], dim=-1) #(bs, dim, length, n_t + c_dim)
        
        else:
            b_ids = (torch.arange(bs, device=device) + bs_offset).view(bs, 1, 1)          # (bs,1,1)
            d_ids = (torch.arange(dim, device=device) + d_offset).view(1, dim, 1)         # (1,dim,1)

            if self.dimension_encoding == "ordinal":
                b_feat = b_ids.to(dtype).expand(bs, dim, length).unsqueeze(-1)            # (bs,dim,length,1)
                d_feat = d_ids.to(dtype).expand(bs, dim, length).unsqueeze(-1)            # (bs,dim,length,1)
            elif self.dimension_encoding == "one-hot":
                b_oh = F.one_hot(b_ids.view(-1), num_classes=total_bs_classes).to(dtype)  # (bs, total_bs)
                d_oh = F.one_hot(d_ids.view(-1), num_classes=total_dim_classes).to(dtype) # (dim, total_dim)
                b_feat = b_oh.view(bs, 1, 1, -1).expand(bs, dim, length, -1)              # (bs,dim,length,total_bs)
                d_feat = d_oh.view(1, dim, 1, -1).expand(bs, dim, length, -1)             # (bs,dim,length,total_dim)
            else:
                raise ValueError(f"Unknown id_encoding: {self.dimension_encoding}")

            X = torch.cat([X, b_feat, d_feat], dim=-1) # (bs, dim, length, n_t +  1 or (total_bs+total_dim) )

        return X.reshape(-1, X.shape[-1]), values.reshape(-1)
    
    def _prepare_matrix(self, x, time_features, past_context, future_context):
        """returns X_train, y_train, X_test"""
        bs, dim, lags = x.shape
        horizon = self.horizon

        if self.context_as_features: #add context as features (assumes known horizon context)
            c_train, c_test = None, None
            if future_context is not None:
                c_train = future_context[:, :, :lags]
                c_test = future_context[:, :, lags:lags+horizon]

            X_train, y_train = self._create_tabular_block(x, time_features,
                context_values=c_train, start_idx=0)

            dummy = torch.zeros((bs, dim, horizon), device=x.device, dtype=x.dtype)
            X_test, _ = self._create_tabular_block(dummy, time_features,
                context_values=c_test, start_idx=lags)

        else: #add context as additional samples (requires channel features)
            past_bs = 0 if past_context is None else past_context.shape[0]
            past_dim = 0 if past_context is None else past_context.shape[1]
            future_bs = 0 if future_context is None else future_context.shape[0]
            future_dim = 0 if future_context is None else future_context.shape[1]
            total_samples = max(1, bs + past_bs + future_bs)
            max_dims = max(1, dim, past_dim, future_dim)

            def make_block(vals, start_idx, bs_offset, d_offset):
                return self._create_tabular_block(
                    vals,
                    time_features,
                    start_idx=start_idx,
                    bs_offset=bs_offset,
                    d_offset=d_offset,
                    total_bs_classes=total_samples,
                    total_dim_classes=max_dims,
                )

            X_train_blocks, y_train_blocks = [], []

            # targets lookback
            X_t, y_t = make_block(x, start_idx=0, bs_offset=0, d_offset=0)
            X_train_blocks.append(X_t)
            y_train_blocks.append(y_t)

            # past context
            if past_context is not None:
                X_p, y_past = make_block(past_context, start_idx=0, bs_offset=bs, d_offset=dim)
                X_train_blocks.append(X_p)
                y_train_blocks.append(y_past)

            # future context (lookback and horizon)
            if future_context is not None:
                X_fp, y_fp = make_block(future_context[:, :, :lags], start_idx=0, bs_offset=bs+past_bs, d_offset=0)
                X_train_blocks.append(X_fp)
                y_train_blocks.append(y_fp)

                X_ff, y_ff = make_block(future_context[:, :, lags:], start_idx=lags, bs_offset=bs+past_bs, d_offset=0)
                X_train_blocks.append(X_ff)
                y_train_blocks.append(y_ff)

            X_train = torch.cat(X_train_blocks, dim=0)
            y_train = torch.cat(y_train_blocks, dim=0)

            # targets horizon
            dummy = torch.zeros((bs, dim, horizon), device=x.device, dtype=x.dtype)
            X_test, _ = make_block(dummy, start_idx=lags, bs_offset=0, d_offset=0)

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
                x_i = x[i].unsqueeze(0) # (1, dim, lags)
                past_only_i, future_included_i = past_only, future_included # (1*(bs_c), dim_c, lags)
                if past_only is not None and past_only.shape[0] == bs:
                    past_only_i = past_only[i].unsqueeze(0) 
                if future_included is not None and future_included.shape[0] == bs:
                    future_included_i = future_included[i].unsqueeze(0)
                X_train, y_train, X_test = self._prepare_matrix(x_i, time_features, past_only_i, future_included_i)
                
                self.model.fit(X_train, y_train)
                p_flat = self.model.predict(X_test)
                p_reshaped = torch.from_numpy(p_flat).to(x.device).reshape(1, dim, horizon)
                preds_list.append(p_reshaped)
            preds = torch.cat(preds_list, dim=0)

        return preds # (bs, dim, horizon)
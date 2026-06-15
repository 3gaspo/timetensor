## Adapted from https://github.com/PriorLabs/TabPFN and https://github.com/PriorLabs/tabpfn-time-series
import numpy as np
import torch
import torch.nn.functional as F

from tabpfn import TabPFNRegressor
from hydra.utils import to_absolute_path


class TabPFN:
    """TabPFN wrapper for time-series forecasting with optional covariates.

    Args:
        context_mode: "past_only" | "future" | "any"
        seasonal_periods: Optional list of seasonal periods for time features.
        cross_learning: If True, fit one model on the whole batch.
        dimension_encoding: "ordinal" | "one-hot" | "categorical"
        context_as_features: If True, future-known context is appended as features.
        weights_path: Local path to TabPFN weights.
        shared_context: If True and cross_learning=False, all provided context batch items
            are available to each sample.
    """

    def __init__(
        self,
        lags,
        horizon,
        context_mode="past_only",
        seasonal_periods=None,
        cross_learning=False,
        dimension_encoding="ordinal",
        context_as_features=True,
        use_time_features=True,
        device="cuda",
        weights_path="src/timetensor/sota/tabpfnts/weights/tabpfn-v2.5-regressor-v2.5_default.ckpt",
        shared_context=False,
        **kwargs
    ):
        self.lags = lags
        self.horizon = horizon
        self.context_mode = context_mode
        self.device = device
        self.cross_learning = cross_learning
        self.dimension_encoding = dimension_encoding
        self.context_as_features = context_as_features
        self.shared_context = shared_context
        self.use_time_features = use_time_features

        if seasonal_periods is not None:
            self.seasonal_periods = seasonal_periods
        else:
            self.seasonal_periods = []
            if lags > 24:
                self.seasonal_periods.append(24)
            if lags > 168:
                self.seasonal_periods.append(168)

        if not self.use_time_features and not self.context_as_features:
            raise ValueError("use_time_features=False requires context_as_features=True")
    
        local_model_dir = to_absolute_path(weights_path)
        self.model = TabPFNRegressor(device=device, model_path=local_model_dir)

    def _generate_time_features(self, lookback, window_length, device, dtype):
        """generates embeddings (t/L, cos(wt), sin(wt) for w in seasonal_periods)"""
        if not self.use_time_features:
            return None
    
        t_idx = torch.arange(window_length, device=device, dtype=dtype)  # (window_length,)
        norm_idx = t_idx / lookback
        features = [norm_idx.unsqueeze(1)]  # [(window_length, 1)]

        for p in self.seasonal_periods:
            omega = 2 * np.pi / p
            features.append(torch.sin(omega * t_idx).unsqueeze(1))  # (window_length, 1)
            features.append(torch.cos(omega * t_idx).unsqueeze(1))  # (window_length, 1)

        return torch.cat(features, dim=1)  # (window_length, n_time_features)
        
    def _split_context(self, c):
        if c is None:
            return None, None

        if self.context_mode == "past_only":
            assert torch.is_tensor(c), f"Expected tensor context, got {type(c)}"
            assert c.shape[-1] >= self.lags, f"Wrong context shape: {tuple(c.shape)}"
            return c[:, :, : self.lags], None

        if self.context_mode == "future":
            assert torch.is_tensor(c), f"Expected tensor context, got {type(c)}"
            assert c.shape[-1] == self.lags + self.horizon, f"Wrong context shape: {tuple(c.shape)}"
            return None, c

        if self.context_mode == "any":
            assert isinstance(c, tuple) and len(c) == 2, f"Expected tuple(past, future), got {type(c)}"
            past_context, future_context = c
            assert past_context.shape[-1] == self.lags, f"Wrong past context shape: {tuple(past_context.shape)}"
            assert future_context.shape[-1] == self.lags + self.horizon, (
                f"Wrong future context shape: {tuple(future_context.shape)}"
            )
            return past_context, future_context

        raise ValueError(f"Unknown context_mode: {self.context_mode}")

    def _append_identity_features(
        self,
        X,
        bs,
        dim,
        length,
        device,
        dtype,
        bs_offset=0,
        d_offset=0,
        total_bs_classes=1,
        total_dim_classes=1,
    ):
        b_ids = (torch.arange(bs, device=device) + bs_offset).view(bs, 1, 1)  # (bs, 1, 1)
        d_ids = (torch.arange(dim, device=device) + d_offset).view(1, dim, 1)  # (1, dim, 1)

        if self.dimension_encoding == "ordinal":
            b_feat = b_ids.to(dtype).expand(bs, dim, length).unsqueeze(-1)  # (bs, dim, length, 1)
            d_feat = d_ids.to(dtype).expand(bs, dim, length).unsqueeze(-1)  # (bs, dim, length, 1)
            return torch.cat([X, b_feat, d_feat], dim=-1)  # (bs, dim, length, n_features + 2)

        if self.dimension_encoding == "one-hot":
            series_id = b_ids * total_dim_classes + d_ids  # (bs, dim, 1)
            num_series = total_bs_classes * total_dim_classes
            series_oh = F.one_hot(series_id.expand(bs, dim, length), num_classes=num_series).to(dtype)
            return torch.cat([X, series_oh], dim=-1)  # (bs, dim, length, n_features + num_series)

        if self.dimension_encoding == "categorical":
            raise NotImplementedError("dimension_encoding='categorical' is not implemented yet.")

        raise ValueError(f"Unknown dimension_encoding: {self.dimension_encoding}")

    def _create_tabular_block(
        self,
        values,
        time_features,
        context_values=None,
        start_idx=0,
        bs_offset=0,
        d_offset=0,
        total_bs_classes=1,
        total_dim_classes=1,
    ):
        bs, dim, length = values.shape
        device, dtype = values.device, values.dtype

        feature_parts = []

        if time_features is not None:
            tf_subset = time_features[start_idx : start_idx + length]  # (length, n_t)
            feature_parts.append(tf_subset.view(1, 1, length, -1).expand(bs, dim, length, -1))  # (bs, dim, length, n_t)

        if self.context_as_features and context_values is not None:
            bs_c, dim_c, c_len = context_values.shape
            assert c_len == length, f"Context length mismatch: got {c_len}, expected {length}"

            c_feat = context_values.permute(2, 0, 1).reshape(length, bs_c * dim_c)
            c_feat = c_feat.view(1, 1, length, -1).expand(bs, dim, length, -1)  # (bs, dim, length, bs_c*dim_c)
            # X = torch.cat([X, c_feat], dim=-1)  
            feature_parts.append(c_feat)

        if not feature_parts:
            raise ValueError("No base features available. Enable time features or provide context_as_features inputs.")
        X = torch.cat(feature_parts, dim=-1) # (bs, dim, length, n_features)

        X = self._append_identity_features(
            X=X,
            bs=bs,
            dim=dim,
            length=length,
            device=device,
            dtype=dtype,
            bs_offset=bs_offset,
            d_offset=d_offset,
            total_bs_classes=total_bs_classes,
            total_dim_classes=total_dim_classes,
        )  # (bs, dim, length, n_features)

        return X.reshape(-1, X.shape[-1]), values.reshape(-1)  # (bs*dim*length, n_features), (bs*dim*length,)

    def _prepare_matrix(self, x, time_features, past_context, future_context):
        bs, dim, lags = x.shape
        horizon = self.horizon

        if self.context_as_features:
            c_train, c_test = None, None
            if future_context is not None: #we don't use past_context as it doesn't have horizon values for Xtest
                c_train = future_context[:, :, :lags]  # (bs_c_f, dim_c_f, lags)
                c_test = future_context[:, :, lags : lags + horizon]  # (bs_c_f, dim_c_f, horizon)

            X_train, y_train = self._create_tabular_block(
                values=x,  # (bs, dim, lags)
                time_features=time_features,  # (lags+horizon, n_t)
                context_values=c_train,  # (bs_c_f, dim_c_f, lags) | None
                start_idx=0,
                bs_offset=0,
                d_offset=0,
                total_bs_classes=max(1, bs),
                total_dim_classes=max(1, dim),
            )

            dummy = torch.zeros((bs, dim, horizon), device=x.device, dtype=x.dtype)  # (bs, dim, horizon)
            X_test, _ = self._create_tabular_block(
                values=dummy,  # (bs, dim, horizon)
                time_features=time_features,  # (lags+horizon, n_t)
                context_values=c_test,  # (bs_c_f, dim_c_f, horizon) | None
                start_idx=lags,
                bs_offset=0,
                d_offset=0,
                total_bs_classes=max(1, bs),
                total_dim_classes=max(1, dim),
            )

            return X_train.cpu().numpy(), y_train.cpu().numpy(), X_test.cpu().numpy()

        past_bs = 0 if past_context is None else past_context.shape[0]
        past_dim = 0 if past_context is None else past_context.shape[1]
        future_bs = 0 if future_context is None else future_context.shape[0]
        future_dim = 0 if future_context is None else future_context.shape[1]

        total_bs_classes = max(1, bs + past_bs + future_bs)
        total_dim_classes = max(1, dim + past_dim + future_dim)

        def make_block(vals, start_idx, bs_offset, d_offset):
            return self._create_tabular_block(
                values=vals,
                time_features=time_features,
                context_values=None,
                start_idx=start_idx,
                bs_offset=bs_offset,
                d_offset=d_offset,
                total_bs_classes=total_bs_classes,
                total_dim_classes=total_dim_classes,
            )

        X_train_blocks, y_train_blocks = [], []

        X_t, y_t = make_block(
            x,  # (bs, dim, lags)
            start_idx=0,
            bs_offset=0,
            d_offset=0,
        )
        X_train_blocks.append(X_t)
        y_train_blocks.append(y_t)

        if past_context is not None:
            X_p, y_p = make_block(
                past_context,  # (past_bs, past_dim, lags)
                start_idx=0,
                bs_offset=bs,
                d_offset=dim,
            )
            X_train_blocks.append(X_p)
            y_train_blocks.append(y_p)

        if future_context is not None:
            X_fp, y_fp = make_block(
                future_context[:, :, :lags],  # (future_bs, future_dim, lags)
                start_idx=0,
                bs_offset=bs + past_bs,
                d_offset=dim + past_dim,
            )
            X_train_blocks.append(X_fp)
            y_train_blocks.append(y_fp)

            X_ff, y_ff = make_block(
                future_context[:, :, lags : lags + horizon],  # (future_bs, future_dim, horizon)
                start_idx=lags,
                bs_offset=bs + past_bs,
                d_offset=dim + past_dim,
            )
            X_train_blocks.append(X_ff)
            y_train_blocks.append(y_ff)

        X_train = torch.cat(X_train_blocks, dim=0)  # (n_train_rows, n_features)
        y_train = torch.cat(y_train_blocks, dim=0)  # (n_train_rows,)

        dummy = torch.zeros((bs, dim, horizon), device=x.device, dtype=x.dtype)  # (bs, dim, horizon)
        X_test, _ = make_block(
            dummy,  # (bs, dim, horizon)
            start_idx=lags,
            bs_offset=0,
            d_offset=0,
        )

        return X_train.cpu().numpy(), y_train.cpu().numpy(), X_test.cpu().numpy()

    def __call__(self, x, c=None):
        assert x.ndim == 3, f"Expected x with shape (bs, dim, lags), got {tuple(x.shape)}"
        assert x.shape[-1] == self.lags, f"Expected lags={self.lags}, got {x.shape[-1]}"

        bs, dim, lags = x.shape
        horizon = self.horizon

        past_only, future_included = self._split_context(c)
        time_features = self._generate_time_features(lags, lags + horizon, x.device, x.dtype)  # (lags+horizon, n_t)

        if self.cross_learning:
            X_train, y_train, X_test = self._prepare_matrix(
                x=x,  # (bs, dim, lags)
                time_features=time_features,  # (lags+horizon, n_t)
                past_context=past_only,  # (bs_c_p, dim_c_p, lags) | None
                future_context=future_included,  # (bs_c_f, dim_c_f, lags+horizon) | None
            )
            self.model.fit(X_train, y_train)
            preds_flat = self.model.predict(X_test)  # (bs*dim*horizon,)
            return torch.from_numpy(preds_flat).to(x.device).reshape(bs, dim, horizon)  # (bs, dim, horizon)

        preds_list = []
        for i in range(bs):
            x_i = x[i].unsqueeze(0)  # (1, dim, lags)

            past_only_i = past_only
            future_included_i = future_included

            if not self.shared_context:
                if past_only is not None and past_only.shape[0] == bs:
                    past_only_i = past_only[i].unsqueeze(0)  # (1, dim_c_p, lags)
                if future_included is not None and future_included.shape[0] == bs:
                    future_included_i = future_included[i].unsqueeze(0)  # (1, dim_c_f, lags+horizon)

            X_train, y_train, X_test = self._prepare_matrix(
                x=x_i,  # (1, dim, lags)
                time_features=time_features,  # (lags+horizon, n_t)
                past_context=past_only_i,  # (...) | None
                future_context=future_included_i,  # (...) | None
            )

            self.model.fit(X_train, y_train)
            preds_flat = self.model.predict(X_test)  # (dim*horizon,)
            preds_i = torch.from_numpy(preds_flat).to(x.device).reshape(1, dim, horizon)  # (1, dim, horizon)
            preds_list.append(preds_i)

        return torch.cat(preds_list, dim=0)  # (bs, dim, horizon)
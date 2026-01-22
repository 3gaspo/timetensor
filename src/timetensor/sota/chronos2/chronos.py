## Adapted from https://github.com/amazon-science/chronos-forecasting

import torch

from chronos import BaseChronosPipeline, Chronos2Pipeline
from hydra.utils import to_absolute_path


class Chronos:
    def __init__(self, lags, horizon, context_mode="past_only", cross_learning=False, device_map="cuda", weights_path="src/timetensor/sota/chronos2/weights"):
        super(Chronos, self).__init__()
        self.lags, self.horizon = lags, horizon
        self.context_mode = context_mode  # "past_only" | "future" | "any"
        self.cross_learning = bool(cross_learning)

        local_model_dir = to_absolute_path(weights_path)
        self.pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
            local_model_dir,
            device_map=device_map,
            local_files_only=True,
        )

    def _split_context(self, c):
        """returns (past_only, future_included) contexts from c"""
        if c is None:
            return None, None

        if self.context_mode == "past_only":
            assert c.shape[-1] >= self.lags, f"Wrong context shape: {c}"
            return c[:, :, :self.lags], None

        if self.context_mode == "future":
            assert c.shape[-1] == self.lags+self.horizon, f"Wrong context shape: {c}"
            return None, c

        if self.context_mode == "any":
            assert type(c) == tuple and len(c) == 2, f"Wrong context shape: {type(c)}, {len(c)}"
            return c

    def __call__(self, x, c=None): #x (bs, dim, lags)
        bs, dim, lags = x.shape
        assert lags == self.lags
        past_only, future_included = self._split_context(c)
        assert past_only is None or past_only.shape[-1] == self.lags
        assert future_included is None or future_included.shape == self.horizon

        inputs = []
        for b in range(bs):
            d = {"target": x[b].cpu()}  # (dim, lags)

            past_cov = {}
            fut_cov = {}

            if past_only is not None:
                cs, dim, ds = past_only.shape
                if bs == cs:
                    for i in range(dim):
                        past_cov[f"past_{i}"] = past_only[b, i, :].cpu()
                else:
                    for b_ in range(cs):
                        for i in range(dim):
                            past_cov[f"past_{b_}_{i}"] = past_only[b_, i, :].cpu()


            if future_included is not None:
                cs, dim, ds = future_included.shape
                if bs == cs:
                    for i in range(dim):
                        name = f"fut_{i}"
                        past_cov[name] = future_included[b, i, :lags]
                        fut_cov[name] = future_included[b, i, lags : lags + self.horizon]
                else:
                    for b_ in range(cs):
                        for i in range(dim):
                            past_cov[f"past_{b_}_{i}"] = past_only[b_, i, :]

            if len(past_cov) > 0:
                d["past_covariates"] = past_cov
            if len(fut_cov) > 0:
                d["future_covariates"] = fut_cov
            inputs.append(d)

        preds_list = self.pipeline.predict(
            inputs=inputs,
            prediction_length=self.horizon,
            cross_learning=self.cross_learning
        )

        out_items = []
        for pred in preds_list:  # pred: (dim, Q, H)
            q_med = pred.shape[1] // 2
            out_items.append(pred[:, q_med, :])

        return torch.stack(out_items, dim=0).to(device=x.device, dtype=x.dtype) # (bs, dim, horizon)
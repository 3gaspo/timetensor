## Adapted from https://github.com/amazon-science/chronos-forecasting

from chronos import BaseChronosPipeline, Chronos2Pipeline
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from hydra.utils import to_absolute_path

import os

class Chronos:
    def __init__(self, horizon, cross_learning=False):
        super(Chronos, self).__init__()
        self.horizon = horizon
        self.cross_learning=cross_learning
        
        local_model_dir = to_absolute_path("src/timetensor/sota/chronos2/weights")
        self.pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
            local_model_dir,
            device_map="cuda",
            local_files_only=True,
        )

    def predict_as_df(self, x, c=None): #x (bs, dim, lags)
        """returns prediction via the predict_df pipeline"""
        bs, dim, lags = x.shape
        x_reordered = x.permute(0, 2, 1) # (bs, lags, dim)
        x_arr = x_reordered.reshape(-1, dim).detach().cpu().numpy() # (bs*lags, dim)
        ids = np.repeat(np.arange(bs), lags)
        timestamps = np.tile(np.arange(lags), bs)

        dim_cols = [f"dim_{i}" for i in range(dim)]
        df = pd.DataFrame(x_arr, columns=dim_cols)
        df.insert(0, "timestamps", timestamps)
        df.insert(0, "ids", ids)

        preds_per_dim = []
        for dim_ in range(dim):
            preds = self.pipeline.predict_df(df,
                prediction_length=self.horizon,
                id_column="ids",
                quantile_levels=[0.5],
                timestamp_column="timestamps",
                target=f"dim_{dim_}")
            
            y_hat = (
                preds
                .pivot(index="ids", columns="timestamps", values="predictions")
                .sort_index()
                .to_numpy()          # shape (bs, horizon)
            )
            preds_per_dim.append(y_hat)

        preds_np = np.stack(preds_per_dim, axis=1)      # (bs, dim, horizon)
        preds = torch.from_numpy(preds_np).to(x.device, dtype=x.dtype)
        return preds


    def __call__(self, x, c=None): #x (bs, dim, lags)
        inputs = {"target": x}
        if c is not None:
            inputs["past_covariates"] = c[:, :, :x.shape[-1]]
            if c.shape[-1] > x.shape[-1]: #TODO gerer le cas où certains sont past-only et d'autres non
                inputs["future_covariates"] = c[:, :, x.shape[-1]:]

        quantile_preds = self.pipeline.predict(
            inputs=inputs,
            prediction_length=self.horizon,
            cross_learning=self.cross_learning,
            quantile_levels=[0.5],
        )

        preds = []
        for pred in quantile_preds:
            preds.append(pred[:, 0, :])
        return preds
    
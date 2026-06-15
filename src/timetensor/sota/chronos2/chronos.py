## Adapted from https://github.com/amazon-science/chronos-forecasting

import torch

from chronos import BaseChronosPipeline, Chronos2Pipeline
from hydra.utils import to_absolute_path


class Chronos:
    def __init__(self, lags, horizon, context_mode="past_only", cross_learning=False,
        device_map="cuda", weights_path="src/timetensor/sota/chronos2/weights",
        shared_context=False, frozen=True, **kwargs
        ):
        """
        shared_context: if True, use same context for all samples
        """
        super(Chronos, self).__init__()
        self.lags, self.horizon = lags, horizon
        self.context_mode = context_mode  # "past_only" | "future" | "any"
        self.cross_learning = bool(cross_learning)
        self.shared_context = shared_context

        local_model_dir = to_absolute_path(weights_path)
        self.pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
            local_model_dir,
            device_map=device_map,
            local_files_only=True,
        )

        if frozen:
            model = getattr(self.pipeline, "model", None)
            if model is not None:
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False

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
            assert isinstance(c,tuple) and len(c) == 2, f"Wrong context shape: {type(c)}"
            return c
        
        raise ValueError(f"Unknown context_mode={self.context_mode}")
    
    def _prepare_inputs(self, x, past_only, future_included):
        """Constructs inputs as a list of dictionaries:
        target: (dim, lags)
        past_covariates: (dim_c (*bs), lags)
        future_covariates: (dim_c (*bs), horizon)

        shared_context: if True, use same context for all samples (c has different batch size than x)
        """
        bs, dim, lags = x.shape
        inputs = []
        for b in range(bs):
            d = {"target": x[b].cpu()}  # (dim, lags)

            past_cov = {}
            fut_cov = {}

            if past_only is not None:
                cs, dim_c, _ = past_only.shape
                if not self.shared_context and bs == cs:
                    for i in range(dim_c):
                        past_cov[f"past_{i}"] = past_only[b, i, :].cpu()
                else:
                    for b_ in range(cs):
                        for i in range(dim_c):
                            past_cov[f"past_{b_}_{i}"] = past_only[b_, i, :].cpu()

            if future_included is not None:
                cs, dim_c, _ = future_included.shape
                if not self.shared_context and bs == cs:
                    for i in range(dim_c):
                        past_cov[f"fut_{i}"] = future_included[b, i, :lags].cpu()
                        fut_cov[f"fut_{i}"] = future_included[b, i, lags : lags + self.horizon].cpu()
                else:
                    for b_ in range(cs):
                        for i in range(dim_c):
                            past_cov[f"fut_{b_}_{i}"] = future_included[b_, i, :lags].cpu()
                            fut_cov[f"fut_{b_}_{i}"] = future_included[b_, i, lags : lags + self.horizon].cpu()

            if len(past_cov) > 0:
                d["past_covariates"] = past_cov
            if len(fut_cov) > 0:
                d["future_covariates"] = fut_cov
            inputs.append(d)
        return inputs

    def __call__(self, x, c=None): #x (bs, dim, lags)
        assert x.shape[-1] == self.lags
        
        past_only, future_included = self._split_context(c)
        inputs = self._prepare_inputs(x, past_only, future_included)

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
    

    @torch.no_grad()
    def representation(self, x, c=None, pool=False):
        """Returns Chronos-2 encoder representations."""
        assert x.shape[-1] == self.lags

        past_only, future_included = self._split_context(c)
        inputs = self._prepare_inputs(x, past_only, future_included)

        assert hasattr(self.pipeline, "embed")

        emb, _ = self.pipeline.embed(
            inputs=inputs #,cross_learning=self.cross_learning, context?
        ) #list of L//16 patches, embedded as vectors of size 768
        raw_emb = torch.stack(emb, dim=0).to(device=x.device)
        
        if pool:
            final_emb = raw_emb.mean(dim=1).mean(dim=1) #dim = hidden_dim
        else:
            final_emb = raw_emb.flatten(start_dim=1)
        return final_emb
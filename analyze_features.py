"""Analyze merged train/eval neighbor features with per-user loss scatter plots."""

import hydra
import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

from src.timetensor.pipeline import Loss
from src.timetensor.utils import get_dirs

warnings.simplefilter(action="ignore", category=FutureWarning)


def symlog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def reduce_user(x):
    """Average all dimensions except user dim, axis=1."""
    x = to_numpy(x).astype(float)
    if x.ndim < 2:
        raise ValueError(f"Expected user dimension at axis=1, got shape {x.shape}")
    axes = tuple(i for i in range(x.ndim) if i != 1)
    return np.nanmean(x, axis=axes)


def concat_payloads(payloads):
    merged = {}

    tensor_keys = sorted(
        {
            key
            for payload in payloads.values()
            for key, value in payload.items()
            if torch.is_tensor(value)
        }
    )

    base_names = sorted(
        {
            key.split("_", 1)[1]
            for key in tensor_keys
            if key.startswith("train_") or key.startswith("eval_")
        }
    )

    for base in base_names:
        values = []

        for split_name in ["train", "eval"]:
            if split_name not in payloads:
                continue

            payload = payloads[split_name]
            key = f"{split_name}_{base}"

            if key in payload and torch.is_tensor(payload[key]):
                values.append(payload[key])

        if values:
            merged[base] = torch.cat(values, dim=0)

    for key in ["lags", "horizon", "individuals", "neighbors"]:
        for payload in payloads.values():
            if key in payload:
                merged[key] = payload[key]
                break

    return merged

def compute_features(payload):
    loss_fn = Loss(nn.MSELoss(reduction="none"))

    y = payload["Y_values"]
    yc = payload["Yc_values"]
    pred = payload["preds"]
    pred_c = payload["preds_context"]
    pred_xc = payload["preds_xc"]

    mu_x = payload["mu_x"]
    mu_xc = payload["mu_xc"]
    sigma_x = payload["sigma_x"]
    sigma_xc = payload["sigma_xc"]
    d = payload["d_x_xc"]

    L = loss_fn(pred, y)
    Lc = loss_fn(pred_c, y)
    Lxc = loss_fn(pred_xc, yc)
    Lpred_yc = loss_fn(pred[:, :, None, :], yc)

    L_oracle = torch.minimum(L, Lc)

    mu_diff = mu_x[:, :, None] - mu_xc
    sigma_diff = sigma_x[:, :, None] - sigma_xc

    df = pd.DataFrame(
        {
            "mu_diff": reduce_user(mu_diff),
            "sig_diff": reduce_user(sigma_diff),
            "L": reduce_user(L),
            "Lc": reduce_user(Lc),
            "Lxc": reduce_user(Lxc),
            "pred_yc": reduce_user(Lpred_yc),
            "d": reduce_user(d),
            "imp": reduce_user(L - Lc),
            "imp_oracle": reduce_user(L - L_oracle),
        }
    )

    return df.replace([np.inf, -np.inf], np.nan)


def plot_scatter(df, x_col, y_col, title, filename, plots_dir):
    plot_df = df[[x_col, y_col]].dropna().copy()
    if len(plot_df) == 0:
        return

    x_label = f"slog_{x_col}"
    y_label = f"slog_{y_col}"

    plot_df[x_label] = symlog(plot_df[x_col].values)
    plot_df[y_label] = symlog(plot_df[y_col].values)

    g = sns.jointplot(
        data=plot_df,
        x=x_label,
        y=y_label,
        kind="scatter",
    )

    g.figure.suptitle(title, fontsize=18)
    g.figure.tight_layout(rect=[0, 0, 1, 0.97])
    g.figure.savefig(plots_dir / filename)
    plt.close(g.figure)


def plot_all(df, plots_dir):
    plots = [
        ("mu_diff", "sig_diff", "sigma diff = f(mu diff)", "sig_vs_mu.pdf"),
        ("Lc", "L", "L = f(Lc)", "L_vs_Lc.pdf"),
        ("Lxc", "L", "L = f(Lxc)", "L_vs_Lxc.pdf"),
        ("d", "imp", "improvement = f(d)", "imp_vs_d.pdf"),
        ("pred_yc", "imp", "improvement = f(pred-yc loss)", "imp_vs_pred_yc.pdf"),
        ("Lxc", "imp", "improvement = f(Lxc)", "imp_vs_Lxc.pdf"),
        ("d", "imp_oracle", "oracle improvement = f(d)", "oracle_vs_d.pdf"),
        ("pred_yc", "imp_oracle", "oracle improvement = f(pred-yc loss)", "oracle_vs_pred_yc.pdf"),
        ("Lxc", "imp_oracle", "oracle improvement = f(Lxc)", "oracle_vs_Lxc.pdf"),
    ]

    for x_col, y_col, title, filename in plots:
        plot_scatter(df, x_col, y_col, title, filename, plots_dir)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    logger.info(" ")
    logger.info("===== Running merged feature-loss analysis script =====")

    model_name, norm_name = cfg.model.name, cfg.normalization.name
    if norm_name == "None":
        norm_name = None

    _, save_dir = get_dirs(
        cfg.misc.output_dir,
        cfg.misc.save_name,
        model_name,
        norm_name,
    )
    save_dir = Path(save_dir)

    plots_dir = save_dir / "merged_loss_feature_scatter_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    payloads = {}
    for split_name in ["train", "eval"]:
        path = save_dir / f"{split_name}_payload.pt"
        if not path.exists():
            logger.warning(f"Missing payload: {path}")
            continue
        logger.info(f"Loading payload: {path}")
        payloads[split_name] = torch.load(path, map_location="cpu")

    if not payloads:
        raise FileNotFoundError(f"No train/eval payloads found in {save_dir}")

    required_suffixes = [
        "Y_values",
        "Yc_values",
        "preds",
        "preds_context",
        "preds_xc",
        "mu_x",
        "mu_xc",
        "sigma_x",
        "sigma_xc",
        "d_x_xc",
    ]

    for split_name, payload in payloads.items():
        missing = [
            f"{split_name}_{suffix}"
            for suffix in required_suffixes
            if f"{split_name}_{suffix}" not in payload
        ]
        if missing:
            raise KeyError(
                f"Missing keys in {split_name}_payload.pt: {missing}. "
                f"Rerun extraction with preds_xc enabled."
            )

    logger.info("Merging train/eval payloads over date dimension")
    payload = concat_payloads(payloads)

    logger.info("Computing per-user merged features")
    df = compute_features(payload)

    csv_path = plots_dir / "merged_per_user_features.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved merged per-user features to: {csv_path}")

    logger.info("Plotting requested scatter plots")
    plot_all(df, plots_dir)

    logger.info(f"Saved plots to: {plots_dir}")
    logger.info("End of script")


if __name__ == "__main__":
    run()
"""
Train a LightGBM gate on T1 payload features and evaluate:
1) soft_mix = g * pred_context + (1 - g) * pred
2) hard_tau_0.5 = pred_context if g > 0.5 else pred
3) hard_tau_0.7 = pred_context if g > 0.7 else pred

Save metrics, plots, and gate outputs on both T1 and T2.
"""

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
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

from src.timetensor.pipeline import Loss
from src.timetensor.utils import get_dirs, save_results, set_seed, symlog

warnings.simplefilter(action="ignore", category=FutureWarning)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    logger.info("===== Running LightGBM gating script =====")

    model_name, norm_name = cfg.model.name, cfg.normalization.name
    if norm_name == "None":
        norm_name = None

    output_dir = cfg.misc.output_dir
    save_name = cfg.misc.save_name
    verbose = cfg.misc.verbose
    seed = cfg.misc.seed

    set_seed(seed)
    criterion = Loss(nn.MSELoss(reduction="none"))

    _, save_dir = get_dirs(output_dir, save_name, model_name, norm_name)
    save_dir = Path(save_dir)
    plots_dir = save_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    train_payload_path = save_dir / "train_payload.pt"
    eval_payload_path = save_dir / "eval_payload.pt"

    if not train_payload_path.exists():
        raise FileNotFoundError(f"Missing training payload: {train_payload_path}")
    if not eval_payload_path.exists():
        raise FileNotFoundError(f"Missing eval payload: {eval_payload_path}")

    lgbm_kwargs = {
        "objective": "binary",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": seed,
        "verbosity": -1,
    }

    feature_names = [
        "mu_x",
        "sigma_x",
        "avg_mu_xc",
        "avg_sigma_xc",
        "std_mu_xc",
        "std_sigma_xc",
        "avg_d_x_xc",
        "std_d_x_xc",
        "avg_d_predc_yc",
        "std_d_predc_yc",
        "avg_d_pred_yc",
        "std_d_pred_yc",
    ]

    def get_tensor(payload, prefix, name):
        key = f"{prefix}_{name}"
        if key not in payload:
            raise KeyError(f"Missing key '{key}' in payload.")
        value = payload[key]
        if not torch.is_tensor(value):
            raise TypeError(f"Key '{key}' is not a tensor.")
        return value.float()

    def compute_losses(preds, preds_context, y_values):
        loss = criterion(preds, y_values).mean(dim=-1)
        loss_context = criterion(preds_context, y_values).mean(dim=-1)
        return loss, loss_context

    def build_features_and_labels(payload, prefix):
        preds = get_tensor(payload, prefix, "preds")
        preds_context = get_tensor(payload, prefix, "preds_context")
        y_values = get_tensor(payload, prefix, "Y_values")
        y_c = get_tensor(payload, prefix, "Yc_values")
        mu_x = get_tensor(payload, prefix, "mu_x")
        sigma_x = get_tensor(payload, prefix, "sigma_x")
        mu_xc = get_tensor(payload, prefix, "mu_xc")
        sigma_xc = get_tensor(payload, prefix, "sigma_xc")
        d_x_xc = get_tensor(payload, prefix, "d_x_xc")

        if d_x_xc.shape[-1] == 0:
            raise ValueError("Context is disabled (neighbors=0). Gating requires neighbors > 0.")

        loss, loss_context = compute_losses(preds, preds_context, y_values)
        oracle = (loss_context < loss).float()

        d_predc_yc = ((preds_context.unsqueeze(2) - y_c) ** 2).mean(dim=-1)
        d_pred_yc = ((preds.unsqueeze(2) - y_c) ** 2).mean(dim=-1)

        avg_mu_xc = mu_xc.mean(dim=-1)
        avg_sigma_xc = sigma_xc.mean(dim=-1)
        std_mu_xc = mu_xc.std(dim=-1, unbiased=False)
        std_sigma_xc = sigma_xc.std(dim=-1, unbiased=False)

        features = torch.stack(
            [
                mu_x,
                sigma_x,
                avg_mu_xc,
                avg_sigma_xc,
                std_mu_xc,
                std_sigma_xc,
                d_x_xc.mean(dim=-1),
                d_x_xc.std(dim=-1, unbiased=False),
                d_predc_yc.mean(dim=-1),
                d_predc_yc.std(dim=-1, unbiased=False),
                d_pred_yc.mean(dim=-1),
                d_pred_yc.std(dim=-1, unbiased=False),
            ],
            dim=-1,
        )

        n_dates, n_individuals, n_features = features.shape
        features_np = features.reshape(-1, n_features).cpu().numpy().astype(np.float32)
        labels_np = oracle.reshape(-1).cpu().numpy().astype(np.int64)

        aux = {
            "preds": preds,
            "preds_context": preds_context,
            "y_values": y_values,
            "d_x_xc": d_x_xc,
            "loss": loss,
            "loss_context": loss_context,
            "oracle": oracle,
            "n_dates": n_dates,
            "n_individuals": n_individuals,
        }
        return features_np, labels_np, aux

    def summarize_strategy(final_pred, aux):
        y_values = aux["y_values"]
        d_x_xc = aux["d_x_xc"]
        base_loss = aux["loss"]
        n_individuals = aux["n_individuals"]

        final_loss = criterion(final_pred, y_values).mean(dim=-1)
        improvement = 100.0 * (base_loss - final_loss) / torch.clamp(base_loss, min=1e-8)
        improve_ratio = float(100.0 * (improvement > 0).float().mean().item())

        final_loss_np = final_loss.cpu().numpy()
        improvement_np = improvement.cpu().numpy()
        distance_np = d_x_xc.mean(dim=-1).cpu().numpy().reshape(-1)

        per_user_losses = []
        stds_per_user_losses = []
        per_user_improvements = []
        stds_per_user_improvements = []

        for indiv in range(n_individuals):
            indiv_loss = final_loss_np[:, indiv]
            per_user_losses.append(float(np.mean(indiv_loss)))
            stds_per_user_losses.append(float(np.std(indiv_loss)))

            indiv_improvement = improvement_np[:, indiv]
            per_user_improvements.append(float(np.mean(indiv_improvement)))
            stds_per_user_improvements.append(float(np.std(indiv_improvement)))

        per_user_losses = np.asarray(per_user_losses, dtype=np.float32)
        stds_per_user_losses = np.asarray(stds_per_user_losses, dtype=np.float32)
        per_user_improvements = np.asarray(per_user_improvements, dtype=np.float32)
        stds_per_user_improvements = np.asarray(stds_per_user_improvements, dtype=np.float32)

        nMSE = float(np.mean(per_user_losses))

        tail_count = max(1, int(np.ceil(0.1 * len(per_user_losses))))
        tail_start = len(per_user_losses) - tail_count
        w10_nMSE = float(np.mean(np.partition(per_user_losses, tail_start)[tail_start:]))

        return {
            "final_loss": final_loss,
            "improvement": improvement,
            "improve_ratio": improve_ratio,
            "nMSE": nMSE,
            "w10_nMSE": w10_nMSE,
            "per_user_losses": per_user_losses,
            "stds_per_user_losses": stds_per_user_losses,
            "per_user_improvements": per_user_improvements,
            "stds_per_user_improvements": stds_per_user_improvements,
            "distance_np": distance_np,
            "improvement_np": improvement_np,
        }

    def save_strategy_plots(split_name, save_prefix, strategy_name, summary):
        tag = strategy_name.replace(".", "p")

        error_df = pd.DataFrame(
            {
                "log(mean_error)": [symlog(x) for x in summary["per_user_losses"]],
                "log(std_error)": [symlog(x) for x in summary["stds_per_user_losses"]],
            }
        ).dropna()

        g = sns.jointplot(
            data=error_df,
            x="log(mean_error)",
            y="log(std_error)",
            kind="scatter",
        )
        g.figure.suptitle(
            f"Per-user nMSE of {save_name} on {split_name} [{strategy_name}] "
            f"(mean:{summary['nMSE']:.4f}, W10:{summary['w10_nMSE']:.4f})",
            fontsize=20,
        )
        g.figure.tight_layout(rect=[0, 0, 1, 0.98])
        g.figure.savefig(plots_dir / f"{save_prefix}_{tag}_user_errors.pdf")
        plt.close(g.figure)

        improvement_df = pd.DataFrame(
            {
                "log(mean_improvement)": [symlog(x) for x in summary["per_user_improvements"]],
                "log(std_improvement)": [symlog(x) for x in summary["stds_per_user_improvements"]],
            }
        ).dropna()

        g = sns.jointplot(
            data=improvement_df,
            x="log(mean_improvement)",
            y="log(std_improvement)",
            kind="scatter",
        )
        g.figure.suptitle(
            f"Per-user improvement of {save_name} on {split_name} [{strategy_name}] "
            f"(mean ratio:{summary['improve_ratio']:.4f})",
            fontsize=20,
        )
        g.figure.tight_layout(rect=[0, 0, 1, 0.98])
        g.figure.savefig(plots_dir / f"{save_prefix}_{tag}_user_improvements.pdf")
        plt.close(g.figure)

        plt.figure(figsize=(10, 7))
        plt.scatter(np.log(summary["distance_np"] + 1e-8), summary["improvement_np"].reshape(-1))
        plt.title(f"Improvements on {split_name} [{strategy_name}]")
        plt.xlabel("log(distance)")
        plt.ylabel("improvement")
        plt.tight_layout()
        plt.savefig(plots_dir / f"{save_prefix}_{tag}_improvements.pdf")
        plt.close()

    def evaluate_split(split_name, save_prefix, probs, aux):
        preds = aux["preds"]
        preds_context = aux["preds_context"]
        oracle = aux["oracle"]
        n_dates = aux["n_dates"]
        n_individuals = aux["n_individuals"]

        gate_probs = torch.from_numpy(probs.astype(np.float32)).reshape(n_dates, n_individuals)
        gate_probs_expanded = gate_probs.unsqueeze(-1)

        strategy_preds = {
            "soft_mix": gate_probs_expanded * preds_context + (1.0 - gate_probs_expanded) * preds,
            "hard_tau_0.5": torch.where(gate_probs_expanded > 0.5, preds_context, preds),
            "hard_tau_0.7": torch.where(gate_probs_expanded > 0.7, preds_context, preds),
        }

        oracle_np = oracle.cpu().numpy().reshape(-1)
        acc_05 = accuracy_score(oracle_np, (probs > 0.5).astype(np.int64))
        acc_07 = accuracy_score(oracle_np, (probs > 0.7).astype(np.int64))

        save_results(acc_05, output_dir, "mean_results.json", save_name, f"{split_name} gate acc tau=0.5")
        save_results(acc_07, output_dir, "mean_results.json", save_name, f"{split_name} gate acc tau=0.7")

        gate_payload = {
            f"{save_prefix}_g": gate_probs.cpu(),
            f"{save_prefix}_oracle": oracle.cpu(),
        }

        for strategy_name, final_pred in strategy_preds.items():
            summary = summarize_strategy(final_pred, aux)

            save_results(
                summary["improve_ratio"],
                output_dir,
                "mean_results.json",
                save_name,
                f"{split_name} improvements [{strategy_name}]",
            )
            save_results(
                summary["nMSE"],
                output_dir,
                "mean_results.json",
                save_name,
                f"{split_name} nMSE [{strategy_name}]",
            )
            save_results(
                summary["w10_nMSE"],
                output_dir,
                "mean_results.json",
                save_name,
                f"{split_name} w10 nMSE [{strategy_name}]",
            )

            save_strategy_plots(split_name, save_prefix, strategy_name, summary)

            gate_payload[f"{save_prefix}_{strategy_name}_final_pred"] = final_pred.cpu()
            gate_payload[f"{save_prefix}_{strategy_name}_final_loss"] = summary["final_loss"].cpu()
            gate_payload[f"{save_prefix}_{strategy_name}_improvement"] = summary["improvement"].cpu()

            logger.info(
                f"{split_name} [{strategy_name}]: "
                f"improve_ratio={summary['improve_ratio']:.4f}, "
                f"nMSE={summary['nMSE']:.4f}, "
                f"w10={summary['w10_nMSE']:.4f}"
            )

        torch.save(gate_payload, save_dir / f"{save_prefix}_gate_outputs.pt")

    train_payload = torch.load(train_payload_path, map_location="cpu")
    eval_payload = torch.load(eval_payload_path, map_location="cpu")

    if verbose:
        logger.info("Building train features and labels")
    train_X, train_y, train_aux = build_features_and_labels(train_payload, "train")

    if verbose:
        logger.info("Building eval features and labels")
    eval_X, eval_y, eval_aux = build_features_and_labels(eval_payload, "eval")

    logger.info(f"Train dataset: X={train_X.shape}, y={train_y.shape}")
    logger.info(f"Eval dataset: X={eval_X.shape}, y={eval_y.shape}")
    logger.info(f"Features: {feature_names}")

    gate = LGBMClassifier(**lgbm_kwargs)
    gate.fit(train_X, train_y)

    train_probs = gate.predict_proba(train_X)[:, 1]
    eval_probs = gate.predict_proba(eval_X)[:, 1]

    feat_imp = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": gate.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    feat_imp.to_csv(save_dir / "lightgbm_feature_importance.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.barh(feat_imp["feature"], feat_imp["importance"])
    plt.gca().invert_yaxis()
    plt.title("LightGBM feature importance")
    plt.tight_layout()
    plt.savefig(plots_dir / "lightgbm_feature_importance.pdf")
    plt.close()

    evaluate_split(
        split_name="train",
        save_prefix="train",
        probs=train_probs,
        aux=train_aux,
    )

    evaluate_split(
        split_name="eval",
        save_prefix="eval",
        probs=eval_probs,
        aux=eval_aux,
    )

    logger.info("End of script")


if __name__ == "__main__":
    run()
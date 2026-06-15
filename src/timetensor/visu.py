import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from tabulate import tabulate
import os
import torch
import seaborn as sns

from .dataset import fetch_example_data

## series plots

def plot_serie(x, path="", name="series.pdf", title="Time series", axis=True, show=False):
    """plots example serie"""
    fig = plt.figure(figsize=(20,5))
    plt.plot(range(len(x)), x)
    if not axis:
      plt.axis('off')
      plt.title(None)
    plt.title(title)
    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(path+name)
    plt.close()

def plot_series(x, path="", name="series.pdf", title="Multiple series", axis=True, show=False):
    """plots multiple series"""
    fig = plt.figure(figsize=(20,5))
    for key, serie in x.items():
        plt.plot(range(len(serie)), serie, label=f"{key}")
    plt.legend(bbox_to_anchor=(0.5, -0.15), ncol=3, loc='center', fontsize=14)
    if not axis:
      plt.axis('off')
      plt.title(None)
    plt.title(title)
    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(path+name)
    plt.close()

def plot_example(x, y, path="", name="example.pdf", title="Example", axis=True, show=False):
    """plots example input output"""
    lag = len(x)
    horizon = len(y)
    fig = plt.figure(figsize=(20,5))
    plt.plot(range(lag+1), x+[y[0]], label="Lookback")
    plt.plot(range(lag, lag+horizon), y, label="Horizon")
    plt.axvline(x=lag, color='black', linestyle='--')
    plt.legend(bbox_to_anchor=(0.5, -0.15), ncol=3, loc='center', fontsize=14)
    if not axis:
      plt.axis('off')
      plt.title(None)
    plt.title(title)
    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(path+name)
    plt.close()

def plot_named_example(path, name):
    x, c, y, i, d  = fetch_example_data(path, name)
    plot_example(x[0].cpu().detach().tolist(), y[0].cpu().detach().tolist(), path + f"/{name}/", f"example.pdf", f"Example window (user {i} date {d})")


def plot_pred(x, y, pred, path="", name="prediction.pdf", title="Predictions", axis=True, show=False):
    """plots example prediction"""
    lag = len(x)
    horizon = len(y)
    fig = plt.figure(figsize=(20,5))
    plt.plot(range(lag+1), x+[pred[0]], label="Lookback")
    plt.plot(range(lag, lag+horizon), pred, label="Prediction")
    plt.plot(range(lag, lag+horizon), y, label="Horizon")
    plt.axvline(x=lag, color='black', linestyle='--')
    plt.legend(bbox_to_anchor=(0.5, -0.15), ncol=3, loc='center', fontsize=14)
    if not axis:
      plt.axis('off')
      plt.title(None)
    plt.title(title)
    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(path+name)
    plt.close()


def plot_preds(x, y, preds, path="", name="prediction.pdf", title="Predictions", axis=True, show=False):
    """plots multiple example predictions"""
    lag = len(x)
    horizon = len(y)
    fig = plt.figure(figsize=(20,5))
    plt.plot(range(lag+1), x+[y[0]], label="Lookback")
    for key, pred in preds.items():
        plt.plot(range(lag, lag+horizon), pred, label=f"{key}")
    plt.plot(range(lag, lag+horizon), y, "--", label="Horizon")
    plt.axvline(x=lag, color='black', linestyle='--')
    plt.legend(bbox_to_anchor=(0.5, -0.15), ncol=3, loc='center', fontsize=14)
    if not axis:
      plt.axis('off')
      plt.title(None)
    plt.title(title)
    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(path+name)
    plt.close()


## 2D plots

def plot_2D(matrix, path, name="weights.pdf", title='Model weights', x_name="x", y_name="y"):
    """plots weights of a model"""
    plt.figure()
    plt.imshow(matrix, aspect='auto', cmap='viridis')
    plt.colorbar(label='Weight value')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)
    plt.savefig(path + name)
    plt.close()

def plot_weights_(weights, path, name="weights.pdf", title='Model weights'):
    """plots weights of a model"""
    plt.figure()
    plt.imshow(weights, aspect='auto', cmap='viridis')
    plt.colorbar(label='Weight value')
    plt.xlabel('Inputs (lookback)')
    plt.ylabel('Outputs (horizon)')
    plt.title(title)
    plt.savefig(path + name)
    plt.close()


## losses plots

def plot_losses(train_losses, valid_losses_dict=None, path="", name="losses.pdf", title="Losses", logscale=True, eval_freq=10, show=False):
    """plots training loss (and valids) during training"""
    fig = plt.figure(figsize=(10,5))
    if valid_losses_dict is not None:
        plt.plot(range(1, len(train_losses)+1), train_losses, label="train")
        for key, values in valid_losses_dict.items():
            T = [1]
            if len(values)>1:
                T += [k*eval_freq for k in range(1, len(values)-1)] + [len(train_losses)]
            plt.plot(T, values, label=key)
        plt.legend()
    else:
        plt.plot(range(1, len(train_losses)+1), train_losses)
    if logscale:
      plt.yscale('log')
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title(title)
    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(path+name)
    plt.close()

def plot_multi_losses(losses_dict, path="", name="losses.pdf", title="Losses", logscale=True, x_every=None, eval_freq=1, show=False):
    """plots multiple losses during training"""
    fig = plt.figure(figsize=(10,5))
    for expe_name, losses in losses_dict.items():
        T = [1] + [k*eval_freq for k in range(1,len(losses))]
        plt.plot(T, losses, label=f"{expe_name}")
    if x_every is not None:
        for k in range(1, (len(losses)+1)//x_every):
            plt.axvline(k*x_every, linestyle="--", color="red")
    if logscale:
      plt.yscale('log')
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(path+name)
    plt.close()

def plot_errors(losses, path="", name="errors.pdf", title="Loss distribution", show=False):
    """plots histogram of errors"""
    fig = plt.figure(figsize=(10,5))
    # plt.hist(losses, bins=100, density=True)
    sns.kdeplot(losses, log_scale=True)
    # plt.xscale("log")
    plt.title(title)
    plt.xlabel("Losses")
    plt.ylabel("Frequency")
    if show:
        plt.show()
    else:
        plt.savefig(path+name)
    plt.close()

def plot_horizon_errors(losses, path="", name="horizon.pdf", title="Mean errors by horizon", show=False):
    """plots errors according to horizon"""
    fig = plt.figure(figsize=(15,5))
    plt.bar(range(len(losses)), losses)
    plt.title(title)
    plt.xlabel("Horizon")
    plt.ylabel("Mean error")
    if show:
        plt.show()
    else:
        plt.savefig(path+name)
    plt.close()



## results

def get_errors_df(dir_name, file_name, multipliers=None, names=None, save=False):
    """formats errors json at path"""
    with open(dir_name+file_name) as file:
        data = json.load(file)
    df = pd.DataFrame(data)
    if names=="None":
        names=None
    if names is not None:
        if type(names)==str:
            names=names.split(";")
        df = df[names]
    if multipliers is not None:
        if type(multipliers) == str:
            multipliers = multipliers.split(" ")
            multipliers = [int(w) for w in multipliers]
        new_index = list(df.index)
        for k in range(min(len(multipliers), df.shape[0])):
            if multipliers[k] != 0:
                df.iloc[k] = df.iloc[k] * 10**multipliers[k]
                new_index[k] = new_index[k] + f" * 1e{multipliers[k]}"
        df.index = new_index
    if save:
        df.to_csv(dir_name + 'errors.csv')
    return df

def get_expe_results(dir_name, file_name, multipliers=None, names=None, print_table=True, save_path=None, save_name="errors.pdf"):
    """prints table of errors for one seed"""
    df = get_errors_df(dir_name, file_name, multipliers, names)
    if print_table:
        table = tabulate(df, headers='keys', tablefmt='grid', showindex=True, floatfmt=".4f")
        print(f"==Table of {dir_name}==")
        print(table)

    plt.figure(figsize=(10,5))
    plt.grid()
    plt.scatter(list(df.columns), df.iloc[0].values, s=100)
    plt.xticks(rotation = 45)
    plt.title("Experiment results")
    plt.tight_layout()

    if save_path is None:
        save_path = dir_name+"plots/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(save_path + save_name)
    plt.close()


def get_multiple_errors_df(dir_name, file_name, n_paths, multipliers=None, names=None, baseline=None, save=False, percents=False):
    """formats errors json from multipled seeds in dir_name"""
    paths = [dir_name + f"seed_{k}/" + file_name for k in range(1,n_paths+1)]
    dfs = []
    for path in paths:
        with open(path) as file:
            data = json.load(file)
        df = pd.DataFrame(data)
        if names=="None":
            names=None
        if names is not None:
            if type(names)==str:
                names=names.split(";")
            df = df[names]

        if baseline is not None and baseline in df.columns:
            baseline_vals = df[baseline].copy()
            df = df.subtract(baseline_vals, axis=0)
            if percents:
                df = 100 * df.divide(baseline_vals, axis=0)
        dfs.append(df)

    df_mean = pd.concat(dfs).groupby(level=0).mean()
    df_std = pd.concat(dfs).groupby(level=0).std()

    if multipliers is not None:
        if type(multipliers) == str:
            multipliers = multipliers.split(" ")
            multipliers = [int(w) for w in multipliers]
        new_index = list(df_mean.index)
        for k in range(min(len(multipliers), df_mean.shape[0])):
            if multipliers[k] != 0:
                df_mean.iloc[k] = df_mean.iloc[k] * 10**multipliers[k]
                df_std.iloc[k] = df_std.iloc[k] * 10**multipliers[k]
                new_index[k] = new_index[k] + f" * 1e{multipliers[k]}"
        df_mean.index = new_index
        df_std.index = new_index

    if save:
        df_mean.to_csv(dir_name + 'mean_errors.csv')
        df_std.to_csv(dir_name + 'std_errors.csv')
    return df_mean, df_std

def get_multiple_expe_results(dir_name, file_name, n_paths, multipliers=None, names=None, show_std=True, baseline=None, print_table=True, show_row=0, save_path=None, save_name="errors.df"):
    """prints results of multiple experiments"""
    df_mean, df_std = get_multiple_errors_df(dir_name, file_name, n_paths, multipliers, names, baseline)

    if show_std:
        df_formatted = df_mean.copy()
        for col in df_mean.columns:
            df_formatted[col] = df_mean[col].map("{:.4f}".format) + " ± " + df_std[col].map("{:.4f}".format)
    else:
        df_formatted = df_mean.applymap("{:.4f}".format)

    if save_path is None:
        save_path = dir_name + "plots/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if print_table:
        table = tabulate(df_formatted, headers='keys', tablefmt='grid', showindex=True)
        print(f"==Table of {dir_name}==")
        print(table)

    plt.figure(figsize=(10,5))
    plt.grid()
    plt.scatter(list(df_mean.columns), df_mean.iloc[show_row].values, s=100)
    plt.xticks(rotation = 45)
    plt.title("Experiment results")
    plt.tight_layout()
    plt.savefig(save_path + save_name)
    plt.close()


def get_boxplots(dir_name, file_name, n_paths, col="Test MSE", save_path=None, save_name="boxplot.pdf", names=None, baseline=None):
    """print table from dataframe in path"""
    paths = [dir_name + f"seed_{k}/" + file_name for k in range(1,n_paths+1)]
    
    box_df = []
    for k, path in enumerate(paths):
        with open(path) as file:
            data = json.load(file)
        df = pd.DataFrame(data)
        if names=="None":
            names=None            
        if names is not None:
            if type(names)==str:
                names=names.split(";")
            df = df[names]
        if baseline is not None:
            assert (baseline in df.columns)
            df = df.subtract(df[baseline], axis=0)
        df = df.loc[col]
        for algo, value in df.items():
            box_df.append({"Algorithm": algo, f"{col}": value, "seed":k})

    box_df = pd.DataFrame(box_df)
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=box_df, x='Algorithm', y=col)#, hue="seed")
    plt.title(f"Experiment results")
    plt.xlabel("Experiment")
    plt.ylabel(f"{col}")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    if save_path is None:
        save_path = dir_name+"plots/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(save_path + save_name)
    plt.close()



## scripts

def plot_weights(model, save_dir, save_name):
    """plotting weights scripts"""
    model_name = model.model_name
    if model_name in ["linear", "sklinear"]:
        if model_name == "sklinear":
            weights = model.reg.coef_
        else:
            weights = model.fc.weight.detach().cpu().numpy()
        plot_weights_(weights, save_dir + "plots/", title=f'{save_name} weights')
        
    elif model_name == "DLinear":
        linear_weights = model.Linear_Seasonal[0].weight.detach().cpu().numpy()
        season_weights = model.Linear_Trend[0].weight.detach().cpu().numpy()
        plot_weights_(linear_weights, save_dir + "plots/", name="season_weights.pdf", title=f'{save_name} seasonal weights')
        plot_weights_(season_weights, save_dir + "plots/", name="trend_weights.pdf", title=f'{save_name} trend weights')
    

def plot_expe(losses_path, eval_freq=10, names=None, save_path=None, lr=None, bs=None, epochs=None):
    """plots losses for list of experiments in path"""
    if type(eval_freq)==str:
        eval_freq=int(eval_freq)
    if names=="None":
        names=None
    if names is not None:
        if type(names)==str:
            names=names.split(";")
    if save_path is None:
        save_path = losses_path+"plots/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    expe_names = [name for name in os.listdir(losses_path) if (names is None and os.path.exists(losses_path + f"{name}/" + "valid_losses1.pt")) or (names is not None and name in names)]

    title_sfx = ""
    if lr is not None:
        title_sfx += f",lr={lr}"
    if bs is not None:
        title_sfx += f",bs={bs}"
    if epochs is not None:
        title_sfx += f",e={epochs}"

    if len(expe_names) >0:
        losses_dict1 = {}
        losses_dict2 = {}
        losses_dict3 = {}

        for expe_name in expe_names:
            valid_losses1 = torch.load(losses_path + expe_name + "/" + "valid_losses1.pt", weights_only=False)
            valid_losses2 = torch.load(losses_path + expe_name + "/" + "valid_losses2.pt", weights_only=False)
            valid_losses3 = torch.load(losses_path + expe_name + "/" + "valid_losses3.pt", weights_only=False)

            for loss_name in valid_losses1:
                if loss_name not in losses_dict1:
                    losses_dict1[loss_name] = {}
                    losses_dict2[loss_name] = {}
                    losses_dict3[loss_name] = {}
                losses_dict1[loss_name][expe_name] = valid_losses1[loss_name]
                losses_dict2[loss_name][expe_name] = valid_losses2[loss_name]
                losses_dict3[loss_name][expe_name] = valid_losses3[loss_name]

        for loss_name in valid_losses1:
            plot_multi_losses(losses_dict1[loss_name], save_path, f"{loss_name}_valid1.pdf", f"Valid {loss_name}" + title_sfx, eval_freq=eval_freq)
            plot_multi_losses(losses_dict2[loss_name], save_path, f"{loss_name}_valid2.pdf", f"Valid2 {loss_name}" + title_sfx, eval_freq=eval_freq)
            plot_multi_losses(losses_dict3[loss_name], save_path, f"{loss_name}_valid3.pdf", f"Valid3 {loss_name}"+ title_sfx, eval_freq=eval_freq)



## Latex


# def latex_formated_number(value, decimals=3, color=False, row=None, std=None):
#     """return formated string value"""
#     if value is None:
#         return "--"

#     if std is not None:
#         fmt = f"{{:.{decimals}f}}" + " ± " + f"{std:.2f}"
#     else:
#         fmt = f"{{:.{decimals}f}}"
#     s = fmt.format(value)

#     if row is not None:
#         m = min(row)
#         if value == m:
#             s = r"\textbf{" + s + "}"
#     if color:
#         if value > 0:
#             return r"{\color{red}" + s + "}"
#         elif value < 0:
#             return r"{\color{green}" + s + "}"
#     return s

# def build_results_table_latex(
#     save_dir, datasets, settings, show_row=0, models="RevIN", file_name="test1_mean_results.json", n_paths=1, multipliers=None, baseline=None, title="1e5 * MSE", save_name="test1_mean_results.tex", color=False, decimals=2, show_std=False, n_settings=4):
#     """
#     Returns a LaTeX tabular string
#     Directory layout assumed: {save_dir}/{dataset}/lags{L}_horizon{H}/
#     """
#     datasets = text_list(datasets)
#     settings = text_list(settings) #of size datasets * (settings per dataset)
#     norm_settings = []
#     for s in settings:
#         _s = s.split("-")
#         L, H = int(_s[0]), int(_s[1])
#         norm_settings.append((L, H))

#     n_paths = text_list(n_paths)
#     n_paths = [int(text) for text in n_paths]
#     if len(n_paths) == 1 and len(settings)>1:
#         n_paths = [n_paths[0] for _ in range(len(settings))]
#     models = text_list(models)

#     # Collect values
#     values = {}
#     values_percent = {}
#     values_std = {}
#     multipliers = multipliers.split(";")
#     for i, (L, H) in enumerate(norm_settings):
#         for model in models:
#             ds = datasets[i // n_settings]
#             dir_name = save_dir + f"{ds}/lags{L}_horizon{H}/"
#             df, df_std = get_multiple_errors_df(
#                     dir_name=dir_name,
#                     file_name=file_name,
#                     n_paths=n_paths[i],
#                     multipliers=multipliers[i],
#                     baseline=None
#                 )

#             key = f"{ds}_{L}_{H}"
#             if key not in values:
#                 values[key] = []
#                 values_percent[key] = []
#                 values_std[key] = []
#             try:
#                 values[key].append(df.iloc[show_row][model])
#             except:
#                 raise ValueError(f"{i} {ds} {L} {H} {df.iloc[show_row]}")
#             if show_std:
#                 values_std[key].append(df_std.iloc[show_row][model])
#             if baseline is not None:
#                 df, _ = get_multiple_errors_df(
#                         dir_name=dir_name,
#                         file_name=file_name,
#                         n_paths=n_paths[i],
#                         multipliers=None,
#                         baseline=baseline,
#                         percents=True
#                     )
#                 values_percent[key].append(df.iloc[show_row][model])
    
#     lines = []
#     colspec = "l" + "c" + "c" * len(models)
#     lines.append(f"\\begin{{tabular}}{{{colspec}}}")
#     lines.append("\\toprule")
#     # lines.append(title + " & " + " & ".join(pretty_headers) + r" \\")
#     lines.append(title + " & " + "L-H" + " & " + " & ".join([model.replace("_", r"\_") for model in models]) + r" \\")
#     lines.append("\\midrule")

#     # for ds in datasets:
#     for i, (L, H) in enumerate(norm_settings):
#         ds = datasets[i // n_settings]
#         key = f"{ds}_{L}_{H}"
#         ds_latex = ds.replace("_", r"\_").capitalize()
#         if show_std:
#             std = values_std[key][i]
#         else:
#             std = None
#         cells = [latex_formated_number(v, decimals=decimals, color=color, row=values[key], std=std) for i, v in enumerate(values[key])]
#         if i % n_settings == 0: #TODO: below instead of n_settings, it should depend on the dataset...
#             lines.append("\\multirow{" + str(n_settings) + "}{*}{" + ds_latex + "}" + " & " + f"{L}-{H}" + " & " + " & ".join(cells) + r" \\")
#         else:
#             lines.append(" & " + f"{L}-{H}" + " & " + " & ".join(cells) + r" \\")
#         if i % n_settings == n_settings - 1:
#             lines.append("\\midrule")

#     if baseline is not None:
#         lines.append("\\midrule")
#         values_percent = pd.DataFrame.from_dict(values_percent, orient="index")
#         values_percent.columns = models
#         means = values_percent.mean(axis=0).values
#         lines.append("Improvements" + " & " + " & " + " & ".join([str(round(mean,2)) + r" \% " for mean in means]) + r" \\")

#     lines.append("\\bottomrule")
#     lines.append("\\end{tabular}")
#     latex = "\n".join(lines)
    
#     with open(save_dir + save_name, "w", encoding="utf-8") as f:
#         f.write(latex)


# def generate_results_table(
#     experiment_dir: str,
#     dataset_names: list = None,  # Default to None for auto-detection
#     settings: any = None,        # Default to None for auto-detection
#     json_filename: str = "results.json",
#     model_names: any = None,     # Default to None for auto-detection
#     metric_key: str = "nMSE",    # Default to nMSE
#     output_tex_path: str = None,
#     lower_is_better: bool = True,
#     decimals: int = 4,
#     baseline_idx: int = 0,
#     multiplier: any = 1,
# ):
#     """Generates a LaTeX table from JSON result files. Auto-detects structure (datasets, settings, models) from the file system."""
#     if type(model_names) == str:
#         model_names = model_names.split(";")
#     if type(settings) == str:
#         settings = settings.split(";")

#     # --- 0. Path Setup ---
#     if output_tex_path is None:
#         output_tex_path = os.path.join(experiment_dir, "results.tex")

#     # --- 1. Auto-Detection of Structure ---
#     # A. Detect Datasets
#     if dataset_names is None:
#         if os.path.exists(experiment_dir):
#             dataset_names = sorted(
#                 [
#                     d
#                     for d in os.listdir(experiment_dir)
#                     if os.path.isdir(os.path.join(experiment_dir, d)) and not d.startswith(".")
#                 ]
#             )
#             print(f"Auto-detected datasets: {dataset_names}")
#         else:
#             print(f"Error: Experiment directory '{experiment_dir}' does not exist.")
#             return

#     # B. Detect Settings
#     if settings is None:
#         detected_settings = {}
#         for ds in dataset_names:
#             ds_path = os.path.join(experiment_dir, ds)
#             if os.path.exists(ds_path):
#                 subs = sorted(
#                     [
#                         s
#                         for s in os.listdir(ds_path)
#                         if os.path.isdir(os.path.join(ds_path, s)) and not s.startswith(".")
#                     ]
#                 )
#                 detected_settings[ds] = subs
#             else:
#                 detected_settings[ds] = []
#         settings = detected_settings

#     # C. Detect Models (From Folders)
#     if model_names is None or len(model_names) == 0:
#         found_models = set()
#         for ds in dataset_names:
#             current_settings = settings.get(ds, []) if isinstance(settings, dict) else settings
#             for setting in current_settings:
#                 setting_path = os.path.join(experiment_dir, ds, setting)
#                 if os.path.exists(setting_path):
#                     subdirs = [
#                         d
#                         for d in os.listdir(setting_path)
#                         if os.path.isdir(os.path.join(setting_path, d)) and not d.startswith(".")
#                     ]
#                     found_models.update(subdirs)
#         if found_models:
#             model_names = sorted(list(found_models))
#             print(f"Auto-detected models from folders: {model_names}")
#         else:
#             print("Warning: No model folders found. Will attempt to detect from JSON keys.")
#             model_names = []

#     # --- 2. Data Loading ---
#     table_data = {}
#     auto_detect_from_json = len(model_names) == 0
#     for dataset in dataset_names:
#         table_data[dataset] = []
#         current_settings = settings.get(dataset, []) if isinstance(settings, dict) else settings
#         current_mult = multiplier.get(dataset, 1) if isinstance(multiplier, dict) else multiplier

#         for setting in current_settings:
#             file_path = os.path.join(experiment_dir, dataset, setting, json_filename)
#             row_values = []
#             if os.path.exists(file_path):
#                 try:
#                     with open(file_path, "r") as f:
#                         data = json.load(f)
#                     if auto_detect_from_json:
#                         model_names = sorted(list(data.keys()))
#                         auto_detect_from_json = False
#                         print(f"Auto-detected models from JSON keys: {model_names}")
#                     if model_names:
#                         for model in model_names:
#                             val = data.get(model, {}).get(metric_key, float("nan"))
#                             if not np.isnan(val):
#                                 val = val * current_mult
#                             row_values.append(val)
#                     else:
#                         row_values = []
#                 except Exception as e:
#                     print(f"Error reading {file_path}: {e}")
#                     row_values = []
#             else:
#                 row_values = []
#             table_data[dataset].append((setting, row_values))

#     if not model_names:
#         print("Error: Could not detect model names from folders or JSON files.")
#         return

#     # Post-processing: Fill missing file rows with NaNs
#     num_models = len(model_names)
#     for ds in dataset_names:
#         new_rows = []
#         for setting, vals in table_data[ds]:
#             if len(vals) != num_models:
#                 new_rows.append((setting, [float("nan")] * num_models))
#             else:
#                 new_rows.append((setting, vals))
#         table_data[ds] = new_rows

#     # --- 3. Compute Improvements ---
#     # A) Overall improvements
#     improvement_percentages = []
#     flat_rows = []
#     for ds in dataset_names:
#         for _, values in table_data[ds]:
#             flat_rows.append(values)

#     for col_idx in range(len(model_names)):
#         if col_idx == baseline_idx:
#             improvement_percentages.append(0.0)
#             continue
#         rel_diffs = []
#         for row in flat_rows:
#             baseline_val = row[baseline_idx]
#             current_val = row[col_idx]
#             if np.isnan(baseline_val) or np.isnan(current_val) or baseline_val == 0:
#                 continue
#             if lower_is_better:
#                 diff = (baseline_val - current_val) / baseline_val
#             else:
#                 diff = (current_val - baseline_val) / baseline_val
#             rel_diffs.append(diff)
#         improvement_percentages.append(np.mean(rel_diffs) * 100 if rel_diffs else 0.0)

#     # Per-dataset improvements (avg over all settings for each dataset)
#     per_dataset_improvements = {}
#     for ds in dataset_names:
#         rows = [values for _, values in table_data[ds]]
#         imps = []
#         for col_idx in range(len(model_names)):
#             if col_idx == baseline_idx:
#                 imps.append(0.0)
#                 continue
#             rel_diffs = []
#             for row in rows:
#                 baseline_val = row[baseline_idx]
#                 current_val = row[col_idx]
#                 if np.isnan(baseline_val) or np.isnan(current_val) or baseline_val == 0:
#                     continue
#                 if lower_is_better:
#                     diff = (baseline_val - current_val) / baseline_val
#                 else:
#                     diff = (current_val - baseline_val) / baseline_val
#                 rel_diffs.append(diff)
#             imps.append(np.mean(rel_diffs) * 100 if rel_diffs else 0.0)
#         per_dataset_improvements[ds] = imps

#     # Per-setting improvements: for each distinct setting, average over all datasets
#     setting_to_rows = {}
#     for ds in dataset_names:
#         for setting, values in table_data[ds]:
#             setting_to_rows.setdefault(setting, []).append(values)

#     per_setting_improvements = {}
#     for setting, rows in setting_to_rows.items():
#         imps = []
#         for col_idx in range(len(model_names)):
#             if col_idx == baseline_idx:
#                 imps.append(0.0)
#                 continue
#             rel_diffs = []
#             for row in rows:
#                 baseline_val = row[baseline_idx]
#                 current_val = row[col_idx]
#                 if np.isnan(baseline_val) or np.isnan(current_val) or baseline_val == 0:
#                     continue
#                 if lower_is_better:
#                     diff = (baseline_val - current_val) / baseline_val
#                 else:
#                     diff = (current_val - baseline_val) / baseline_val
#                 rel_diffs.append(diff)
#             imps.append(np.mean(rel_diffs) * 100 if rel_diffs else 0.0)
#         per_setting_improvements[setting] = imps

#     # --- 4. Generate LaTeX ---
#     def _is_best(a, b, tol=1e-12):
#         return a is not None and b is not None and abs(a - b) <= tol

#     def _fmt_imp(v):
#         return f"{v:.{decimals}f}\\%"

#     lines = []
#     lines.append(r"\begin{table}[h!]")
#     lines.append(fr"\caption{{Results for {metric_key} metric.}}")
#     lines.append(r"\vspace{-4mm}")
#     lines.append(r"\centering")
#     lines.append(r"\scalebox{0.4}{")
#     col_def = "lc" + "c" * len(model_names)
#     lines.append(fr"\begin{{tabular}}{{{col_def}}}")
#     lines.append(r"\toprule")

#     header_cells = ["", "Setting"] + [fr"\thead{{{m.replace('_', r'\_')}}}" for m in model_names]
#     lines.append(" & ".join(header_cells) + r" \\")
#     lines.append(r"\midrule")

#     for d_idx, dataset in enumerate(dataset_names):
#         rows = table_data[dataset]
#         num_rows = len(rows)

#         for r_idx, (setting, values) in enumerate(rows):
#             if r_idx == 0:
#                 ds_display = dataset.replace("_", r"\_")
#                 ds_cell = fr"\multirow{{{num_rows}}}{{*}}{{{ds_display}}}"
#             else:
#                 ds_cell = ""

#             setting_display = setting.replace("_", "-")
#             valid_values = [v for v in values if not np.isnan(v)]
#             best_val = None
#             if valid_values:
#                 best_val = min(valid_values) if lower_is_better else max(valid_values)

#             val_cells = []
#             for v in values:
#                 if np.isnan(v):
#                     val_cells.append("-")
#                 else:
#                     s_val = f"{v:.{decimals}f}"
#                     if best_val is not None and abs(v - best_val) < 1e-9:
#                         val_cells.append(fr"\textbf{{{s_val}}}")
#                     else:
#                         val_cells.append(s_val)

#             line_content = [ds_cell, setting_display] + val_cells
#             lines.append(" & ".join(line_content) + r" \\")

#         # Per-dataset improvement row (bold best improvement)
#         ds_imps = per_dataset_improvements.get(dataset, [0.0] * len(model_names))
#         best_imp = max(ds_imps) if ds_imps else None

#         imp_val_cells = []
#         for imp in ds_imps:
#             s = _fmt_imp(imp)
#             if _is_best(imp, best_imp):
#                 s = fr"\textbf{{{s}}}"
#             imp_val_cells.append(s)

#         imp_cells = ["", r"\textit{Improvement}"] + imp_val_cells
#         lines.append(" & ".join(imp_cells) + r" \\")

#         if d_idx < len(dataset_names) - 1:
#             lines.append(r"\midrule")

#     # Per-setting improvements section
#     lines.append(r"\midrule")
#     lines.append(r"\midrule")
#     distinct_settings = []
#     seen = set()
#     for ds in dataset_names:
#         for setting, _ in table_data[ds]:
#             if setting not in seen:
#                 seen.add(setting)
#                 distinct_settings.append(setting)

#     for i, setting in enumerate(distinct_settings):
#         label_cell = "Improvements" if i == 0 else ""
#         setting_display = setting.replace("_", "-")
#         imps = per_setting_improvements[setting]
#         best_imp = max(imps) if imps else None

#         imp_val_cells = []
#         for imp in imps:
#             s = _fmt_imp(imp)
#             if _is_best(imp, best_imp):
#                 s = fr"\textbf{{{s}}}"
#             imp_val_cells.append(s)

#         imp_cells = [label_cell, setting_display] + imp_val_cells
#         lines.append(" & ".join(imp_cells) + r" \\")

#     # Overall improvements row (bold best improvement)
#     lines.append(r"\midrule")
#     best_imp = max(improvement_percentages) if improvement_percentages else None

#     imp_val_cells = []
#     for imp in improvement_percentages:
#         s = _fmt_imp(imp)
#         if _is_best(imp, best_imp):
#             s = fr"\textbf{{{s}}}"
#         imp_val_cells.append(s)

#     imp_cells = ["Overall improvements", ""] + imp_val_cells
#     lines.append(" & ".join(imp_cells) + r" \\")

#     lines.append(r"\bottomrule")
#     lines.append(r"\end{tabular}")
#     lines.append(r"}")
#     lines.append(r"\label{tab:main}")
#     lines.append(r"\end{table}")

#     with open(output_tex_path, "w") as f:
#         f.write("\n".join(lines))
#     print(f"LaTeX table generated at: {output_tex_path}")



from pathlib import Path
from typing import Optional, Sequence

def _split_semicolon_str(x: str) -> list[str]:
    return [p.strip() for p in x.split(";") if p.strip()]


def _normalize_name_list(x) -> Optional[list[str]]:
    if x is None:
        return None
    if isinstance(x, str):
        return _split_semicolon_str(x)
    if isinstance(x, Sequence) and not isinstance(x, (bytes, bytearray)):
        return [str(v).strip() for v in x if v is not None and str(v).strip()]
    raise TypeError(f"Expected None, str, or Sequence[str], got {type(x)!r}")


def _list_subdirs(p: Path) -> list[str]:
    if not p.exists():
        return []
    return sorted([d.name for d in p.iterdir() if d.is_dir() and not d.name.startswith(".")])


def _read_json(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _mean_std(xs: list[float]) -> tuple[float, float]:
    xs = [x for x in xs if not np.isnan(x)]
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return float(xs[0]), 0.0
    arr = np.array(xs, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1))


def _fmt_value(v: float, decimals: int) -> str:
    return f"{v:.{decimals}f}"


def _fmt_value_with_std(mean: float, std: float, decimals: int, show_std: bool) -> str:
    if np.isnan(mean):
        return "-"
    if show_std and not np.isnan(std):
        return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"
    return f"{mean:.{decimals}f}"


def _is_best(a: float, b: float, tol: float = 1e-9) -> bool:
    return (not np.isnan(a)) and (not np.isnan(b)) and abs(a - b) <= tol


def _improvement_pct(ref: float, cur: float, lower_is_better: bool) -> float:
    if np.isnan(ref) or np.isnan(cur) or ref == 0:
        return float("nan")
    if lower_is_better:
        return (ref - cur) / ref * 100.0
    return (cur - ref) / ref * 100.0


def generate_results_table(
    experiment_dir: str,
    dataset_names=None,          # list[str] | "a;b" | None
    settings=None,               # list[str] | "a;b" | None (assumed same across datasets)
    json_filename: str = "results.json",
    model_names=None,            # list[str] | "a;b" | None
    metric_key: str = "nMSE",
    output_tex_path: Optional[str] = None,
    lower_is_better: bool = True,
    decimals: int = 4,
    ref_model: Optional[str] = None,
    multiplier=1,                # scalar or "a;b" (per-setting) or list
    seeds: bool = False,
    show_std: bool = False,
) -> None:
    exp = Path(experiment_dir)
    if not exp.exists():
        print(f"Error: Experiment directory '{experiment_dir}' does not exist.")
        return

    dataset_names_n = _normalize_name_list(dataset_names)
    settings_n = _normalize_name_list(settings)
    model_names_n = _normalize_name_list(model_names)

    if output_tex_path is None:
        output_tex_path = str(exp / "results.tex")

    # datasets (discover once)
    if dataset_names_n is None:
        dataset_names_n = sorted([d.name for d in exp.iterdir() if d.is_dir() and not d.name.startswith(".")])
        print(f"Auto-detected datasets: {dataset_names_n}")
        if not dataset_names_n:
            print("Error: No datasets found.")
            return

    # settings (discover once from first dataset)
    if settings_n is None:
        sdir = exp / dataset_names_n[0]
        settings_n = _list_subdirs(sdir)
        if not settings_n:
            print(f"Error: No settings found under: {sdir}")
            return

    # multiplier normalization (supports scalar, list, or ";" string)
    mult_list = _normalize_name_list(multiplier) if isinstance(multiplier, str) else None
    if mult_list is not None:
        mult_vals = [float(x) for x in mult_list]
    elif isinstance(multiplier, Sequence) and not isinstance(multiplier, (str, bytes, bytearray)):
        mult_vals = [float(x) for x in multiplier]
    else:
        mult_vals = [float(multiplier)] * len(settings_n)
    if len(mult_vals) == 1 and len(settings_n) > 1:
        mult_vals = mult_vals * len(settings_n)
    if len(mult_vals) != len(settings_n):
        raise ValueError("multiplier must be scalar or have same length as settings")

    # seeds (discover once from first dataset+setting if enabled)
    seed_names: list[str] = []
    if seeds:
        base = exp / dataset_names_n[0] / settings_n[0]
        seed_names = _list_subdirs(base)
        if not seed_names:
            print(f"Error: seeds=True but no seed folders found under: {base}")
            return
        print(f"Auto-detected seeds: {seed_names}")

    # model_names discovery:
    # - if not provided: try model subfolders (only for discovery), else fallback to JSON keys.
    if model_names_n is None or len(model_names_n) == 0:
        if seeds:
            probe = exp / dataset_names_n[0] / settings_n[0] / seed_names[0]
        else:
            probe = exp / dataset_names_n[0] / settings_n[0]
        found = _list_subdirs(probe)
        if found:
            model_names_n = found
            print(f"Auto-detected models from folders: {model_names_n}")
        else:
            # fallback to JSON keys
            json_probe = (probe / json_filename) if seeds else (probe / json_filename)
            if json_probe.exists():
                data = _read_json(json_probe)
                model_names_n = sorted(list(data.keys()))
                print(f"Auto-detected models from JSON keys: {model_names_n}")
            else:
                print("Error: Could not detect model names from folders or JSON files.")
                return

    if not model_names_n:
        print("Error: model_names is empty.")
        return

    if ref_model is None:
        ref_model = model_names_n[0]
    if ref_model not in model_names_n:
        raise ValueError(f"ref_model={ref_model!r} not in model_names={model_names_n}")

    # Load data: store mean/std per cell if seeds, else raw in mean and std=nan
    # table_data[dataset] = [(setting, means[], stds[])]
    table_data: dict[str, list[tuple[str, list[float], list[float]]]] = {ds: [] for ds in dataset_names_n}

    for ds in dataset_names_n:
        for si, setting in enumerate(settings_n):
            mult = mult_vals[si]
            means_row: list[float] = []
            stds_row: list[float] = []

            if seeds:
                # per-model aggregate over seeds
                seed_vals_by_model = {m: [] for m in model_names_n}
                for seed in seed_names:
                    fp = exp / ds / setting / seed / json_filename
                    if not fp.exists():
                        continue
                    try:
                        data = _read_json(fp)
                    except Exception:
                        continue
                    for m in model_names_n:
                        v = data.get(m, {}).get(metric_key, float("nan"))
                        v = float(v)
                        if not np.isnan(v):
                            v *= mult
                        seed_vals_by_model[m].append(v)

                for m in model_names_n:
                    mean, std = _mean_std(seed_vals_by_model[m])
                    means_row.append(mean)
                    stds_row.append(std)
            else:
                fp = exp / ds / setting / json_filename
                if fp.exists():
                    try:
                        data = _read_json(fp)
                    except Exception:
                        data = {}
                else:
                    data = {}

                for m in model_names_n:
                    v = data.get(m, {}).get(metric_key, float("nan"))
                    v = float(v)
                    if not np.isnan(v):
                        v *= mult
                    means_row.append(v)
                    stds_row.append(float("nan"))

            table_data[ds].append((setting, means_row, stds_row))

    # Improvements blocks (%): per-dataset, per-setting, overall (all vs ref_model)
    ref_idx = model_names_n.index(ref_model)

    def _avg_improvements(rows: list[list[float]]) -> list[float]:
        imps: list[float] = []
        for j in range(len(model_names_n)):
            if j == ref_idx:
                imps.append(0.0)
                continue
            diffs = []
            for r in rows:
                refv = r[ref_idx]
                curv = r[j]
                imp = _improvement_pct(refv, curv, lower_is_better)
                if not np.isnan(imp):
                    diffs.append(imp)
            imps.append(float(np.mean(diffs)) if diffs else 0.0)
        return imps

    per_dataset_improvements: dict[str, list[float]] = {}
    for ds in dataset_names_n:
        rows = [means for _, means, _ in table_data[ds]]
        per_dataset_improvements[ds] = _avg_improvements(rows)

    # per-setting: group across datasets
    setting_to_rows: dict[str, list[list[float]]] = {s: [] for s in settings_n}
    for ds in dataset_names_n:
        for setting, means, _ in table_data[ds]:
            setting_to_rows[setting].append(means)

    per_setting_improvements: dict[str, list[float]] = {}
    for setting, rows in setting_to_rows.items():
        per_setting_improvements[setting] = _avg_improvements(rows)

    # overall: across all rows
    all_rows: list[list[float]] = []
    for ds in dataset_names_n:
        for _, means, _ in table_data[ds]:
            all_rows.append(means)
    overall_improvements = _avg_improvements(all_rows)

    # LaTeX generation (full table env, multirows, bold best per row, improvements blocks)
    def _fmt_imp(v: float) -> str:
        return f"{v:.{decimals}f}\\%"

    lines: list[str] = []
    lines.append(r"\begin{table}[h!]")
    lines.append(fr"\caption{{Results for {metric_key} metric.}}")
    lines.append(r"\vspace{-4mm}")
    lines.append(r"\centering")
    lines.append(r"\scalebox{0.4}{")
    col_def = "lc" + "c" * len(model_names_n)
    lines.append(fr"\begin{{tabular}}{{{col_def}}}")
    lines.append(r"\toprule")

    header_cells = ["", "Setting"] + [fr"\thead{{{m.replace('_', r'\_')}}}" for m in model_names_n]
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    for d_idx, ds in enumerate(dataset_names_n):
        rows = table_data[ds]
        num_rows = len(rows)

        for r_idx, (setting, means, stds) in enumerate(rows):
            ds_cell = fr"\multirow{{{num_rows}}}{{*}}{{{ds.replace('_', r'\_')}}}" if r_idx == 0 else ""
            setting_display = setting.replace("_", "-")

            valid = [v for v in means if not np.isnan(v)]
            best_val = None
            if valid:
                best_val = min(valid) if lower_is_better else max(valid)

            val_cells: list[str] = []
            for j, v in enumerate(means):
                if np.isnan(v):
                    val_cells.append("-")
                    continue
                std = stds[j] if (seeds and show_std) else float("nan")
                s_val = _fmt_value_with_std(v, std, decimals=decimals, show_std=(seeds and show_std))

                if best_val is not None and _is_best(v, best_val):
                    s_val = fr"\textbf{{{s_val}}}"
                val_cells.append(s_val)

            lines.append(" & ".join([ds_cell, setting_display] + val_cells) + r" \\")

        # per-dataset improvement row
        ds_imps = per_dataset_improvements.get(ds, [0.0] * len(model_names_n))
        best_imp = max(ds_imps) if ds_imps else None
        imp_cells = []
        for imp in ds_imps:
            s = _fmt_imp(imp)
            if best_imp is not None and _is_best(imp, best_imp):
                s = fr"\textbf{{{s}}}"
            imp_cells.append(s)
        lines.append(" & ".join(["", r"\textit{Improvement}"] + imp_cells) + r" \\")

        if d_idx < len(dataset_names_n) - 1:
            lines.append(r"\midrule")

    # per-setting improvements block
    lines.append(r"\midrule")
    lines.append(r"\midrule")
    for i, setting in enumerate(settings_n):
        label_cell = "Improvements" if i == 0 else ""
        setting_display = setting.replace("_", "-")
        imps = per_setting_improvements.get(setting, [0.0] * len(model_names_n))
        best_imp = max(imps) if imps else None

        imp_cells = []
        for imp in imps:
            s = _fmt_imp(imp)
            if best_imp is not None and _is_best(imp, best_imp):
                s = fr"\textbf{{{s}}}"
            imp_cells.append(s)

        lines.append(" & ".join([label_cell, setting_display] + imp_cells) + r" \\")

    # overall improvements row
    lines.append(r"\midrule")
    best_imp = max(overall_improvements) if overall_improvements else None

    imp_cells = []
    for imp in overall_improvements:
        s = _fmt_imp(imp)
        if best_imp is not None and _is_best(imp, best_imp):
            s = fr"\textbf{{{s}}}"
        imp_cells.append(s)

    lines.append(" & ".join([f"Overall improvements", ""] + imp_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\label{tab:main}")
    lines.append(r"\end{table}")

    out_path = Path(output_tex_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"LaTeX table generated at: {out_path}")
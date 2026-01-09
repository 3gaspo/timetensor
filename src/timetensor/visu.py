import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from tabulate import tabulate
import os
import torch
import seaborn as sns

from .dataset import fetch_example_data
# from .analysis import *
from .utils import text_list

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

    # if save_latex:
    #     df_latex = df_mean.copy()
    #     for col in df_mean.columns:
    #         df_latex[col] = df_mean[col].map("{:.2f}".format)
    #     df_latex.columns = df_latex.columns.str.replace("_", "\\_", regex=False)
    #     pd_to_latex(df_latex, save_path, save_name=f"{save_name[:-3]}.tex")

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

# def pd_to_latex(df, save_path, save_name="results_table.tex"):
#     """returns latex code to create table from dataframe in path"""
#     colfmt = "l" + "c" * len(df.columns)
#     latex_str = df.to_latex(
#         index=True, escape=False, bold_rows=False,
#         column_format=colfmt, na_rep="--", caption=None, label=None, buf=None,
#         longtable=False, multirow=False, multicolumn=True, multicolumn_format='c',
#         header=True)
#     with open(save_path + save_name, "w", encoding="utf-8") as f:
#         f.write(latex_str)

def latex_formated_number(value, decimals=3, color=False, row=None, std=None):
    """return formated string value"""
    if value is None:
        return "--"

    if std is not None:
        fmt = f"{{:.{decimals}f}}" + " ± " + f"{std:.2f}"
    else:
        fmt = f"{{:.{decimals}f}}"
    s = fmt.format(value)

    if row is not None:
        m = min(row)
        if value == m:
            s = r"\textbf{" + s + "}"
    if color:
        if value > 0:
            return r"{\color{red}" + s + "}"
        elif value < 0:
            return r"{\color{green}" + s + "}"
    return s

def build_results_table_latex(
    save_dir, datasets, settings, show_row=0, models="RevIN", file_name="test1_mean_results.json", n_paths=1, multipliers=None, baseline=None, title="1e5 * MSE", save_name="test1_mean_results.tex", color=False, decimals=2, show_std=False, n_settings=4):
    """
    Returns a LaTeX tabular string
    Directory layout assumed: {save_dir}/{dataset}/lags{L}_horizon{H}/
    """
    datasets = text_list(datasets)
    settings = text_list(settings) #of size datasets * (settings per dataset)
    norm_settings = []
    for s in settings:
        _s = s.split("-")
        L, H = int(_s[0]), int(_s[1])
        norm_settings.append((L, H))

    n_paths = text_list(n_paths)
    n_paths = [int(text) for text in n_paths]
    if len(n_paths) == 1 and len(settings)>1:
        n_paths = [n_paths[0] for _ in range(len(settings))]
    models = text_list(models)

    # Collect values
    values = {}
    values_percent = {}
    values_std = {}
    multipliers = multipliers.split(";")
    for i, (L, H) in enumerate(norm_settings):
        for model in models:
            ds = datasets[i // n_settings]
            dir_name = save_dir + f"{ds}/lags{L}_horizon{H}/"
            df, df_std = get_multiple_errors_df(
                    dir_name=dir_name,
                    file_name=file_name,
                    n_paths=n_paths[i],
                    multipliers=multipliers[i],
                    baseline=None
                )

            key = f"{ds}_{L}_{H}"
            if key not in values:
                values[key] = []
                values_percent[key] = []
                values_std[key] = []
            try:
                values[key].append(df.iloc[show_row][model])
            except:
                raise ValueError(f"{i} {ds} {L} {H} {df.iloc[show_row]}")
            if show_std:
                values_std[key].append(df_std.iloc[show_row][model])
            if baseline is not None:
                df, _ = get_multiple_errors_df(
                        dir_name=dir_name,
                        file_name=file_name,
                        n_paths=n_paths[i],
                        multipliers=None,
                        baseline=baseline,
                        percents=True
                    )
                values_percent[key].append(df.iloc[show_row][model])
    
    lines = []
    colspec = "l" + "c" + "c" * len(models)
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")
    # lines.append(title + " & " + " & ".join(pretty_headers) + r" \\")
    lines.append(title + " & " + "L-H" + " & " + " & ".join([model.replace("_", r"\_") for model in models]) + r" \\")
    lines.append("\\midrule")

    # for ds in datasets:
    for i, (L, H) in enumerate(norm_settings):
        ds = datasets[i // n_settings]
        key = f"{ds}_{L}_{H}"
        ds_latex = ds.replace("_", r"\_").capitalize()
        if show_std:
            std = values_std[key][i]
        else:
            std = None
        cells = [latex_formated_number(v, decimals=decimals, color=color, row=values[key], std=std) for i, v in enumerate(values[key])]
        if i % n_settings == 0: #TODO: below instead of len(datasets), should be len(settings per dataset) but it depends on the dataset...
            lines.append("\\multirow{" + str(n_settings) + "}{*}{" + ds_latex + "}" + " & " + f"{L}-{H}" + " & " + " & ".join(cells) + r" \\")
        else:
            lines.append(" & " + f"{L}-{H}" + " & " + " & ".join(cells) + r" \\")
        if i % n_settings == n_settings - 1:
            lines.append("\\midrule")

    if baseline is not None:
        lines.append("\\midrule")
        values_percent = pd.DataFrame.from_dict(values_percent, orient="index")
        values_percent.columns = models
        means = values_percent.mean(axis=0).values
        lines.append("Improvements" + " & " + " & " + " & ".join([str(round(mean,2)) + r" \% " for mean in means]) + r" \\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    latex = "\n".join(lines)
    
    with open(save_dir + save_name, "w", encoding="utf-8") as f:
        f.write(latex)

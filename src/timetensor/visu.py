import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import json
from tabulate import tabulate
import os
import torch
import ipywidgets as widgets
from IPython.display import display
import seaborn as sns

from .utils import fetch_example_data #, get_stats
from .analysis import get_gammas

def plot_serie(x, path="", name="series.pdf", title="Time series", axis=True):
    """plots example data"""
    plt.clf()
    fig = plt.figure(figsize=(20,5))
    plt.plot(range(len(x)), x)
    if not axis:
      plt.axis('off')
      plt.title(None)
    plt.title(title)
    fig.tight_layout()
    plt.savefig(path + name)
    plt.close()

def plot_example(x, y, path="", name="example.pdf", title="Example", axis=True):
    """plots example data"""
    plt.clf()
    lag = len(x)
    horizon = len(y)
    fig = plt.figure(figsize=(20,5))
    plt.plot(range(lag), x, label="Lookback")
    plt.plot(range(lag, lag+horizon), y, label="Horizon")
    plt.axvline(x=lag, color='black', linestyle='--')
    plt.legend(bbox_to_anchor=(0.5, -0.15), ncol=3, loc='center', fontsize=14)
    if not axis:
      plt.axis('off')
      plt.title(None)
    plt.title(title)
    fig.tight_layout()
    plt.savefig(path + name)
    plt.close()

def plot_named_example(path, name):
    x, c, y, i, d  = fetch_example_data(path, name)
    plot_example(x[0], y[0], path + f"/{name}/", f"example.pdf", "Example")


# def plot_global_stats(values, path="", dim=0):
#     """plots stats of datasets"""
#     plt.clf()
#     fig = plt.figure(figsize=(15,5))
    
#     global_mean = values[:, dim, :].mean()
#     local_means = values[:, dim, :].mean(dim=-1)
#     plt.hist(local_means, bins=np.logspace(0, 6))
#     plt.xscale("log")
#     plt.xlabel("kWh")
#     plt.ylabel("Counts")        
#     plt.title(f"Distribution of users means (total avg: {global_mean:.2f} kWh)")
#     plt.savefig(path + "global_means.pdf")

#     global_std = values[:, dim, :].std()
#     local_std = values[:, dim, :].std(dim=-1)
#     plt.hist(local_std, bins=np.logspace(0, 6))
#     plt.xscale("log")
#     plt.xlabel("kWh")
#     plt.ylabel("Counts")        
#     plt.title(f"Distribution of users std (total avg: {global_std:.2f} kWh)")
#     plt.savefig(path + "global_stds.pdf")
#     plt.close()


# def plot_stats(values, path="", name="stats.pdf", dim=0, title=None, logscale=True, limits=None):
#     """plots stats of datasets"""
#     plt.clf()
#     fig = plt.figure(figsize=(15,5))
#     if type(values) is dict:
#         for split_name, split_values in values.items():
#             mean_values, total_mean = get_stats(split_values, "mean", dim)
#             if len(mean_values)>2000:
#                 idx=random.sample(range(len(mean_values)),1000)
#                 mean_values = mean_values[idx]
#             if logscale:
#                 bins = np.logspace(-3, 3, 100)
#             else:
#                 bins = 100
#             plt.hist(mean_values, bins=bins, range=limits, density=True, alpha=0.5, label= f"{split_name} - mean={total_mean:.2f}")
#             plt.legend()
#     else:
#         if logscale:
#             bins = np.logspace(-3, 3, 100)
#         else:
#             bins = 100
#         plt.hist(values, bins=bins, range=limits, density=True, alpha=0.5)
        
    
#     if title is None:
#         plt.title(f"Means distribution")
#     else:
#         plt.title(title)
#     plt.xlabel(f"means")
#     if logscale:
#         plt.xscale("log")
#     plt.ylabel("Density")
#     plt.savefig(path + name)
#     plt.close()


# def scatter_stats(values_dict, path="", name="stats.pdf", dim=0, title=None, logscale=True):
#     """plots stats of datasets"""
#     plt.clf()
#     fig = plt.figure(figsize=(10,5))
#     for split_name, split_values in values_dict.items():
#         std_values, total_std = get_stats(split_values, "std", dim)
#         mean_values, total_mean = get_stats(split_values, "mean", dim)
#         if len(mean_values)>2000:
#             idx=random.sample(range(len(mean_values)),1000)
#             mean_values, std_values = mean_values[idx], std_values[idx]
#         plt.scatter(mean_values, std_values, label= f"{split_name} - std={total_std:.2f}, mean={total_mean:.2f}", s=10)

#     plt.legend()
#     if title is None:
#         plt.title(f"Distributions")
#     else:
#         plt.title(title)
#     plt.xlabel(f"mean")
#     if logscale:
#         plt.xscale("log")
#         plt.yscale("log")
#     plt.ylabel("std")
#     plt.savefig(path + name)
#     plt.close()

# def scatter_input_output(x_dict, y_dict, path="", name="stats.pdf", dim=0, title=None, logscale=True):
#     """plots stats of datasets"""
#     plt.clf()
#     fig = plt.figure(figsize=(10,5))
#     for key in x_dict:
#         xmean_values, xtotal_mean = get_stats(x_dict[key], "mean", dim)
#         ymean_values, ytotal_mean = get_stats(y_dict[key], "mean", dim)

#         if len(xmean_values)>2000:
#             idx=random.sample(range(len(xmean_values)),1000)
#             xmean_values, ymean_values = xmean_values[idx], ymean_values[idx]

#         plt.scatter(xmean_values, ymean_values, label= f"{key} - mean_x={xtotal_mean:.2f}, mean_y={ytotal_mean:.2f}", s=10)

#     plt.legend()
#     if title is None:
#         plt.title(f"Output/Input mean distributions")
#     else:
#         plt.title(title)
#     plt.xlabel(f"Input means")
#     plt.ylabel("Output means")
#     if logscale:
#         plt.xscale("log")
#         plt.yscale("log")
#     plt.savefig(path + name)
#     plt.close()

def plot_stats(data, path="", name="stats.pdf", show=False, per_user=True, lookback=336, samples=1000, title=None, remove_cte=True):
    """plots means and stds. data must be pandas dataframe or dict of df"""
    if type(data) != dict:
        data = {"data":data}

    keys, means_list, stds_list = [], [], []
    for key, df in data.items():
        if per_user:
            means = df.mean(axis=0)
            stds = df.std(axis=0)
        else:
            means = df.rolling(window=lookback).mean()[lookback:].stack().sample(samples)
            stds = df.rolling(window=lookback).std()[lookback:].stack().sample(samples)
        
        if remove_cte:
            keep_idx = np.where(stds>0)[0]
        else:
            keep_idx = np.array(means.index)
        keys += [key + f" (mean: {means.iloc[keep_idx].median():.2f} | std: {stds.iloc[keep_idx].median():.2f})" for k in range(len(means.iloc[keep_idx]))]
        means_list += np.log(np.where(means.iloc[keep_idx]>0, means.iloc[keep_idx], 1e-8)).tolist()
        stds_list += np.log(stds.iloc[keep_idx]).tolist() #np.where(stds>0, stds, 1e-8)).tolist()

    stats_df = pd.DataFrame({
        'key': keys,
        'log(mean)': means_list,
        'log(std)': stds_list})

    sns.set_theme(style="white")

    g = sns.jointplot(
        data=stats_df,
        x='log(mean)',
        y='log(std)',
        hue='key',
        kind='scatter',
        palette='Set1',
        marginal_kws=dict(common_norm=False, fill=True, alpha=0.5)
    )

    g.plot_joint(sns.kdeplot, hue='key', fill=False, alpha=0.3)

    if title is None:
        plt.suptitle("Statistics distribution", y=1.02)
    else:
        plt.suptitle(title)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(path+name)
    plt.close()


def plot_means(data, path="", name="stats.pdf", show=False, per_user=True, lookback=336, samples=1000, title=None, remove_cte=True):

    if type(data) != dict:
        data = {"data":data}

    keys, means_list = [], []
    for key, df in data.items():
        if per_user:
            means = df.mean(axis=0)
            stds = df.std(axis=0)
        else:
            means = df.rolling(window=lookback).mean()[lookback:].stack().sample(samples)
            stds = df.rolling(window=lookback).std()[lookback:].stack().sample(samples)
        
        if remove_cte:
            keep_idx = np.where(stds>0)[0]
        else:
            keep_idx = np.array(means.index)
        keys += [key + f" (mean: {means.iloc[keep_idx].median():.2f} | std: {stds.iloc[keep_idx].median():.2f})" for k in range(len(means.iloc[keep_idx]))]
        means_list += np.log(np.where(means.iloc[keep_idx]>0, means.iloc[keep_idx], 1e-8)).tolist()
    
    means_df = pd.DataFrame({
        'key': keys,
        'log(mean)': means_list,})
    
    sns.kdeplot(means_df, x="log(mean)", hue="key", fill=True)#, log_scale=False), #label=f"{key} (avg:{means.mean():.2f})")

    if title is None:
        plt.title(f"Means distribution")
    else:
        plt.title(title)
    plt.xlabel("Values")
    plt.ylabel("Density")
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(path+name)
    plt.close()


def plot_box(data, users, dates):
    plt.imshow(data.values.T[:users, :dates])



def plot_losses(train_losses, valid_losses_dict=None, path="", name="losses.pdf", title="Losses", logscale=True, eval_freq=10):
    """plots losses during training"""
    plt.clf()
    fig = plt.figure(figsize=(10,5))
    if valid_losses_dict is not None:
        plt.plot(range(1, len(train_losses)+1), train_losses, label="train")
        for key, values in valid_losses_dict.items():
            T = [1]
            k = 1
            while len(T) < len(values)-1:
                T.append(eval_freq * k)
                k+=1
            T.append(len(train_losses))
            plt.plot(T, values, label="valid")
        plt.legend()
    else:
        plt.plot(range(1, len(train_losses)+1), train_losses)
    if logscale:
      plt.yscale('log')
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title(title)
    fig.tight_layout()
    plt.savefig(path + name)
    plt.close()

def plot_multi_losses(losses_dict, path="", name="losses.pdf", title="Losses", logscale=True, x_every=None, eval_freq=10):
    """plots losses during training"""
    plt.clf()
    fig = plt.figure(figsize=(10,5))
    for expe_name, losses in losses_dict.items():
        T = [1] + [k*eval_freq for k in range(1,len(losses))]
        #plt.plot(range(1, len(losses)+1), losses, label=f"{expe_name}")
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
    plt.savefig(path + name)
    plt.close()


def plot_errors(losses, path="", name="errors.pdf", title="Loss distribution"):
    """plots histogram of errors"""
    plt.clf()
    fig = plt.figure(figsize=(10,5))
    plt.hist(losses, bins=100)
    plt.yscale("log")
    plt.title(title)
    plt.xlabel("Losses")
    plt.ylabel("Frequency")
    plt.savefig(path + name)
    plt.close()


def plot_horizon_errors(losses, path="", name="horizon.pdf", title="Mean errors by horizon"):
    """plots errors according to horizon"""
    plt.clf()
    fig = plt.figure(figsize=(15,5))
    plt.bar(range(len(losses)), losses)
    plt.title(title)
    plt.xlabel("Horizon")
    plt.ylabel("Mean error")
    plt.savefig(path + name)
    plt.close()


def plot_pred(x, y, pred, path="", name="prediction.pdf", title="Predictions", axis=True):
    """plots example prediction"""
    plt.clf()
    lag = len(x)
    horizon = len(y)
    fig = plt.figure(figsize=(20,5))
    plt.plot(range(lag), x, label="Lookback")
    plt.plot(range(lag, lag+horizon), pred, label="Prediction")
    plt.plot(range(lag, lag+horizon), y, label="Horizon")
    plt.axvline(x=lag, color='black', linestyle='--')
    plt.legend(bbox_to_anchor=(0.5, -0.15), ncol=3, loc='center', fontsize=14)
    if not axis:
      plt.axis('off')
      plt.title(None)
    plt.title(title)
    fig.tight_layout()
    plt.savefig(path + name)
    plt.close()



def pd_to_latex(path):
    """returns latex code to create table from dataframe in path"""
    df = pd.read_csv(path)
    latex_output = df.to_latex(index=False, float_format="%.4f")
    print(latex_output)


def print_nice_table(path, multipliers=None, names=None):
    """print table from dataframe in path"""
    with open(path) as file:
        data = json.load(file)
    df = pd.DataFrame(data)
    if names=="None":
        names=None
    if names is not None:
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
    table = tabulate(df, headers='keys', tablefmt='grid', showindex=True, floatfmt=".4f")
    print(table)


def print_nice_tables(dir_name, file_name, n_paths, multipliers=None, names=None, show_std=True, baseline=None):
    """print table from dataframe in path"""
    paths = [dir_name + f"seed_{k}/" + file_name for k in range(1,n_paths+1)]
    dfs = []
    for path in paths:
        with open(path) as file:
            data = json.load(file)
        df = pd.DataFrame(data)
        if names=="None":
            names=None
        if names is not None:
            df = df[names]

        if baseline is not None and baseline in df.columns:
            df = df.subtract(df[baseline], axis=0)
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

    if show_std:
        df_formatted = df_mean.copy()
        for col in df_mean.columns:
            df_formatted[col] = df_mean[col].map("{:.4f}".format) + " ± " + df_std[col].map("{:.4f}".format)
    else:
        df_formatted = df_mean.applymap("{:.4f}".format)

    table = tabulate(df_formatted, headers='keys', tablefmt='grid', showindex=True)
    print(table)


def get_boxplots(dir_name, file_name, n_paths, col="Test MSE", names=None, baseline=None, save_path=""):
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
            df = df[names]
        df = df.loc[col]
        if baseline is not None:
            assert (baseline in df.columns)
            df = df.subtract(df[baseline], axis=0)
        
        for algo, value in df.items():
            box_df.append({"Algorithm": algo, f"{col}": value, "seed":k})

    #df_values = pd.concat(dfs, axis=1)
    #df_long = pd.concat(dfs, axis=1).reset_index().melt(id_vars='index', var_name='Method', value_name=col)
    box_df = pd.DataFrame(box_df)

    plt.figure(figsize=(10, 6))
    #plt.boxplot(df_values.values.T, labels=df_values.index)
    sns.boxplot(data=box_df, x='Algorithm', y=col)#, hue="seed")
    plt.title(f"Experiment results")
    plt.xlabel("Experiment")
    plt.ylabel(f"{col}")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path + "boxplot.pdf")
    plt.close()


def plot_weights(weights, path, name="weights.pdf", title='Model weights'):
    plt.figure()
    plt.imshow(weights, aspect='auto', cmap='viridis')
    plt.colorbar(label='Weight value')
    plt.xlabel('Inputs (lookback)')
    plt.ylabel('Outputs (horizon)')
    plt.title(title)
    plt.savefig(path + name)
    plt.close()


def plot_expe(losses_path, eval_freq=10, names=None, save_path=None):
    """plots losses for list of experiments in path"""
    if type(eval_freq)==str:
        eval_freq=int(eval_freq)
    if names=="None":
        names=None
    if save_path is None:
        save_path = losses_path+"plots/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    expe_names = [name for name in os.listdir(losses_path) if (names is None and os.path.exists(losses_path + f"{name}/" + "valid_losses1.pt")) or (names is not None and name in names)]

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
            plot_multi_losses(losses_dict1[loss_name], save_path, f"{loss_name}_valid1.pdf", f"Valid {loss_name}", eval_freq=eval_freq)
            plot_multi_losses(losses_dict2[loss_name], save_path, f"{loss_name}_valid2.pdf", f"Valid2 {loss_name}", eval_freq=eval_freq)
            plot_multi_losses(losses_dict3[loss_name], save_path, f"{loss_name}_valid3.pdf", f"Valid3 {loss_name}", eval_freq=eval_freq)



def visu_widget(data, lookback, horizon, eps=1e-6):

    alphas, betas = get_gammas(data, lookback, horizon, eps)
    dataframes = {'original': data, 'alpha': alphas, 'beta': betas}
    dataframe_names = list(dataframes.keys())
    column_names = list(data.columns)

    dataframe_dropdown = widgets.Dropdown(
        options=dataframe_names,
        value=dataframe_names[0],
        description='Select DataFrame:'
    )
    column_dropdown = widgets.Dropdown(
        options=column_names,
        value=column_names[0],
        description='Select Column:'
    )

    next_button = widgets.Button(description="Next Column")
    output = widgets.Output()

    def update_plot(dataframe_name, column_name):
        with output:
            output.clear_output(wait=True)
            df = dataframes[dataframe_name]
            plt.figure(figsize=(12, 6))
            plt.plot(df[column_name])
            plt.title(f'{dataframe_name} time Series for user {column_name}')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.grid(True)
            plt.show()

    def on_dropdown_change(change):
        update_plot(dataframe_dropdown.value, column_dropdown.value)

    def on_next_button_click(b):
        current_column_index = column_names.index(column_dropdown.value)
        next_column_index = (current_column_index + 1) % len(column_names)
        column_dropdown.value = column_names[next_column_index]

    dataframe_dropdown.observe(on_dropdown_change, names='value')
    column_dropdown.observe(on_dropdown_change, names='value')
    next_button.on_click(on_next_button_click)

    display(dataframe_dropdown, column_dropdown, next_button, output)
    update_plot(dataframe_dropdown.value, column_dropdown.value)

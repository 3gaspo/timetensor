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

from .dataset import fetch_example_data
from .analysis import *

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


# def valid_for_kde(sub, keyx, keyy):
#     return (
#         len(sub) >= 3 and
#         sub[keyy].nunique() >= 2 and sub[keyx].nunique() >= 2 and
#         sub[keyy].std() > 0 and sub[keyx].std() > 0
#     )
def valid_for_kde(sub, keyx, keyy):
    a = sub[keyy].to_numpy()
    b = sub[keyx].to_numpy()
    na = np.sum(~np.isnan(a))
    nb = np.sum(~np.isnan(b))
    if na < 2 or nb < 2:
        return False
    if np.nanmin(a) == np.nanmax(a) or np.nanmin(b) == np.nanmax(b):
        return False
    return len(sub) >= 3

def plot_stats(data, path="", name="stats.pdf", per_user=True, lookback=336, samples=1000, title=None, remove_cte=True, log=False, show=False):
    """plots means and stds. data must be pandas dataframe or dict of df"""
    if type(data) != dict:
        data = {"data":data}

    keys, means_list, stds_list = [], [], []
    for key, df in data.items():
        if per_user:
            clean_df = df.copy()
            if remove_cte:
                cte_mask, _ = identify_cte(df, lookback)
                clean_df[cte_mask] = pd.NA
            means = clean_df.mean(axis=0)
            stds = clean_df.std(axis=0)
            if remove_cte and np.any(stds==0):
                raise ValueError("Constant windows wrongly kept")
        else:
            means = df.rolling(window=lookback).mean()[lookback:].stack()#.sample(samples)
            stds = df.rolling(window=lookback).std()[lookback:].stack()#.sample(samples)
            sampled_idx = np.random.choice(len(means), size=samples, replace=False)
            means = means.iloc[sampled_idx]
            stds = stds.iloc[sampled_idx]
            if remove_cte:
                keep_idx = np.where(stds>0)[0]
                means, stds = means.iloc[keep_idx], stds.iloc[keep_idx]
        keys += [key + f" (mean: {means.mean():.2f} | stds: {stds.mean():.2f}" for _ in range(len(means))]
        if log:
            means_list += np.log(np.where(means>0, means, 1e-8)).tolist()
            stds_list += np.log(np.where(stds>0, stds, 1e-8)).tolist()
            xlbl, ylbl = "log(mean)", "log(std)"
        else:
            means_list += means.tolist()
            stds_list += stds.tolist()
            xlbl, ylbl = "mean", "std"

    stats_df = pd.DataFrame({
        'key': keys,
        xlbl: means_list,
        ylbl: stds_list})

    sns.set_theme(style="white")

    g = sns.jointplot(
        data=stats_df,
        x=xlbl,
        y=ylbl,
        hue='key',
        kind='scatter',
        palette='Set1',
        marginal_kws=dict(common_norm=False, fill=True, alpha=0.5)
    )

    #g.plot_joint(sns.kdeplot, hue='key', fill=False, alpha=0.3)
    ax = g.ax_joint
    hue_order = list(dict.fromkeys(stats_df["key"]))  # preserves first-seen order
    palette = sns.color_palette("Set1", n_colors=len(hue_order))
    color_for = dict(zip(hue_order, palette))
    for key, sub in stats_df.groupby("key"):
        if not valid_for_kde(sub, xlbl, ylbl):
            continue
        try:
            sns.kdeplot(
                data=sub,
                x="beta", y="alpha",
                ax=ax,
                color=color_for[key],   # match scatter color
                fill=False, alpha=0.3,
                levels=10,              # strictly increasing
                thresh=1e-6,
                bw_adjust=1.2,
                warn_singular=False,
                common_norm=False,
                legend=False,           # avoid legend duplication
            )
        except ValueError: # If a group still blows up, just skip its KDE
            pass

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


def plot_means(data, path="", name="stats.pdf", per_user=True, lookback=336, samples=1000, title=None, remove_cte=True, log=False, show=False):

    if type(data) != dict:
        data = {"data":data}

    keys, means_list = [], []
    for key, df in data.items():
        if per_user:
            clean_df = df.copy()
            if remove_cte:
                cte_mask, _ = identify_cte(df, lookback)
                clean_df[cte_mask] = pd.NA
            means = clean_df.mean(axis=0)
        else:
            means = df.rolling(window=lookback).mean()[lookback:].stack()#.sample(samples)
            stds = df.rolling(window=lookback).std()[lookback:].stack()#.sample(samples)
            sampled_idx = np.random.choice(len(means), size=samples, replace=False)
            means = means.iloc[sampled_idx]
            stds = stds.iloc[sampled_idx]
            if remove_cte:
                keep_idx = np.where(stds>0)[0]
                means = means.iloc[keep_idx]

        keys += [key + f" (mean: {means.mean():.2f}" for _ in range(len(means))]
        if log:
            means_list += np.log(np.where(means>0, means, 1e-8)).tolist()
        else:
            means_list += means.tolist()

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
    plt.savefig(path + name)
    plt.close()

def plot_multi_losses(losses_dict, path="", name="losses.pdf", title="Losses", logscale=True, x_every=None, eval_freq=1):
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


def get_errors_df(dir_name, file_name, multipliers=None, names=None, save=True):
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




def get_multiple_errors_df(dir_name, file_name, n_paths, multipliers=None, names=None, baseline=None, save=False):
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

    if save:
        df_mean.to_csv(dir_name + 'mean_errors.csv')
        df_std.to_csv(dir_name + 'std_errors.csv')
    return df_mean, df_std


def get_expe_results(dir_name, file_name, multipliers=None, names=None, print_table=True, save_path=None, save_name="errors.pdf"):
    df = get_errors_df(dir_name, file_name, multipliers, names, save=True)
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

def get_multiple_expe_results(dir_name, file_name, n_paths, multipliers=None, names=None, show_std=True, baseline=None, print_table=True, show_row=0, save_path=None,save_name="errors.df"):
    df_mean, df_std = get_multiple_errors_df(dir_name, file_name, n_paths, multipliers, names, baseline, save=True)

    if show_std:
        df_formatted = df_mean.copy()
        for col in df_mean.columns:
            df_formatted[col] = df_mean[col].map("{:.4f}".format) + " ± " + df_std[col].map("{:.4f}".format)
    else:
        df_formatted = df_mean.applymap("{:.4f}".format)

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

    if save_path is None:
        save_path = dir_name+"plots/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

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

    if save_path is None:
        save_path = dir_name+"plots/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.savefig(save_path + save_name)
    plt.close()


def plot_expe(losses_path, eval_freq=10, names=None, save_path=None):
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



def visu_widget(data, lookback, horizon, eps=1e-8):

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


def plot_clustering(raw_df, feature_df, n_clusters, lags, horizon, clustering_name, plot_dir, do_heterogeneity=True, remove_cte=True):
    if do_heterogeneity:
        plot_heterogeneity(feature_df, path=plot_dir, name="heterogeneity.pdf")
    Z, distances_matrix = init_clusters(feature_df)
    labels, cluster_indices = get_clusters(Z, n_clusters)
    plot_dendogram(Z, path=plot_dir, name="dendogram.pdf")
    plot_distances(distances_matrix, path=plot_dir, name="distances.pdf")
    centroids = get_centroids(feature_df, cluster_indices)
    plot_centroids(centroids, path=plot_dir, name="centroids.pdf")
    centroids = get_centroids(raw_df, cluster_indices)
    plot_centroids(centroids, path=plot_dir, name="raw_centroids.pdf")
    df_dict = get_cluster_dicts(raw_df, cluster_indices)
    plot_stats(df_dict, plot_dir, name="stats.pdf", per_user=True, lookback=lags, title=f"{clustering_name} input statistics", remove_cte=remove_cte, log=True)
    plot_gamma(df_dict, plot_dir, "gammas.pdf", per_user=True, lookback=lags, horizon=horizon, remove_cte=remove_cte, log=False)
    return cluster_indices

def plot_weights_(weights, path, name="weights.pdf", title='Model weights'):
    plt.figure()
    plt.imshow(weights, aspect='auto', cmap='viridis')
    plt.colorbar(label='Weight value')
    plt.xlabel('Inputs (lookback)')
    plt.ylabel('Outputs (horizon)')
    plt.title(title)
    plt.savefig(path + name)
    plt.close()

def plot_weights(model, learner, save_dir, save_name):
    model_name = model.name
    if model_name in ["linear", "sklinear"]:
        if model_name == "sklinear":
            weights = learner.get_weights()
        else:
            weights = model.fc.weight.detach().cpu().numpy()
        plot_weights_(weights, save_dir + "plots/", title=f'{save_name} weights')
        
    if model_name == "DLinear":
        linear_weights = model.model.Linear_Seasonal[0].weight.detach().cpu().numpy()
        season_weights = model.model.Linear_Trend[0].weight.detach().cpu().numpy()
        plot_weights_(linear_weights, save_dir + "plots/", name="season_weights.pdf", title=f'{save_name} seasonal weights')
        plot_weights_(season_weights, save_dir + "plots/", name="trend_weights.pdf", title=f'{save_name} trend weights')
    
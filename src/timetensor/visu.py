import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import json
from tabulate import tabulate
import os
import torch

from .utils import get_stats, fetch_example_data


def plot_example(x, y, path="", name="example.pdf", title="Example"):
    """plots example data"""
    plt.clf()
    lag = len(x)
    horizon = len(y)
    fig = plt.figure(figsize=(20,5))
    plt.plot(range(lag), x, label="Lookback")
    plt.plot(range(lag, lag+horizon), y, label="Horizon")
    plt.axvline(x=lag, color='black', linestyle='--')
    plt.legend(bbox_to_anchor=(0.5, -0.15), ncol=3, loc='center', fontsize=14)
    plt.title(title)
    fig.tight_layout()
    plt.savefig(path + name)

def plot_named_example(path, name):
    x, c, y, i, d  = fetch_example_data(path, name)
    plot_example(x[0], y[0], path + f"/{name}/", f"example.pdf", "Example")


def plot_stats(values, path="", name="stats.pdf", dim=0, title=None, logscale=True, limits=None):
    """plots stats of datasets"""
    plt.clf()
    fig = plt.figure(figsize=(15,5))
    if type(values) is dict:
        for split_name, split_values in values.items():
            mean_values, total_mean = get_stats(split_values, "mean", dim)
            if len(mean_values)>2000:
                idx=random.sample(range(len(mean_values)),1000)
                mean_values = mean_values[idx]
            if logscale:
                bins = np.logspace(-3, 3, 100)
            else:
                bins = 100
            plt.hist(mean_values, bins=bins, range=limits, density=True, alpha=0.5, label= f"{split_name} - mean={total_mean:.2f}")
            plt.legend()
    else:
        if logscale:
            bins = np.logspace(-3, 3, 100)
        else:
            bins = 100
        plt.hist(values, bins=bins, range=limits, density=True, alpha=0.5)
        
    
    if title is None:
        plt.title(f"Means distribution")
    else:
        plt.title(title)
    plt.xlabel(f"means")
    if logscale:
        plt.xscale("log")
    plt.ylabel("Density")
    plt.savefig(path + name)


def scatter_stats(values_dict, path="", name="stats.pdf", dim=0, title=None, logscale=True):
    """plots stats of datasets"""
    plt.clf()
    fig = plt.figure(figsize=(10,5))
    for split_name, split_values in values_dict.items():
        std_values, total_std = get_stats(split_values, "std", dim)
        mean_values, total_mean = get_stats(split_values, "mean", dim)
        if len(mean_values)>2000:
            idx=random.sample(range(len(mean_values)),1000)
            mean_values, std_values = mean_values[idx], std_values[idx]
        plt.scatter(mean_values, std_values, label= f"{split_name} - std={total_std:.2f}, mean={total_mean:.2f}", s=10)

    plt.legend()
    if title is None:
        plt.title(f"Distributions")
    else:
        plt.title(title)
    plt.xlabel(f"mean")
    if logscale:
        plt.xscale("log")
        plt.yscale("log")
    plt.ylabel("std")
    plt.savefig(path + name)

def scatter_input_output(x_dict, y_dict, path="", name="stats.pdf", dim=0, title=None, logscale=True):
    """plots stats of datasets"""
    plt.clf()
    fig = plt.figure(figsize=(10,5))
    for key in x_dict:
        xmean_values, xtotal_mean = get_stats(x_dict[key], "mean", dim)
        ymean_values, ytotal_mean = get_stats(y_dict[key], "mean", dim)

        if len(xmean_values)>2000:
            idx=random.sample(range(len(xmean_values)),1000)
            xmean_values, ymean_values = xmean_values[idx], ymean_values[idx]

        plt.scatter(xmean_values, ymean_values, label= f"{key} - mean_x={xtotal_mean:.2f}, mean_y={ytotal_mean:.2f}", s=10)

    plt.legend()
    if title is None:
        plt.title(f"Output/Input mean distributions")
    else:
        plt.title(title)
    plt.xlabel(f"Input means")
    plt.ylabel("Output means")
    if logscale:
        plt.xscale("log")
        plt.yscale("log")
    plt.savefig(path + name)


def plot_losses(train_losses, valid_losses=None, valid_losses2=None, path="", name="losses.pdf", title="Losses", logscale=True, eval_freq=10):
    """plots losses during training"""
    plt.clf()
    fig = plt.figure(figsize=(10,5))
    if valid_losses is not None:
        plt.plot(range(1, len(train_losses)+1), train_losses, label="train")
        T = [1]
        k = 1
        while len(T) < len(valid_losses)-1:
            T.append(eval_freq * k)
            k+=1
        T.append(len(train_losses))
        plt.plot(T, valid_losses, label="valid")
        if valid_losses2 is not None:
            plt.plot(T, valid_losses2, label="valid2")
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

def plot_multi_losses(losses_dict, path="", name="losses.pdf", title="Losses", logscale=True, x_every=None):
    """plots losses during training"""
    plt.clf()
    fig = plt.figure(figsize=(10,5))
    for expe_name, losses in losses_dict.items():
        plt.plot(range(1, len(losses)+1), losses, label=f"{expe_name}")
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


def plot_horizon_errors(losses, path="", name="horizon.pdf", title="Mean errors by horizon"):
    """plots errors according to horizon"""
    plt.clf()
    fig = plt.figure(figsize=(15,5))
    plt.bar(range(len(losses)), losses)
    plt.title(title)
    plt.xlabel("Horizon")
    plt.ylabel("Mean error")
    plt.savefig(path + name)


def plot_pred(x, y, pred, path="", name="prediction.pdf", title="Predictions"):
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
    plt.title(title)
    fig.tight_layout()
    plt.savefig(path + name)



def pd_to_latex(path):
    """returns latex code to create table from dataframe in path"""
    df = pd.read_csv(path)
    latex_output = df.to_latex(index=False, float_format="%.4f")
    print(latex_output)


def print_nice_table(path, multipliers=None):
    """print table from dataframe in path"""
    with open(path) as file:
        data = json.load(file)
    df = pd.DataFrame(data)
    if multipliers is not None:
        if type(multipliers) == str:
            multipliers = multipliers.split(" ")
            multipliers = [int(w) for w in multipliers]
        new_index = list(df.index)
        for k in range(min(len(multipliers), df.shape[1])):
            if multipliers[k] != 0:
                df.iloc[k] = df.iloc[k] * 10**multipliers[k]
                new_index[k] = new_index[k] + f" * 1e{multipliers[k]}"
        df.index = new_index
    table = tabulate(df, headers='keys', tablefmt='grid', showindex=True, floatfmt=".4f")
    print(table)


def plot_weights(weights, path, name="weights.pdf", title='Model weights'):
    plt.figure()
    plt.imshow(weights, aspect='auto', cmap='viridis')
    plt.colorbar(label='Weight value')
    plt.xlabel('Inputs (lookback)')
    plt.ylabel('Outputs (horizon)')
    plt.title(title)
    plt.savefig(path + name)


def plot_expe(path):
    """plots losses for list of experiments in path"""
    expe_names = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) and name not in ["persistence", "repeat", "lookback"] and "sklinear" not in name]

    if len(expe_names) >0:
        losses_dict = {}
        losses_dict2 = {}

        for expe_name in expe_names:
            valid_losses = torch.load(path + expe_name + "/" + "valid_losses.pt", weights_only=False)
            valid_losses2 = torch.load(path + expe_name + "/" + "valid_losses2.pt", weights_only=False)


            for loss_name in valid_losses:
                if loss_name not in losses_dict:
                    losses_dict[loss_name] = {}
                    losses_dict2[loss_name] = {}
                losses_dict[loss_name][expe_name] = valid_losses[loss_name]
                losses_dict2[loss_name][expe_name] = valid_losses2[loss_name]

        for loss_name in valid_losses:
            plot_multi_losses(losses_dict[loss_name], path, f"{loss_name}_valid.pdf", f"Valid {loss_name}")
            plot_multi_losses(losses_dict2[loss_name], path, f"{loss_name}_valid2.pdf", f"Valid2 {loss_name}")
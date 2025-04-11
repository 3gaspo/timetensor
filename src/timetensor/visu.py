import matplotlib.pyplot as plt
import numpy as np
import random

from .utils import get_stats

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


def plot_stats(values_dict, path="", name="stats.pdf", dim=0, title=None, logscale=True):
    """plots stats of datasets"""
    plt.clf()
    fig = plt.figure(figsize=(10,5))
    for split_name, split_values in values_dict.items():
        mean_values, total_mean = get_stats(split_values, "mean", dim)
        if len(mean_values)>2000:
            idx=random.sample(range(len(mean_values)),1000)
            mean_values = mean_values[idx]
        if logscale:
            bins = np.logspace(-2, 6, 100)
        else:
            bins = 100
        plt.hist(mean_values, bins=bins, density=True, alpha=0.5, label= f"{split_name} - mean={total_mean:.2f}")

    plt.legend()
    if title is None:
        plt.title(f"Max mean distribution")
    else:
        plt.title(title)
    plt.xlabel(f"mean")
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
        plt.title(f"Max mean distribution")
    else:
        plt.title(title)
    plt.xlabel(f"mean")
    if logscale:
        plt.xscale("log")
        plt.yscale("log")
    plt.ylabel("std")
    plt.savefig(path + name)


def plot_losses(train_losses, valid_losses=None, path="", name="losses.pdf", title="Losses", logscale=True, eval_freq=10):
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

def plot_multi_losses(losses_dict, path="", name="losses.pdf", title="Losses", logscale=True):
    """plots losses during training"""
    plt.clf()
    fig = plt.figure(figsize=(10,5))
    for expe_name, losses in losses_dict.items():
        plt.plot(range(1, len(losses)+1), losses, label=f"{expe_name}")
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
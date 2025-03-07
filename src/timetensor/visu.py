import matplotlib.pyplot as plt
import numpy as np

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

def plot_stats(splits_dict, stat, path="", name="stats.pdf", dim=0):
    """plots stats of datasets"""
    plt.clf()
    fig = plt.figure(figsize=(10,5))
    for split_name, split_dict in splits_dict.items():
        if split_dict.get("values") is not None:
            stat_values, total_stat = get_stats(split_dict["values"], stat, dim)
            bins = np.logspace(-2, 6, 100)
            plt.hist(stat_values, label= f"{split_name} - {stat}={total_stat:.2f}", bins=bins, density=True, alpha=0.5)
        else:
            _, total_stat = get_stats(split_dict["input"], stat, None)
            plt.axvline(x=total_stat, color='red', linestyle='--', linewidth=2)

    plt.legend()
    plt.title(f"{stat} distribution")
    plt.xlabel(f"{stat}")
    plt.xscale("log")
    plt.ylabel("Counts")
    plt.savefig(path + name)


def plot_losses(train_losses, valid_losses=None, path="", name="losses.pdf", title="Losses", logscale=True):
    """plots losses during training"""
    plt.clf()
    fig = plt.figure(figsize=(10,5))
    if valid_losses is not None:
        plt.plot(train_losses, label="train")
        n_evals = len(valid_losses)
        test_freq = len(train_losses) // n_evals
        T = [test_freq * k for k in range(n_evals)] + [len(train_losses)]
        plt.plot(T, valid_losses, label="valid")
    else:
        T = range(1, len(train_losses))
        plt.plot(T, train_losses)
    if logscale:
      plt.yscale('log')
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    fig.tight_layout()
    plt.savefig(path + name)
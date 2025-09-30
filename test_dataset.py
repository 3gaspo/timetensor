import hydra
import logging
import torch
import numpy as np

from src.timetensor.dataset import fetch_training_data
from src.timetensor.utils import unroll_windows
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg):
    logger = logging.getLogger(__name__)
    logger.info("=====Running data script=====")

    #configs
    data_path = cfg.data.path
    lags, horizon = int(cfg.task.lags), int(cfg.task.horizon)
    seed = cfg.misc.seed
    split_kwargs, subset_kwargs = cfg.data.splits, cfg.data.subsets
    logger.info("Fetched configs")

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    loaders_dict, stats_dict, _ = fetch_training_data(data_path, split_kwargs, subset_kwargs, cfg.training.bs, lags, horizon, seed=seed)
    
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
    X, Y, C = unroll_windows(loaders_dict["train"], cap=2000, normal=False, do_context=True)
    features = np.concat((X[:,0,:], Y[:,0,:]), axis=0)
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
    Xn, Yn, Cn = unroll_windows(loaders_dict["train"], cap=2000, normal=True, do_context=True)
    nfeatures = np.concat((Xn[:,0,:], Yn[:,0,:]), axis=0)

    print(X.shape, Y.shape, features.shape)
    print(Xn.shape, Yn.shape, nfeatures.shape)

    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt

    def make_cmap(n):
        base = plt.get_cmap('tab20')  # cyclical palette of 20 colors
        # Repeat if you have >20 clusters
        colors = [base(i % 20) for i in range(n)]
        return ListedColormap(colors)

    from sklearn.manifold import TSNE

    print("Starting tsne")
    tsne = TSNE(n_components=2, random_state=seed)
    red_features = tsne.fit_transform(features)
    print("Done raw")
    tsne = TSNE(n_components=2, random_state=seed)
    print("Done normal")
    red_nfeatures = tsne.fit_transform(nfeatures)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)


    cmap = make_cmap(np.max(list(C[:, 0, 0])+list(Cn[:, 0, 0])))

    #raw
    ax = axes[0]
    clusters = C[:, 0, 0]
    sc = ax.scatter(red_features[:, 0], red_features[:, 1], c=clusters, s=8, alpha=0.9, cmap=cmap)
    ax.set_title("t-SNE of raw data")
    
    #normal
    ax = axes[1]
    clusters_n = Cn[:, 0, 0]
    ax.scatter(red_nfeatures[:, 0], red_nfeatures[:, 1], c=clusters_n, s=8, alpha=0.9, cmap=cmap)
    ax.set_title("t-SNE of normalized data")

    plt.savefig("tsnes.pdf")

    logger.info('End of script\n')

if __name__ == "__main__":
    run()


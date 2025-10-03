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
    logger.info("=====Running tsne script=====")

    #configs
    data_path, data_name = cfg.data.path, cfg.data.dataset
    lags, horizon = int(cfg.task.lags), int(cfg.task.horizon)
    seed = cfg.misc.seed
    split_kwargs, subset_kwargs = cfg.data.splits, cfg.data.subsets
    logger.info("Fetched configs")

    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    by_time=False
    by_indiv=True

    if by_time:
        #loader
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
        loaders_dict, _, _ = fetch_training_data(data_path, split_kwargs, subset_kwargs, cfg.training.bs, lags, horizon, seed=seed, shuffle_eval=True)
        #raw
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
        X, Y, C = unroll_windows(loaders_dict["train"], cap=3000, normal=False, do_context=True)
        Xtest, Ytest, Ctest = unroll_windows(loaders_dict["test1"], cap=3000, normal=False, do_context=True)
        features = np.concat((X[:,0,:], Y[:,0,:]), axis=1)
        featurestest = np.concat((Xtest[:,0,:], Ytest[:,0,:]), axis=1)
        #normal
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
        Xn, Yn, Cn = unroll_windows(loaders_dict["train"], cap=3000, normal=True, do_context=True)
        Xntest, Yntest, Cntest = unroll_windows(loaders_dict["test1"], cap=3000, normal=True, do_context=True)
        nfeatures = np.concat((Xn[:,0,:], Yn[:,0,:]), axis=1)
        nfeaturestest = np.concat((Xntest[:,0,:], Yntest[:,0,:]), axis=1)

    if by_indiv:
        indiv_1 = 0
        indiv_2 = 100

        #loader
        loaders_dict1, _, _ = fetch_training_data(data_path, split_kwargs, subset_kwargs, cfg.training.bs, lags, horizon, seed=seed, shuffle_eval=True, fetch_cluster=indiv_1)
        loaders_dict2, _, _ = fetch_training_data(data_path, split_kwargs, subset_kwargs, cfg.training.bs, lags, horizon, seed=seed, shuffle_eval=True, fetch_cluster=indiv_2)
        #raw
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
        X, Y, C = unroll_windows(loaders_dict1["train"], cap=3000, normal=False, do_context=True)
        Xtest, Ytest, Ctest = unroll_windows(loaders_dict2["train"], cap=3000, normal=False, do_context=True)
        features = np.concat((X[:,0,:], Y[:,0,:]), axis=1)
        featurestest = np.concat((Xtest[:,0,:], Ytest[:,0,:]), axis=1)
        #normal
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
        Xn, Yn, Cn = unroll_windows(loaders_dict1["train"], cap=3000, normal=True, do_context=True)
        Xntest, Yntest, Cntest = unroll_windows(loaders_dict2["train"], cap=3000, normal=True, do_context=True)
        nfeatures = np.concat((Xn[:,0,:], Yn[:,0,:]), axis=1)
        nfeaturestest = np.concat((Xntest[:,0,:], Yntest[:,0,:]), axis=1)

    #tsne
    print("Starting tsne")
    tsne = TSNE(n_components=2, random_state=seed)
    red_features = tsne.fit_transform(features)
    print("Done raw")
    tsne = TSNE(n_components=2, random_state=seed)
    print("Done normal")
    red_nfeatures = tsne.fit_transform(nfeatures)
    tsne = TSNE(n_components=2, random_state=seed)
    red_featurestest = tsne.fit_transform(featurestest)
    print("Done test raw")
    tsne = TSNE(n_components=2, random_state=seed)
    print("Done test normal")
    red_nfeaturestest = tsne.fit_transform(nfeaturestest)


    if by_time:
        labels = ["Train", "Test"]
    if by_indiv:
        labels = [f"Indiv {indiv_1}", f"Indiv {indiv_2}"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    #raw
    ax = axes[0]
    sc = ax.scatter(red_features[:, 0], red_features[:, 1], s=8, alpha=0.9, color="C0", label=labels[0])
    #raw test
    ax = axes[0]
    sc = ax.scatter(red_featurestest[:, 0], red_featurestest[:, 1], s=8, alpha=0.9, color="C1", label=labels[1])
    ax.set_title("t-SNE of raw data")
    ax.legend()
    #normal
    ax = axes[1]
    ax.scatter(red_nfeatures[:, 0], red_nfeatures[:, 1], s=8, alpha=0.9, color="C0", label=labels[0])
    #normal test
    ax = axes[1]
    ax.scatter(red_nfeaturestest[:, 0], red_nfeaturestest[:, 1], s=8, alpha=0.9, color="C1", label=labels[1])
    ax.set_title("t-SNE of normalized data")
    ax.legend()

    if by_time:
        plt.savefig(f"{data_name}_time_tsne.pdf")
    if by_indiv:
        plt.savefig(f"{data_name}_indiv_tsne.pdf")

    # if by_indiv:

    #     def make_cmap(n):
    #         base = plt.get_cmap('tab20')  # cyclical palette of 20 colors
    #         # Repeat if you have >20 clusters
    #         colors = [base(i % 20) for i in range(n)]
    #         return ListedColormap(colors)

    #     fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    #     cmap = make_cmap(np.max(list(Ctest[:, 0, 0])+list(Cntest[:, 0, 0])+list(C[:, 0, 0])+list(Cn[:, 0, 0])))
    #     #raw
    #     ax = axes[0]
    #     clusters = C[:, 0, 0]
    #     sc = ax.scatter(red_features[:, 0], red_features[:, 1], c=clusters, s=20, alpha=0.9, cmap=cmap)
    #     ax.set_title("t-SNE of raw data")
    #     #normal
    #     ax = axes[1]
    #     clusters_n = Cn[:, 0, 0]
    #     ax.scatter(red_nfeatures[:, 0], red_nfeatures[:, 1], c=clusters_n, s=20, alpha=0.9, cmap=cmap)
    #     ax.set_title("t-SNE of normalized data")
        
    #     plt.savefig(f"{data_name}_indiv_tsne.pdf")


    logger.info('End of script\n')

if __name__ == "__main__":
    run()


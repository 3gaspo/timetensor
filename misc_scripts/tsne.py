import hydra
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from src.timetensor.dataset import fetch_training_data
from src.timetensor.utils import unroll_windows, set_seed

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
    output_dir = cfg.misc.output_dir

    logger.info("Fetched configs")
    logger.info(f"Loading {data_name}")

    indiv_split = float(split_kwargs["indiv_split"])
    date_splits = split_kwargs["date_splits"]
    date_splits = [float(split) for split in date_splits.split(";")]

    for mode in ["time", "indiv", "conditional"]:

        if mode == "time":
            #loader
            set_seed(seed)
            loaders_dict, _, _ = fetch_training_data(data_path, split_kwargs, subset_kwargs, cfg.training.bs, lags, horizon, seed=seed, shuffle_eval=True)
            labels_ = ["train", "test1"]
            ratio = round(date_splits[0] / date_splits[2])
            logger.info(f"{ratio}, {date_splits}")
            #raw
            set_seed(seed)
            X, Y = unroll_windows(loaders_dict["train"], cap=ratio*1000, normal=False)
            Xtest, Ytest = unroll_windows(loaders_dict["test1"], cap=1000, normal=False)
            features = X[:,0,:]
            featurestest = Xtest[:,0,:]
            full_features = np.vstack([features, featurestest])
            labels = np.array(["train"] * len(features) + ["test1"] * len(featurestest))

            #normal
            set_seed(seed)
            Xn, Yn = unroll_windows(loaders_dict["train"], cap=ratio*1000, normal=True)
            Xntest, Yntest = unroll_windows(loaders_dict["test1"], cap=1000, normal=True)
            nfeatures = Xn[:,0,:]
            nfeaturestest = Xntest[:,0,:]
            nfull_features = np.vstack([nfeatures, nfeaturestest])
            nlabels = np.array(["train"] * len(nfeatures) + ["test1"] * len(nfeaturestest))

            logger.info(f"debug {X.shape}, {Xtest.shape}")
            logger.info(f"debug {Xn.shape}, {Xntest.shape}")

        elif mode == "indiv":
            #loader
            # indiv_1 = 0
            # indiv_2 = 100
            # loaders_dict1, _, _ = fetch_training_data(data_path, split_kwargs, subset_kwargs, cfg.training.bs, lags, horizon, seed=seed, shuffle_eval=True, fetch_cluster=indiv_1)
            # loaders_dict2, _, _ = fetch_training_data(data_path, split_kwargs, subset_kwargs, cfg.training.bs, lags, horizon, seed=seed, shuffle_eval=True, fetch_cluster=indiv_2)
            loaders_dict, _, _ = fetch_training_data(data_path, split_kwargs, subset_kwargs, cfg.training.bs, lags, horizon, seed=seed, shuffle_eval=True)
            labels_ = ["train", "valid2"]
            ratio = round(1/(1-indiv_split))

            #raw
            set_seed(seed)
            # X, Y, C = unroll_windows(loaders_dict1["train"], cap=3000, normal=False, do_context=True)
            # Xtest, Ytest, Ctest = unroll_windows(loaders_dict2["train"], cap=3000, normal=False, do_context=True)
            X, Y = unroll_windows(loaders_dict["train"], cap=ratio*1000, normal=False)
            Xtest, Ytest = unroll_windows(loaders_dict["valid2"], cap=1000, normal=False)
            features = X[:,0,:]
            featurestest = Xtest[:,0,:]
            full_features = np.vstack([features, featurestest])
            labels = np.array(["train"] * len(features) + ["valid2"] * len(featurestest))

            #normal
            set_seed(seed)
            # Xn, Yn, Cn = unroll_windows(loaders_dict2["train"], cap=3000, normal=False, do_context=True)
            # Xntest, Yntest, Cntest = unroll_windows(loaders_dict2["train"], cap=3000, normal=False, do_context=True)
            Xn, Yn = unroll_windows(loaders_dict["train"], cap=ratio*1000, normal=True)
            Xntest, Yntest = unroll_windows(loaders_dict["valid2"], cap=1000, normal=True)
            nfeatures = Xn[:,0,:]
            nfeaturestest = Xntest[:,0,:]
            nfull_features = np.vstack([nfeatures, nfeaturestest])
            nlabels = np.array(["train"] * len(nfeatures) + ["valid2"] * len(nfeaturestest))

        else:
            #loader
            loaders_dict, _, _ = fetch_training_data(data_path, split_kwargs, subset_kwargs, cfg.training.bs, lags, horizon, seed=seed, shuffle_eval=True)
            labels_ = ["train", "test2"]
            ratio = round(date_splits[0] / (date_splits[1] * (1-indiv_split)))

            #raw
            set_seed(seed)
            X, Y = unroll_windows(loaders_dict["train"], cap=ratio*1000, normal=False)
            Xtest, Ytest = unroll_windows(loaders_dict["test2"], cap=1000, normal=False)
            features = np.concat((X[:,0,:], Y[:,0,:]), axis=1)
            featurestest = np.concat((Xtest[:,0,:], Ytest[:,0,:]), axis=1)
            full_features = np.vstack([features, featurestest])
            labels = np.array(["train"] * len(features) + ["test2"] * len(featurestest))

            #normal
            set_seed(seed)
            Xn, Yn = unroll_windows(loaders_dict["train"], cap=ratio*1000, normal=True)
            Xntest, Yntest = unroll_windows(loaders_dict["test2"], cap=1000, normal=True)
            nfeatures = np.concat((Xn[:,0,:], Yn[:,0,:]), axis=1)
            nfeaturestest = np.concat((Xntest[:,0,:], Yntest[:,0,:]), axis=1)
            nfull_features = np.vstack([nfeatures, nfeaturestest])
            nlabels = np.array(["train"] * len(nfeatures) + ["test2"] * len(nfeaturestest))

        # #tsne
        # print("Starting tsne")
        # tsne = TSNE(n_components=2, random_state=seed)
        # red_features = tsne.fit_transform(features)
        # print("Done raw")
        # tsne = TSNE(n_components=2, random_state=seed)
        # print("Done normal")
        # red_nfeatures = tsne.fit_transform(nfeatures)
        # tsne = TSNE(n_components=2, random_state=seed)
        # red_featurestest = tsne.fit_transform(featurestest)
        # print("Done test raw")
        # tsne = TSNE(n_components=2, random_state=seed)
        # print("Done test normal")
        # red_nfeaturestest = tsne.fit_transform(nfeaturestest)

        #tsne
        logger.info("Starting tsne")
        set_seed(seed)
        tsne = TSNE(n_components=2, random_state=seed)
        red_features = tsne.fit_transform(full_features)
        logger.info("Done raw")
        set_seed(seed)
        tsne = TSNE(n_components=2, random_state=seed)
        red_nfeatures = tsne.fit_transform(nfull_features)
        logger.info("Done normal")


        # if mode == "time":
        #     labels_ = ["Train", "Test"]
        # elif mode == "indiv":
        #     # labels = [f"Indiv {indiv_1}", f"Indiv {indiv_2}"]
        #     labels_ = [f"Indivs Train", f"Indivs Valid"]
        # else:
        #     # labels = [f"Indiv {indiv_1}", f"Indiv {indiv_2}"]
        #     labels_ = [f"Train", f"Test 2"]   

        # fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        # #raw
        # ax = axes[0]
        # sc = ax.scatter(red_features[:, 0], red_features[:, 1], s=8, alpha=0.9, color="C0", label=labels[0])
        # #raw test
        # ax = axes[0]
        # sc = ax.scatter(red_featurestest[:, 0], red_featurestest[:, 1], s=8, alpha=0.9, color="C1", label=labels[1])
        # ax.set_title("t-SNE of raw data")
        # ax.legend()
        # #normal
        # ax = axes[1]
        # ax.scatter(red_nfeatures[:, 0], red_nfeatures[:, 1], s=8, alpha=0.9, color="C0", label=labels[0])
        # #normal test
        # ax = axes[1]
        # ax.scatter(red_nfeaturestest[:, 0], red_nfeaturestest[:, 1], s=8, alpha=0.9, color="C1", label=labels[1])
        # ax.set_title("t-SNE of normalized data")
        # ax.legend()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        #raw
        ax = axes[0]
        for lab, color, alpha in zip(labels_, ["steelblue", "tomato"], [0.9, 0.5]):
            mask = labels == lab
            ax.scatter(
                red_features[mask, 0],
                red_features[mask, 1],
                s=8, alpha=alpha, c=color, label=lab
            )
        ax.set_title("t-SNE of raw data")
        ax.legend()

        #normalized
        ax = axes[1]
        for lab, color, alpha in zip(labels_, ["steelblue", "tomato"], [0.9, 0.5]):
            mask = nlabels == lab
            ax.scatter(
                red_nfeatures[mask, 0],
                red_nfeatures[mask, 1],
                s=8, alpha=alpha, c=color, label=lab
            )        
        ax.set_title("t-SNE of normalized data")
        ax.legend()

        if mode == "time":
            plt.savefig(output_dir + f"{data_name}_time_tsne.pdf")
        elif mode == "indiv":
            plt.savefig(output_dir + f"{data_name}_indiv_tsne.pdf")
        else:
            plt.savefig(output_dir + f"{data_name}_cond_tsne.pdf")

    # if by_indiv:
    # from matplotlib.colors import ListedColormap

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


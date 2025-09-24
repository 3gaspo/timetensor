import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import shutil
import copy
import pandas as pd
import warnings
import json

from .utils import normalize, is_cte
from .analysis import get_dataset_stats

#TODO: vérif que l'aggregation dataste clusters fonctionne bien

class TimeSeriesDataset(Dataset):
    """dataset of multiple individuals"""
    def __init__(self, values, datetimes=None, context=None, lags=336, horizon=24, idx_mode="indiv", context_by_individuals=True, return_all_individuals=True, remove_cte=False, stats=None, weight=1):   
        """
        values (N_individuals, dim_values, dates):  past target values 
        datetimes (dates): list of dates in datetime Y-m-d H:M:S format
        context (N_contexts, dim_context, dates): exogenous variates  e.g N_contexts=1 or N_contexts=N_individuals
        lags (int): size of lookback window
        horizon (int): size of target horizon
        idx_mode (bool): access items by date or individuals
        return_all_individuals (bool): return all individuals or a random
        context_by_individuals(bool): return one context per individual or all
        """
        super().__init__()

        self.values, self.context = values, context
        if len(self.values.shape) == 1:
            self.values = self.values.unsqueeze(0)
        if len(self.values.shape) == 2:
            self.values = self.values.unsqueeze(0)
        if self.context is not None and len(self.context.shape) == 1:
            self.context = self.context.unsqueeze(0)
        if self.context is not None and len(self.context.shape) == 2:
            self.context = self.context.unsqueeze(0)
        self.lags, self.horizon = lags, horizon 
            
        self.individuals, self.dim_values, self.dates = self.values.shape
        if self.context is not None:
            self.contexts, self.dim_context, _dates = self.context.shape
            assert _dates == self.dates, "not the same dates in values and context"        
        assert self.dates > self.lags + self.horizon, "not enough dates for this lag and horizon"
        if datetimes is None:
            self.datetimes = np.array(range(0, self.dates))
        self.datetimes = np.array(datetimes)
        self.idx_mode = idx_mode
        self.return_all_individuals, self.context_by_individuals = return_all_individuals, context_by_individuals
        self.remove_cte = remove_cte
        
        self.stats = stats #global normalization
        if self.stats is not None:
            self.values = normalize(self.values, self.stats["mean"], self.stats["std"])

        self.weight = weight #modulus for get item
        if self.idx_mode == "date":
            self.true_len = self.dates - (self.lags + self.horizon)
        elif self.idx_mode == "indiv":
            self.true_len = self.individuals
        else:
            self.true_len = self.individuals * (self.dates - (self.lags + self.horizon))

    @property
    def shape(self):
        if self.context is not None:
            return (self.individuals, self.dim_values, self.dates), (self.contexts, self.dim_context, self.dates)
        else:
            return (self.individuals, self.dim_values, self.dates)

    def __len__(self):
        return self.weight * self.true_len

    def get_df(self, dim=0):
        return pd.DataFrame(self.values[:, dim, :].transpose(0,1), index=self.datetimes)
    def set_stats(self, stats):
        self.stats = stats
        self.values = normalize(self.values, self.stats["mean"], self.stats["std"])

    def __getitem__(self, raw_idx):
        if self.weight > 1:
            idx = raw_idx % self.true_len
        else:
            idx = raw_idx
        
        remove_cte_counter = 0

        if self.idx_mode == "date":
            if self.return_all_individuals: #1 batch = all individuals, batch of dates
                values = self.values[:, :, idx : idx + self.lags + self.horizon] # (individuals, dim_values, lags+horizon)
                if self.remove_cte:
                    std = values[:, :, :self.lags].std(dim=-1).detach() # (individuals, dim_values, 1)
                    mask = (std > 0).any(dim=1)
                    values = values[mask]
                    while values.numel() == 0:
                        if remove_cte_counter > 100:
                            raise ValueError("Overflow constant windows")
                        idx = np.random.randint(self.dates - (self.lags + self.horizon))
                        values = self.values[:, :, idx : idx + self.lags + self.horizon] # (individuals, dim_values, lags+horizon)
                        std = values[:, :, :self.lags].std(dim=-1).detach() # (individuals, dim_values, 1)
                        mask = (std > 0).any(dim=1)
                        values = values[mask]
                        remove_cte_counter += 1
                        
                if self.context is not None:
                    context = self.context[:, :, idx : idx + self.lags + self.horizon] # (contexts, dim_context, lags+horizon)
                    if self.remove_cte:
                        context = context[mask]

            else: #1 batch = 1 individual, batch of dates
                indiv = np.random.randint(self.individuals)
                values = self.values[indiv, :, idx : idx + self.lags + self.horizon].unsqueeze(0) # (1, dim_values, lags+horizon)
                if self.remove_cte: #skip constant windows
                    while is_cte(values[:, :, :self.lags]):
                        if remove_cte_counter > 100:
                            raise ValueError("Overflow constant windows")
                        indiv = np.random.randint(self.individuals) 
                        values = self.values[indiv, :, idx : idx + self.lags + self.horizon].unsqueeze(0)
                        remove_cte_counter += 1
                if self.context is not None:
                    if self.context_by_individuals:
                        context = self.context[indiv, :, idx : idx + self.lags + self.horizon].unsqueeze(0) # (1, dim_context, lags+horizon)
                    else:
                        context = self.context[:, :, idx : idx + self.lags + self.horizon] # (contexts, dim_context, lags+horizon)

        elif self.idx_mode == "indiv": #1 batch = batch of individuals, random date
            if self.remove_cte and is_cte(self.values[idx, :, :]): #indiv is fully constant
                t = 0
                values = self.values[idx, :, t:self.lags+self.horizon]
            else:
                t = np.random.randint(self.dates - self.lags - self.horizon)
                values = self.values[idx, :, t: t + self.lags + self.horizon].unsqueeze(0) # (1, dim_values, lags+horizon)
                if self.remove_cte: #skip constant windows
                    while is_cte(values[:, :, :self.lags]):
                        if remove_cte_counter > 100:
                            raise ValueError("Overflow constant windows")
                        t = np.random.randint(self.dates - self.lags - self.horizon)
                        values = self.values[idx, :, t: t + self.lags + self.horizon].unsqueeze(0)
                        remove_cte_counter += 1
            if self.context is not None:
                if self.context_by_individuals:
                    context = self.context[idx, :, t: t + self.lags + self.horizon].unsqueeze(0) # (1, dim_context, lags+horizon)
                else:
                    context = self.context[:, :, t: t + self.lags + self.horizon] # (contexts, dim_context, lags+horizon)

        else:
            date, indiv = raw_idx // self.individuals, raw_idx % self.individuals
            values = self.values[indiv, :, date : date + self.lags + self.horizon].unsqueeze(0) # (1, dim_values, lags+horizon)
            while is_cte(values[:, :, :self.lags]):
                if remove_cte_counter > 100:
                        raise ValueError("Overflow constant windows")
                idx = np.random.randint(self.weight * self.true_len)
                date, indiv = idx // self.individuals, idx % self.individuals
                values = self.values[indiv, :, date: date + self.lags + self.horizon].unsqueeze(0)
                remove_cte_counter += 1
            if self.context is not None:
                if self.context_by_individuals:
                    context = self.context[indiv, :, date: date + self.lags + self.horizon].unsqueeze(0) # (1, dim_context, lags+horizon)
                else:
                    context = self.context[:, :, date: date + self.lags + self.horizon] # (contexts, dim_context, lags+horizon)

        inputs = values[:, :, :self.lags] # (individuals, dim, lags)
        target = values[:, :, self.lags:] # (individuals, dim, horizon)
        
        if self.context is not None:
            return inputs, context, target
        else:
            return inputs, target


#TODO relire subset, upgrade avec idx_mode
class TimeSeriesSubset(Dataset):
    def __init__(self, dataset, indices, subset_mode="date"):
        self.indices = indices
        self.mode = subset_mode
        self.lags, self.horizon = dataset.lags, dataset.horizon 

        if self.mode == "indiv":
            if dataset.idx_mode == "date":
                self.dataset = copy.deepcopy(dataset)
                self.dataset.values = self.dataset.values[self.indices]
                self.dataset.individuals = len(self.indices)
                if self.dataset.context is not None and self.dataset.context_by_individual:
                    self.dataset.context = self.dataset.context[self.indices]
            else:
                self.dataset = dataset
            self.individuals = len(indices)
            self.dates = self.dataset.dates
            self.dim_values = self.dataset.dim_values
            if self.dataset.context is not None:
                if self.dataset.context_by_individuals:
                    self.contexts = self.individuals
                else:
                    self.contexts = self.dataset.contexts
            else:
                self.context = None

        elif self.mode == "dates":
            assert len(indices) > self.lags + self.horizon, "not enough dates for this lag and horizon"
            if not dataset.by_date:
                self.dataset = copy.deepcopy(dataset)
                self.dataset.values = self.dataset.values[:, :, self.indices]
                self.dataset.datetimes = self.dataset.datetimes[self.indices]
                self.dataset.dates = len(indices)
            else:
                self.dataset = dataset
            self.individuals = self.dataset.individuals
            self.dates = len(indices)
            self.context = self.dataset.context
            self.dim_values = self.dataset.dim_values

        elif self.mode == "dim":
            self.dataset = dataset
            self.individuals = self.dataset.individuals
            self.dates = self.dataset.dates
            self.context = self.dataset.context
            self.dim_values = len(self.indices)

    def __getitem__(self, idx):
        if self.mode == "dim":
            return self.dataset[idx][:, self.indices, :]
        elif (self.mode=="dates" and self.dataset.by_date) or (self.mode=="individuals" and not self.dataset.by_date):
            return self.dataset[self.indices[idx]]
        else:
            return self.dataset[idx]
        
    def __len__(self):
        if self.dataset.by_date:
            if self.mode == "individuals":
                return len(self.dataset)
            else:
                return len(self.indices) - (self.lags + self.horizon)
        else:
            if self.mode == "individuals":
                return len(self.indices)
            else:
                return len(self.dataset)

    @property
    def shape(self):
        if self.context is not None:
            return (self.individuals, self.dim_values, self.dates), (self.dataset.contexts, self.dataset.dim_context, self.dates)
        else:
            return (self.individuals, self.dim_values, self.dates)

    @property
    def values(self):
        if self.mode == "dim":
            return self.dataset.values[:, self.indices, :]
        elif self.mode=="dates":
            return self.dataset.values[:, :, self.indices]
        elif self.mode=="individuals":
            return self.dataset.values[self.indices, :, :]
    @property
    def datetimes(self):
        if self.mode == "dates":
            return self.dataset.datetimes[self.indices]
        else:
            return self.dataset.datetimes

    def get_df(self, dim=0):
        return pd.DataFrame(self.values[:, 0, :].transpose(0,1), index=self.datetimes)


def fetch_csv(data_path, data_name, context_cols=None, drop=None):
    """fetches univariate csv (optional context) and saves pytorch. TODO: for multivariate"""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Could not infer format, so each element will be parsed individually",
            category=UserWarning,
        )
        df = pd.read_csv(data_path + data_name + ".csv", index_col=0, parse_dates=True)
    if context_cols is None:
        values_df = df
        context_df = None
    else:
        context_df = df[context_cols]
        values_df = df.drop(columns=context_cols)
    values_df.columns = range(values_df.shape[1])
    datetimes = list(df.index)
    if drop:
        drop = drop.split(";")
        drop = [int(idx) for idx in drop]
        values_df = values_df.drop(columns=drop)
        values_df.columns = range(values_df.shape[1])
    return values_df, context_df, datetimes


def build_dataset(data_path, data_name, context_cols=None, drop_users=None, raw_format="csv"):
    """fetches univariate csv (optional context) and saves pytorch. TODO: for multivariate"""
    #load csv
    if raw_format == "csv":
        values_df, context_df, datetimes = fetch_csv(data_path, data_name, context_cols, drop=drop_users)
    else:
        raise ValueError("Unsupported input format")
    
    #tensors
    values_pt = values_df.values
    values_pt = torch.tensor(values_pt, dtype=torch.float32).transpose(1,0).unsqueeze(1) #(individuals, 1, dates)
    if context_cols is not None:
        context_pt = torch.tensor(context_df, dtype=torch.float32).transpose(1,0).unsqueeze(1)
    else:
        context_pt =  torch.tensor([[k for _ in range(values_pt.shape[-1])] for k in range(values_pt.shape[0])]).unsqueeze(dim=1)

    #save
    torch.save(values_pt, data_path + "values.pt")
    torch.save(context_pt, data_path + "context.pt")
    torch.save(datetimes, data_path+ "datetimes.pt")



def load_data(path="datasets/", prefix=""):
    """loads values, context, datetimes from path"""
    if prefix is None:
        prefix = ""
    if prefix != "":
        prefix = prefix + "_"
    values = torch.load(path + prefix + "values.pt")#, weights_only=False)
    if os.path.exists(path + prefix + "context.pt"):
        context = torch.load(path + prefix + "context.pt")#, weights_only=False)
    else:
        context = None
    if len(values.shape) == 1:
        values = values.unsqueeze(0)
    if len(values.shape) == 2:
        values = values.unsqueeze(0)
    if context is not None and len(context.shape) == 1:
        context = context.unsqueeze(0)
    if context is not None and len(context.shape) == 2:
        context = context.unsqueeze(0)
    if os.path.exists(path + prefix + "datetimes.pt"):
        datetimes = np.array(torch.load(path + prefix + "datetimes.pt", weights_only=False))
    else:
        datetimes = np.array(range(values.shape[-1]))
    return values, context, datetimes

def load_example(path="datasets/", prefix=""):
    """loads intput, context, target, indiv, date from path (with eventual prefix)"""
    if prefix is None:
        prefix = ""
    elif prefix != "":
        prefix = prefix + "_"
    inpt = torch.load(path + prefix + "input.pt")
    target = torch.load(path + prefix + "target.pt")
    if os.path.exists(path + prefix + "context.pt"):
        context = torch.load(path + prefix + "context.pt")
    else:
        context = None
    indiv, date = torch.load(path + prefix + "indivdate.pt", weights_only=False)
    return inpt, context, target, indiv, date


#TODO relire avec subset dataset
def get_subset_indices(dataset, ratio, subset_mode=None):
    """returns subset of random indices for dataset"""
    if (subset_mode is None and dataset.by_date) or subset_mode=="dates": #sample dates
        old_len = dataset.dates - dataset.lags - dataset.horizon
        new_len = int(old_len * ratio)
        assert new_len > dataset.lags + dataset.horizon, f"Not enough dates: {old_len} -> {new_len}"
        indices = np.random.choice(old_len, size=new_len, replace=False).tolist()
    elif (subset_mode is None and not dataset.by_date) or subset_mode=="individuals": #sample individuals
        new_len = int(dataset.individuals * ratio)
        assert new_len > 0, "Not enough individuals"
        indices = np.random.choice(dataset.individuals, size=new_len, replace=False).tolist()
    elif subset_mode == "dim":
        assert type(ratio) == list and type(list[0])==int
        return ratio
    else:
        raise ValueError("Unrecognized mode: ", subset_mode)
    return indices


def split_3_way(values, context, datetimes, date_splits):
    """returns dict of train/valid/test of provided values,context,datetimes
    """
    dates = len(datetimes)
    stop_date1, stop_date2 = int(date_splits[0] * dates), int((date_splits[0] + date_splits[1])*dates)
    dates_idx1, dates_idx2, dates_idx3 = list(range(stop_date1)), list(range(stop_date1, stop_date2)), list(range(stop_date2, dates))
    dates1, dates2, dates3 = list(datetimes[:stop_date1]), list(datetimes[stop_date1:stop_date2]), list(datetimes[stop_date2:])    
    if context is not None:
        context1 = context[: , :, dates_idx1]
        context2 = context[: , :, dates_idx2]
        context3 = context[: , :, dates_idx3]
    else:
        context1, context2, context3 = None, None, None
    return {"train": (values[:,:,dates_idx1], context1, dates1), "valid1":(values[:,:,dates_idx2], context2, dates2), "test1":(values[:,:,dates_idx3], context3, dates3)}    


def split_4_way(values, context, datetimes, indiv_split, date_split, context_by_individuals=True, save_path=None, reshuffle=True):
    """returns dict of train/valid/test of provided values,context,datetimes
    split parameters can be in [0,1] or str path to indices
    """
    dates = len(datetimes)
    stop_date = int(date_split * dates)
    dates_idx1, dates_idx2 = list(range(stop_date)), list(range(stop_date, dates))
    dates1, dates2 = list(datetimes[:stop_date]), list(datetimes[stop_date:])
    
    save = (save_path is not None)
    if save:
        split_dir = save_path + str(indiv_split) + ";" + str(date_split) + "/"
    if save and (not reshuffle):
        indices1 = list(torch.load(split_dir + "indiv_split1.pt", weights_only=False))
        indices2 = list(torch.load(split_dir + "indiv_split2.pt", weights_only=False))
    else:
        individuals = values.shape[0]
        stop_indiv = int(indiv_split * individuals)
        indices = np.random.permutation(individuals)
        indices1, indices2 = list(indices[:stop_indiv]), list(indices[stop_indiv:])
        if save:
            if os.path.exists(split_dir):
                shutil.rmtree(split_dir)
            os.makedirs(split_dir)
            torch.save(indices1, split_dir + "indiv_split1.pt")
            torch.save(indices2, split_dir + "indiv_split2.pt")

    values1 = values[indices1, :, :][: , :, dates_idx1]
    values2 = values[indices1, :, :][: , :, dates_idx2]
    values3 = values[indices2, :, :][: , :, dates_idx1]
    values4 = values[indices2, :, :][: , :, dates_idx2]
    if context is not None:
        if context_by_individuals:
            context1 = context[indices1, :, :][: , :, dates_idx1]
            context2 = context[indices1, :, :][: , :, dates_idx2]
            context3 = context[indices2, :, :][: , :, dates_idx1]
            context4 = context[indices2, :, :][: , :, dates_idx2]
        else:
            context1 = context[: , :, dates_idx1]
            context2 = context[: , :, dates_idx2]
            context3 = context[: , :, dates_idx1]
            context4 = context[: , :, dates_idx2]
    else:
        context1, context2, context3, context4 = None, None, None, None
    return {"train":(values1, context1, dates1), "valid":(values2, context2, dates2), "valid2":(values3, context3, dates1), "test": (values4, context4, dates2)}


def split_6_way(values, context, datetimes, indiv_split, date_splits, context_by_individuals=True, save_path=False, reshuffle=True):
    """returns dict of train/valid/test of provided values,context,datetimes
    split parameters can be in [0,1] or str path to indices
    """
    dates = len(datetimes)
    dates = len(datetimes)
    stop_date1, stop_date2 = int(date_splits[0] * dates), int((date_splits[0] + date_splits[1])*dates)
    dates_idx1, dates_idx2, dates_idx3 = list(range(stop_date1)), list(range(stop_date1, stop_date2)), list(range(stop_date2, dates))
    dates1, dates2, dates3 = list(datetimes[:stop_date1]), list(datetimes[stop_date1:stop_date2]), list(datetimes[stop_date2:])
    
    save = (save_path is not None)
    if save:
        split_dir = save_path + str(indiv_split) + ";" + str(date_splits) + "/"
    if save and (not reshuffle):
        indices1 = list(torch.load(split_dir + "indiv_split1.pt", weights_only=False))
        indices2 = list(torch.load(split_dir + "indiv_split2.pt", weights_only=False))
    else:
        individuals = values.shape[0]
        stop_indiv = int(indiv_split * individuals)
        indices = np.random.permutation(individuals)
        indices1, indices2 = list(indices[:stop_indiv]), list(indices[stop_indiv:])
        if save:
            if os.path.exists(split_dir):
                shutil.rmtree(split_dir)
            os.makedirs(split_dir)
            torch.save(indices1, split_dir + "indiv_split1.pt")
            torch.save(indices2, split_dir + "indiv_split2.pt")

    values1 = values[indices1, :, :][: , :, dates_idx1]
    values2 = values[indices1, :, :][: , :, dates_idx2]
    values3 = values[indices1, :, :][: , :, dates_idx3]
    values4 = values[indices2, :, :][: , :, dates_idx1]
    values5 = values[indices2, :, :][: , :, dates_idx2]
    values6 = values[indices2, :, :][: , :, dates_idx3]
    if context is not None:
        if context_by_individuals:
            context1 = context[indices1, :, :][: , :, dates_idx1]
            context2 = context[indices1, :, :][: , :, dates_idx2]
            context3 = context[indices1, :, :][: , :, dates_idx3]
            context4 = context[indices2, :, :][: , :, dates_idx1]
            context5 = context[indices2, :, :][: , :, dates_idx2]
            context6 = context[indices2, :, :][: , :, dates_idx3]
    
        else:
            context1 = context[:, :, :][: , :, dates_idx1]
            context2 = context[:, :, :][: , :, dates_idx2]
            context3 = context[:, :, :][: , :, dates_idx3]
            context4 = context[:, :, :][: , :, dates_idx1]
            context5 = context[:, :, :][: , :, dates_idx2]
            context6 = context[:, :, :][: , :, dates_idx3]
    else:
        context1, context2, context3, context4, context5, context6 = None, None, None, None, None, None
    dico = {
        "train":(values1, context1, dates1),
        "valid1":(values2, context2, dates2),
        "valid2":(values4, context4, dates1),
        "valid3": (values5, context5, dates2),
        "test1": (values3, context3, dates3),
        "test2": (values6, context6, dates3)
        }
    return dico


# def get_dataset_splits(data_path="datasets/", indiv_split=None, date_splits=None, context_by_individuals=True, save_path=None, reshuffle=True, cluster_path=None, set_cluster=None, data=None):
def get_dataset_splits(splits, data_path=None, save_path=None, cluster_path=None, set_cluster=None, data=None):
    """splits data from path. If str splits, will load given split, if float will save new split"""
    context_by_indiv, reshuffle = splits.context_by_individuals, splits.reshuffle 
    date_splits, indiv_split = splits.date_splits, splits.indiv_split

    #load whole data
    if data is None:
        values, context, datetimes = load_data(data_path) #load dataset
    else:
        values, context, datetimes = data
    
    #filter values at cluster path
    if cluster_path is not None:
        indices = list(torch.load(cluster_path, weights_only=False))
        values = values[indices]
        if context is not None and context_by_indiv:
            context = context[indices]
        if set_cluster is not None:
            context = torch.tensor([set_cluster for _ in range(len(indices))]).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, values.shape[1], values.shape[2])

    if type(date_splits) == str:
        date_splits = date_splits.split(";")
        date_splits = [float(txt) for txt in date_splits]
    if indiv_split is None or values.shape[0]==1:
        type_split = 3
    elif len(date_splits) == 2:
        type_split = 4
    else:
        type_split = 6
    if type_split == 3:
        data_dict = split_3_way(values, context, datetimes, date_splits)
    elif type_split == 4:
        data_dict = split_4_way(values, context, datetimes, indiv_split, date_splits[0], context_by_indiv, save_path, reshuffle=reshuffle)
    elif type_split == 6:
        data_dict = split_6_way(values, context, datetimes, indiv_split, date_splits, context_by_indiv, save_path, reshuffle=reshuffle)
    else:
        raise ValueError(f"Unrecognized type_split: {type_split}")

    return data_dict



def get_train_loaders(data_dict, batch_size, lags, horizon, splits, subsets, save_path=None, stats=None):
    """returns dataloaders from data_dict as eventual subsets"""
    subset_mode, subsets  = subsets.mode, subsets.sizes
    idx_mode = splits.idx_mode
    reshuffle, context_by_indiv = splits.reshuffle, splits.context_by_individuals
    remove_train_cte, remove_eval_cte = splits.remove_train_cte, splits.remove_eval_cte
    
    if subsets is not None:
        subsets = [float(txt) for txt in subsets.split(";")]
        assert len(subsets) == len(data_dict)
    else:
        subsets = [1 for _ in range(len(data_dict))]
    save = (save_path is not None)
    if save:
        subset_dir = save_path + subset_mode + str(subsets) + "/"
    loaders_dict = {}

    for i, (key, (values, context, datetimes)) in enumerate(data_dict.items()):
        if key == "train":
            if values.shape[0]==1 and batch_size>1:
                weight=batch_size
            else:
                weight=1
            dataset = TimeSeriesDataset(values, datetimes, context, lags, horizon, idx_mode, context_by_indiv, remove_cte=remove_train_cte, weight=weight, stats=stats)
        else:
            dataset = TimeSeriesDataset(values, datetimes, context, lags, horizon, idx_mode="all", context_by_individuals=context_by_indiv, remove_cte=remove_eval_cte, stats=stats)

        subset = subsets[i]
        if subset != 1:
            if save and (not reshuffle):
                subset_indices = list(torch.load(subset_dir + f"{key}_subset.pt", weights_only=False))
            else:
                subset_indices = get_subset_indices(dataset, subset, subset_mode)
            if save:
                if os.path.exists(subset_dir):
                    shutil.rmtree(subset_dir)
                os.makedirs(subset_dir)
                torch.save(subset_indices, subset_dir + f"{key}_subset.pt")
            dataset = TimeSeriesSubset(dataset, subset_indices, subset_mode)

        if key =="train":
            local_collate_fn = lambda x: collate_fn(x, remove_cte=remove_train_cte)
            loaders_dict[key] = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=local_collate_fn)
        else:
            local_collate_fn = lambda x: collate_fn(x, remove_cte=remove_eval_cte)
            # effective_bs = max(int(batch_size // values.shape[0]),1)
            loaders_dict[key] = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=local_collate_fn)
       
    return loaders_dict


def collate_fn(data, remove_cte=False):
    """
       data: is a list of tuples with (input, (context), target)
    """
    if len(data[0]) == 3:
        inputs, contexts, targets = zip(*data)
    else:
        inputs, targets = zip(*data)
        contexts = None

    inputs = torch.cat(inputs, dim=0)   # shape: (bs*individuals, dim, lookback)

    if contexts is not None:
        contexts = torch.cat(contexts, dim=0)
    targets = torch.cat(targets, dim=0)   # shape: (bs*individuals, dim, horiz)

    if remove_cte: #remove constant windows
        stds = inputs.std(dim=-1) #(bs * indiv, dim)
        non_constant_mask = (stds > 0).any(dim=1)  # (bs * indiv)
        inputs, targets = inputs[non_constant_mask], targets[non_constant_mask]
        if contexts is not None:
            contexts = contexts[non_constant_mask]

    return inputs, contexts, targets
    
def aggregate_loaders_dict(loaders_dicts, lags, horizon, splits, batch_size):
    """aggregates loaders of different individuals. Expects same dates."""
    loaders_dict = {}
    keys = list(loaders_dicts[0].keys())
    idx_mode, context_by_individuals = splits.idx_mode, splits.context_by_individuals
    remove_train_cte, remove_test_cte = splits.remove_train_cte, splits.remove_eval_cte

    for key in keys:
        # shuffle = isinstance(loaders_dicts[0][key].sampler, torch.utils.data.RandomSampler)
        # collate_fn = loaders_dicts[0][key].collate_fn
        if key =="train":
            remove_cte = remove_train_cte
            local_collate_fn = lambda x: collate_fn(x, remove_cte=remove_cte)
            shuffle = True
            idx_mode_ = idx_mode
            effective_bs = batch_size

        else:
            remove_cte = remove_test_cte
            local_collate_fn = lambda x: collate_fn(x, remove_cte=remove_cte)
            shuffle = False
            idx_mode_ = "all"
            # effective_bs = max(int(batch_size // values.shape[0]),1)
            effective_bs = batch_size

        datetimes = loaders_dicts[0][key].dataset.datetimes
        if context_by_individuals:
            context_list = []
        else:
            context = loaders_dicts[0][key].dataset.context
        values_list = []
        for new_dict in loaders_dicts:
            values = new_dict[key].dataset.values
            values_list.append(values)
            if context_by_individuals:
                context = new_dict[key].dataset.context
                context_list.append(context)
        if context_by_individuals:
            if context_list[0] is None:
                context = None
            else:
                context = torch.cat(context_list, dim=0)
        extended_dataset = TimeSeriesDataset(torch.cat(values_list, dim=0), datetimes, context, lags, horizon, idx_mode_, context_by_individuals, remove_cte=remove_cte, stats=None)
        extended_loader = DataLoader(extended_dataset, batch_size=effective_bs, shuffle=shuffle, collate_fn=local_collate_fn)
        loaders_dict[key] = extended_loader
    return loaders_dict


def get_sizes(loaders_dict, str_info=False):
    """get data size from loaders"""
    X, c, y = next(iter(loaders_dict["train"])) # (indiv, dim, lags),  #(nc, dim, horizon),  #(indiv, dim, horizon)
    shape = [X.shape[2], X.shape[1], y.shape[2]] #lags, dim, horizon
    if not str_info:
        return shape
    else:
        shapes = {key: loaders_dict[key].dataset.shape for key in loaders_dict}
        shape_str = "Splits shapes:\n" + "\n".join(f"{k}\t{v}" for k, v in shapes.items())        
        if c is not None:
            batch_str = f"Batches:\n X={list(X.shape)}\n c={list(c.shape)}\n y={list(y.shape)}"
        else:
            batch_str = f"Batches:\n X={list(X.shape)}\n y={list(y.shape)}"

        return shape, shape_str, batch_str


def fetch_training_data(data_path, splits, subsets, batch_size, lags, horizon, clusters=None, aggregate=True, seed=None, save=False):
    """returns loaders dict (clusters=> nested dict)"""
    if seed is not None: 
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    #save paths
    if save:
        save_path = data_path
    else:
        save_path = None
    if clusters is not None:
        cluster_path = data_path + clusters + "/"
        if save:
            save_path += clusters + "/" 

    if (clusters is not None) and (subsets.cluster is None): #clustered splits
        cluster_names = [name for name in os.listdir(cluster_path) if name[-3:]==".pt"]
        loaders_dicts, nodes_stats_dict = [], {}
        for k, cluster_name in enumerate(cluster_names):
            if save:
                split_path = save_path+cluster_name[:-3]+"splits/"
                subset_path = save_path+cluster_name[:-3]+"subsets/"
            else:
                split_path, subset_path = None, None
            cluster_path_ = cluster_path+cluster_name
            data_dict = get_dataset_splits(splits, data_path, split_path, cluster_path_, set_cluster=k)
            loaders_dict = get_train_loaders(data_dict, batch_size, lags, horizon, splits, subsets, subset_path)
            loaders_dicts.append(loaders_dict)

            node_dict = {subkey: loader.dataset.get_df() for subkey, loader in loaders_dict.items()}
            if save:
                save_path = save_path+cluster_name[:-3] + "/"
            nodes_stats_dict[f"node{k}"] = get_dataset_stats(node_dict, lags, horizon, splits["remove_train_cte"], splits["remove_eval_cte"], save_path)
        
        if aggregate:
            loaders_dict = aggregate_loaders_dict(loaders_dicts, lags, horizon, splits, batch_size)
            df_dict = {key: loader.dataset.get_df() for key, loader in loaders_dict.items()}
            stats_dict = get_dataset_stats(df_dict, lags, horizon, splits["remove_train_cte"], splits["remove_eval_cte"], save_path)
        else:
            loaders_dict = {f"node{k}": loaders_dicts[k] for k in range(len(loaders_dicts))}
            stats_dict = None
        return loaders_dict, stats_dict, nodes_stats_dict

    else: #1 split
        if subsets["cluster"] is not None:
            cluster_name = subsets["cluster"]
            cluster_path += cluster_name + ".pt"
            if save:
                split_path = save_path+cluster_name[:-3]+"splits/"
                subset_path = save_path+cluster_name[:-3]+"subsets/"
            else:
                split_path, subset_path = None, None
            data_dict = get_dataset_splits(splits, data_path, split_path, cluster_path)
            loaders_dict = get_train_loaders(data_dict, batch_size, lags, horizon, splits, subsets, subset_path)
        else:
            if save:
                split_path = save_path + "splits/"
                subset_path = save_path+ "subsets/"
            else:
                split_path, subset_path = None, None
            data_dict = get_dataset_splits(splits, data_path, split_path)
            loaders_dict = get_train_loaders(data_dict, batch_size, lags, horizon, splits, subsets, subset_path)

        df_dict = {key: loader.dataset.get_df() for key, loader in loaders_dict.items()}
        stats_dict = get_dataset_stats(df_dict, lags, horizon, splits["remove_train_cte"], splits["remove_eval_cte"], save_path)
        return loaders_dict, stats_dict, None


def apply_stats(loaders_dict, stats_dict):
    """apply global normalization to loaders using stats_dict"""
    for key, loader in loaders_dict.items():
        loader.dataset.set_stats(stats_dict[key])


def set_random_data(path="datasets/", lag=168, horizon=24, name="rand", context_by_individuals=True, prefix=""):
    """gets a random individual and random window from dataset"""
    values, context, datetimes = load_data(path, prefix)

    individuals, dim, dates = values.shape
    rand_indiv = np.random.randint(individuals)
    rand_date = np.random.randint(dates - (lag + horizon))

    inputs = values[rand_indiv, :, rand_date : rand_date+lag]
    target = values[rand_indiv, :, rand_date+lag : rand_date+lag+horizon]
    if context is not None:
        if context_by_individuals:
            context = context[rand_indiv, :, rand_date : rand_date+lag+horizon]
        else:
            context = context[:, :, rand_date : rand_date+lag+horizon]
    
    ex_dir = path + "examples/" + f"{lag}_{horizon}/" + name + "/"
    if not os.path.exists(ex_dir):
        os.makedirs(ex_dir)
    torch.save(inputs, ex_dir + "input.pt")
    if context is not None:
        torch.save(context, ex_dir + "context.pt")
    torch.save(target, ex_dir + "target.pt")
    torch.save((rand_indiv, datetimes[rand_date]), ex_dir + "indivdate.pt")


def fetch_example_data(path="datasets/examples/", names=None):
    """fetches example data"""
    if names is None:
        names = [name for name in os.listdir(path)]
    elif type(names) == str:
        return load_example(path + names + "/")
    dico = {}
    for name in names:
        dico[name] = load_example(path + name + "/")
    return dico


def fetch_stats(data_path, clusters, normalization, subsets):
    """returns correct stats dict"""
    if normalization == "cmIN":
        assert clusters is not None
        #total stats
        stats_path = data_path + "raw_stats.json"
        with open(stats_path) as file:
            stats_dict = json.load(file)
        stats_dict["train"]["alpha"], stats_dict["train"]["beta"] = [], []
        #cluster stats
        cluster_names = [name[:-3] for name in os.listdir(data_path + clusters) if name[-3:]==".pt"]
        for cluster_name in cluster_names:
            stats_path = data_path + clusters + "stats/" + cluster_name + "_raw_stats.json"
            with open(stats_path) as file:
                stats_dict_ = json.load(file)
            stats_dict["train"]["alpha"].append(stats_dict_["train"]["alpha"])
            stats_dict["train"]["beta"].append(stats_dict_["train"]["beta"])
    else:
        if subsets["cluster"] is not None:
            stats_path = data_path + clusters + "stats/" + subsets["cluster"] + "_raw_stats.json"
        else:
            stats_path = data_path + "raw_stats.json"
        with open(stats_path) as file:
            stats_dict = json.load(file)
        #TODO stats_dict with means,std aggregate when clusters is not None
    return stats_dict
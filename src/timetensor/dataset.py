import torch
import numpy as np
import os
import shutil
import copy
import pandas as pd
import warnings

from torch.utils.data import Dataset, DataLoader

from .utils import normalize, is_cte, set_seed
from .analysis import get_dataset_stats



class IndexSampler:
    def __init__(self, values, lags, horizon, idx_mode="random", return_all_individuals=False, do_context=True, context_by_individuals=False, remove_cte=False, weight=1, subset_indices=None, subset_mode="dates"):
        """sampler to fetch indiv and date indices for dataset
        idx_mode: what idx corresponds to
        subset_mode: what subsets_indices correspond to
        """
        self.values = values
        self.individuals, self.dim_values, self.dates = self.values.shape
        self.lags, self.horizon = lags, horizon
        self.max_dates = self.dates - (self.lags + self.horizon) + 1
        self.idx_mode, self.return_all_individuals = idx_mode, return_all_individuals
        self.do_context, self.context_by_individuals = do_context, context_by_individuals
        self.remove_cte = remove_cte
        self.weight = weight
        self.subset_indices, self.subset_mode = subset_indices, subset_mode

    def true_len(self):
        if self.idx_mode == "dates":
            if self.subset_indices is not None and self.subset_mode == "dates":
                return len(self.subset_indices)
            return self.max_dates
        elif self.idx_mode == "individuals":
            if self.subset_indices is not None and self.subset_mode == "individuals":
                return len(self.subset_indices)
            return self.individuals
        elif self.idx_mode == "all":
            if self.subset_indices is not None:
                if self.subset_mode == "dates":
                    return self.individuals * len(self.subset_indices)
                elif self.subset_mode == "individuals":
                    return len(self.subset_indices) * self.max_dates
                elif self.subset_mode == "all":
                    return len(self.subset_indices)
            return self.individuals * self.max_dates
        elif self.idx_mode == "random":
            return 1
        else:
            raise ValueError(f"Unrecognized idx_mode: {self.idx_mode}")

    def __len__(self):
        return self.weight * self.true_len()

    def get_mask(self, values, lags, indivs, date):
        """return mask of indiv with non constant lookback"""
        lookbacks = values[indivs, :, date: date + lags] #(individuals, dim_values, lags)
        mask = (lookbacks.std(dim=-1) > 0).any(dim=1)
        return mask
    
    def __call__(self, raw_idx):
        idx = raw_idx % self.true_len()
        
        if self.idx_mode == "dates":
            if self.subset_indices is not None and self.subset_mode == "dates":
                date = self.subset_indices[idx]
                assert date < self.max_dates
            else:
                date = idx
            if self.return_all_individuals: #1 batch = all individuals * batch of dates
                if self.subset_indices is not None and self.subset_mode == "individuals":
                    indivs = self.subset_indices
                else:
                    indivs = slice(None)
                if self.remove_cte:
                    remove_cte_counter = 0
                    mask = self.get_mask(self.values, self.lags, indivs, date)
                    while mask.sum().item() == 0: #search for new date
                        if self.subset_indices is not None and self.subset_mode == "dates":
                            date = np.random.choice(self.subset_indices)
                        else:
                            date = np.random.randint(self.max_dates)
                        remove_cte_counter += 1
                        if remove_cte_counter > 100:
                            raise ValueError("Overflow constant windows")
                        mask = self.get_mask(self.values, self.lags, indivs, date)

            else: #1 batch = 1 individual, batch of dates
                if self.subset_indices is not None and self.subset_mode == "individuals":
                    indivs = [np.random.choice(self.subset_indices)]
                else:
                    indivs = [np.random.randint(self.individuals)]
                if self.remove_cte:
                    remove_cte_counter = 0
                    mask = self.get_mask(self.values, self.lags, indivs, date)
                    while mask.sum().item() == 0: #search for new indiv
                        if self.subset_indices is not None and self.subset_mode == "individuals":
                            indivs = [np.random.choice(self.subset_indices)]
                        else:
                            indivs = [np.random.randint(self.individuals)]
                        remove_cte_counter += 1
                        if remove_cte_counter >= 100:
                            remove_cte_counter_ = 0
                            while mask.sum().item() == 0: #search for new date
                                if self.subset_indices is not None and self.subset_mode == "dates":
                                    date = np.random.choice(self.subset_indices)
                                else:
                                    date = np.random.randint(self.max_dates)
                                remove_cte_counter_ += 1
                                if remove_cte_counter_ > 100:
                                    raise ValueError("Overflow constant windows")
                                mask = self.get_mask(self.values, self.lags, indivs, date)
                        mask = self.get_mask(self.values, self.lags, indivs, date)

        elif self.idx_mode == "individuals": #1 batch = batch of individuals, random date
            if self.subset_indices is not None:
                if self.subset_mode == "dates":
                    indivs, date = [idx], np.random.choice(self.subset_indices)
                elif self.subset_mode == "individuals":
                    indivs, date = [self.subset_indices[idx]], np.random.randint(self.max_dates)
            else:
                indivs, date = [idx], np.random.randint(self.max_dates)
            if self.remove_cte:
                if is_cte(self.values[indivs, :, :]): #indiv is fully constant
                    remove_cte_counter = 0
                    mask = self.get_mask(self.values, self.lags, indivs, date)
                    while mask.sum().item() == 0: #search for new indiv and date
                        if self.subset_indices is not None:
                            if self.subset_mode == "dates":
                                indivs, date =  [np.random.randint(self.individuals)], np.random.choice(self.subset_indices)
                            elif self.subset_mode == "individuals":
                                indivs, date = [np.random.choice(self.subset_indices)], np.random.randint(self.max_dates)
                        else:
                            indivs, date = [idx], np.random.randint(self.max_dates)
                        remove_cte_counter += 1
                        if remove_cte_counter > 100:
                            raise ValueError("Overflow constant windows")
                        mask = self.get_mask(self.values, self.lags, indivs, date)
                else:
                    remove_cte_counter = 0
                    mask = self.get_mask(self.values, self.lags, indivs, date)
                    while mask.sum().item() == 0: #search for new date
                        if self.subset_indices is not None:
                            if self.subset_mode == "dates":
                                indivs, date =  [idx], np.random.choice(self.subset_indices)
                            elif self.subset_mode == "individuals":
                                indivs, date = [self.subset_indices[idx]], np.random.randint(self.max_dates)
                        else:
                            date = np.random.randint(self.max_dates)
                        remove_cte_counter += 1
                        if remove_cte_counter > 100:
                            raise ValueError("Overflow constant windows")
                        mask = self.get_mask(self.values, self.lags, indivs, date)
                
        elif self.idx_mode == "all":
            if self.subset_indices is not None:
                if self.subset_mode == "dates":
                    indivs, date = [idx % self.individuals], idx // self.individuals
                    date = self.subset_indices[date]
                    assert date + self.lags + self.horizon <= self.dates
                elif self.subset_mode == "individuals":
                    indiv, date = idx % len(self.subset_indices), idx // len(self.subset_indices)
                    indivs = [self.subset_indices[indiv]]
                elif self.subset_mode == "all":
                    idx = self.subset_indices[idx]
                    indivs, date = [idx % self.individuals], idx // self.individuals
            else:
                indivs, date = [idx % self.individuals], idx // self.individuals
            if self.remove_cte:
                remove_cte_counter = 0
                mask = self.get_mask(self.values, self.lags, indivs, date)
                while mask.sum().item() == 0: #search for new date
                    idx = np.random.randint(self.true_len())
                    if self.subset_indices is not None:
                        if self.subset_mode == "dates":
                            indivs, date = [idx % self.individuals], idx // self.individuals
                            date = self.subset_indices[date]
                            assert date + self.lags + self.horizon <= self.dates
                        elif self.subset_mode == "individuals":
                            indiv, date = idx % len(self.subset_indices), idx // len(self.subset_indices)
                            indivs = [self.subset_indices[indiv]]
                        elif self.subset_mode == "all":
                            idx = self.subset_indices[idx]
                            indivs, date = [idx % self.individuals], idx // self.individuals
                    else:
                        indivs, date = [idx % self.individuals], idx // self.individuals
                    remove_cte_counter += 1
                    if remove_cte_counter > 100:
                        raise ValueError("Overflow constant windows")
                    mask = self.get_mask(self.values, self.lags, indivs, date)

        elif self.idx_mode == "random":
            if self.subset_indices is not None:
                if self.subset_mode == "dates":
                    indivs, date = [np.random.randint(self.individuals)], np.random.choice(self.subset_indices)
                elif self.subset_mode == "individuals":
                    indivs, date = [np.random.choice(self.subset_indices)], np.random.randint(self.max_dates)
            else:
                indivs, date = [np.random.randint(self.individuals)], np.random.randint(self.max_dates)
            if self.remove_cte:
                remove_cte_counter = 0
                mask = self.get_mask(self.values, self.lags, indivs, date)
                while mask.sum().item() == 0: #search for new date
                    if self.subset_indices is not None:
                        if self.subset_mode == "dates":
                            indivs, date = [np.random.randint(self.individuals)], np.random.choice(self.subset_indices)
                        elif self.subset_mode == "individuals":
                            indivs, date = [np.random.choice(self.subset_indices)], np.random.randint(self.max_dates)
                    else:
                        indivs, date = [np.random.randint(self.individuals)], np.random.randint(self.max_dates)
                    remove_cte_counter += 1
                    if remove_cte_counter > 100:
                        raise ValueError("Overflow constant windows")
                    mask = self.get_mask(self.values, self.lags, indivs, date)

        else:
            raise ValueError(f"Unrecognized idx_mode: {self.idx_mode}")

        if self.do_context:
            if self.context_by_individuals:
                context_idx = indivs
            else:
                context_idx = slice(None)
        else:
            context_idx = None

        return indivs, date, context_idx


class TimeSeriesDataset(Dataset):
    """dataset of multiple individuals"""
    def __init__(self, values, datetimes=None, context=None, lags=168, horizon=24, do_context=True):   
        """
        values (N_individuals, dim_values, dates):  past target values 
        datetimes (dates): list of dates in datetime Y-m-d H:M:S format
        context (N_contexts, dim_context, dates): exogenous variates  e.g N_contexts=1 or N_contexts=N_individuals
        lags (int): size of lookback window
        horizon (int): size of target horizon
        idx_mode (str): access items by date or individuals, or both
        return_all_individuals (bool): return all individuals or a random
        context_by_individuals(bool): return one context per individual or all
        """
        super().__init__()

        self.values, self.context = values, context
        if len(self.values.shape) == 1: #1 user, 1 variate
            self.values = self.values.unsqueeze(0)
        if len(self.values.shape) == 2: #1 user, many variates
            self.values = self.values.unsqueeze(0)
        self.individuals, self.dim_values, self.dates = self.values.shape
        self.lags, self.horizon = lags, horizon 
        assert self.dates >= self.lags + self.horizon, f"not enough dates for this lag and horizon: {self.dates} with {self.lags}-{self.horizon}"

        if datetimes is None:
            self.datetimes = np.array(range(0, self.dates))
        else:
            self.datetimes = np.array(datetimes)

        self.do_context = do_context
        if self.do_context:
            if self.context is None:
                self.context = torch.tensor([[k for _ in range(self.values.shape[-1])] for k in range(self.values.shape[0])]).unsqueeze(dim=1)
            if len(self.context.shape) == 1:
                self.context = self.context.unsqueeze(0)
            if len(self.context.shape) == 2:
                self.context = self.context.unsqueeze(0)
            self.contexts, self.dim_context, self.context_dates = self.context.shape
            assert self.context_dates == self.dates, "not the same dates in values and context"
        else:
            self.contexts, self.dim_context, self.context_dates = 0, 0, 0

        self.index_sampler = IndexSampler(self.values, self.lags, self.horizon) #default random sampler

    @property
    def shape(self):
        return (self.individuals, self.dim_values, self.dates), (self.contexts, self.dim_context, self.dates)

    def __len__(self):
        return len(self.index_sampler)

    def get_df(self, dim=0):
        df = pd.DataFrame(self.values[:, dim, :].T, index=self.datetimes)
        df.columns = [f"user_{k}" for k in range(df.shape[1])]
        return df
        
    def normalize(self, standard_stats):
        """normalizes values using provided stats"""
        self.standard_stats = standard_stats
        self.values = normalize(self.values, self.standard_stats["mean"], self.standard_stats["std"])
    
    def set_sampler(self, **kwargs): #idx_mode="random", return_all_individuals=False, context_by_individuals=False, remove_cte=False, weight=1, subset_indices=None, subset_mode="dates"):
        """updates default sampler for special indexing and subsets"""
        for key, value in kwargs.items():
            if not hasattr(self.index_sampler, key):
                raise AttributeError(f"IndexSampler has no attribute '{key}'")
            setattr(self.index_sampler, key, value)

    def __getitem__(self, idx):                
        indivs, date, context_idx = self.index_sampler(idx)
        values = self.values[indivs, :, date : date + self.lags + self.horizon] # (individuals, dim_values, lags+horizon)
        inputs = values[:, :, :self.lags] # (individuals, dim, lags)
        target = values[:, :, self.lags:] # (individuals, dim, horizon)
        
        context = None
        if self.do_context:
            context = self.context[context_idx, :, date : date + self.lags + self.horizon]
        
        return inputs, context, target, indivs, date


def fetch_csv(data_path, data_name, context_cols=None, drop_users=None):
    """fetches data csv (optional context) and returns dataframe"""
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
    values_df.columns = [f"user_{k}" for k in range(values_df.shape[1])] #range(values_df.shape[1]) 
    datetimes = list(df.index)
    if drop_users:
        drop = drop_users.split(";")
        drop = [f"user_{int(idx)}" for idx in drop]
        values_df = values_df.drop(columns=drop)
        values_df.columns = [f"user_{k}" for k in range(values_df.shape[1])]
    return values_df, context_df, datetimes


def build_dataset(data_path, data_name, context_cols=None, drop_users=None, do_context=True, raw_format="csv"):
    """builds pytorch tensors from csv path"""
    #load csv
    if raw_format == "csv":
        values_df, context_df, datetimes = fetch_csv(data_path, data_name, context_cols, drop_users=drop_users)
    else:
        raise ValueError("Unsupported input format")
    
    #tensors
    values_pt = values_df.values
    values_pt = torch.tensor(values_pt, dtype=torch.float32).transpose(1,0).unsqueeze(1) #(individuals, 1, dates)
    torch.save(values_pt, data_path + "values.pt")
    torch.save(datetimes, data_path+ "datetimes.pt")

    #context
    context_pt = None
    if context_cols is not None:
        context_pt = torch.tensor(context_df, dtype=torch.float32).transpose(1,0).unsqueeze(1)
    elif do_context:
        context_pt =  torch.tensor([[k for _ in range(values_pt.shape[-1])] for k in range(values_pt.shape[0])]).unsqueeze(dim=1)
    if context_pt is not None:
        torch.save(context_pt, data_path + "context.pt")



def load_data(path="datasets/", prefix=""):
    """loads values, context, datetimes from path"""
    if prefix is None:
        prefix = ""
    if prefix != "":
        prefix = prefix + "_"
    values = torch.load(path + prefix + "values.pt")
            
    if len(values.shape) == 1:
        values = values.unsqueeze(0)
    if len(values.shape) == 2:
        values = values.unsqueeze(0)

    if os.path.exists(path + prefix + "datetimes.pt"):
        datetimes = np.array(torch.load(path + prefix + "datetimes.pt", weights_only=False))
    else:
        datetimes = np.array(range(values.shape[-1]))

    context=None
    if os.path.exists(path + prefix + "context.pt"):
        context = torch.load(path + prefix + "context.pt")
    if context is not None:
        if len(context.shape) == 1:
            context = context.unsqueeze(0)
        if len(context.shape) == 2:
            context = context.unsqueeze(0)
    
    return values, context, datetimes

def load_example(path="datasets/", prefix=""):
    """loads intput, context, target, indiv, date from path (with eventual prefix)"""
    if prefix is None:
        prefix = ""
    elif prefix != "":
        prefix = prefix + "_"
    inpt = torch.load(path + prefix + "input.pt")
    target = torch.load(path + prefix + "target.pt")
    
    context=None
    if os.path.exists(path + prefix + "context.pt"):
        context = torch.load(path + prefix + "context.pt")
    indiv, date = torch.load(path + prefix + "indivdate.pt", weights_only=False)
    return inpt, context, target, indiv, date


def get_subset_indices(dates, individuals, lags, horizon, ratio, subset_mode):
    """returns subset of random indices for dataset"""
    if subset_mode=="dates": #sample dates
        old_len = dates - (lags + horizon) +1 
        new_len = int(old_len * ratio)
        assert new_len >= lags + horizon, f"Not enough dates: {old_len} -> {new_len}"
        indices = np.random.choice(old_len, size=new_len, replace=False).tolist()
    elif subset_mode=="individuals": #sample individuals
        new_len = int(individuals * ratio)
        assert new_len > 0, "Not enough individuals"
        indices = np.random.choice(individuals, size=new_len, replace=False).tolist()
    else:
        raise ValueError("Unrecognized mode: ", subset_mode)
    return indices


def split_1_way(values, context, datetimes):
    """returns dict of train/valid/test of provided values,context,datetimes
    """
    return {"test1": (values, context, datetimes)}    

def split_2_way(values, context, datetimes, date_splits):
    """returns dict of train/valid/test of provided values,context,datetimes
    """
    dates = len(datetimes)
    stop_date1 = int(date_splits[0] * dates)
    dates_idx1, dates_idx2 = list(range(stop_date1)), list(range(stop_date1, dates))
    dates1, dates2 = list(datetimes[:stop_date1]), list(datetimes[stop_date1::])    
    if context is not None:
        context1 = context[: , :, dates_idx1]
        context2 = context[: , :, dates_idx2]
    else:
        context1, context2 = None, None
    return {"train": (values[:,:,dates_idx1], context1, dates1), "test1":(values[:,:,dates_idx2], context2, dates2)}    

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
    return {"train":(values1, context1, dates1), "test1":(values2, context2, dates2), "test0":(values3, context3, dates1), "test2": (values4, context4, dates2)}

def split_6_way(values, context, datetimes, indiv_split, date_splits, context_by_individuals=True, save_path=None, reshuffle=True):
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


def get_dataset_splits(splits, data_path=None, save_path=None, cluster_path=None, set_cluster_context=None, data=None, cluster_ids=None):
    """splits data from path. If str splits, will load given split, if float will save new split"""
    context_by_indiv, reshuffle = splits["context_by_individuals"], splits["reshuffle"]
    date_splits, indiv_split = splits["date_splits"], splits["indiv_split"]

    #load whole data
    if data is None:
        values, context, datetimes = load_data(data_path) #load dataset
    else:
        values, context, datetimes = data
    
    #filter values at cluster path
    if cluster_path is not None or cluster_ids is not None:
        if cluster_path is not None:
            indices = list(torch.load(cluster_path, weights_only=False))
        else:
            indices = cluster_ids
        values = values[indices]
        if context is not None and context_by_indiv:
            context = context[indices]
        if set_cluster_context is not None:
            context = torch.tensor([set_cluster_context for _ in range(len(indices))]).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, values.shape[1], values.shape[2])

    if type(date_splits) == float:
        date_splits = [date_splits]
    elif type(date_splits) == str:
        date_splits = date_splits.split(";")
        date_splits = [float(txt) for txt in date_splits]
    if type(indiv_split) == str:
        indiv_split = float(indiv_split)
    if date_splits is None or (type(date_splits)==list and date_splits[0]==1) or date_splits==1:
        type_split = 1
    elif len(date_splits) == 1:
        if indiv_split is None or (type(indiv_split)==list and date_splits[0]==1) or indiv_split ==  1 or values.shape[0]==1:
            type_split = 2
        else:
            type_split = 4
    elif len(date_splits) >= 2:
        if indiv_split is None or (type(indiv_split)==list and date_splits[0]==1) or indiv_split ==  1 or values.shape[0]==1:
            type_split = 3
        else:
            type_split = 6
    if type_split == 1:
        data_dict = split_1_way(values, context, datetimes)
    elif type_split == 2:
        data_dict = split_2_way(values, context, datetimes, date_splits)
    elif type_split == 3:
        data_dict = split_3_way(values, context, datetimes, date_splits)
    elif type_split == 4:
        data_dict = split_4_way(values, context, datetimes, indiv_split, date_splits[0], context_by_indiv, save_path, reshuffle=reshuffle)
    elif type_split == 6:
        data_dict = split_6_way(values, context, datetimes, indiv_split, date_splits, context_by_indiv, save_path, reshuffle=reshuffle)
    else:
        raise ValueError(f"Unrecognized type_split: {type_split}")

    return data_dict



def get_train_loaders(data_dict, batch_size, lags, horizon, splits, subsets, save_path=None, standard_stats=None, shuffle_eval=False, random_eval=False):
    """returns dataloaders from data_dict as eventual subsets"""
    subset_mode, subsets  = subsets["mode"], subsets["sizes"]
    idx_mode = splits["idx_mode"]
    reshuffle, context_by_indiv, return_all_indiv = splits["reshuffle"], splits["context_by_individuals"], splits["return_all_individuals"]
    remove_train_cte, remove_eval_cte = splits["remove_train_cte"], splits["remove_eval_cte"]
    
    if subsets is not None:
        subsets = [float(txt) for txt in subsets.split(";")]
    else:
        subsets = [1 for _ in range(len(data_dict))]
    save = (save_path is not None)
    if save:
        subset_dir = save_path + subset_mode + str(subsets) + "/"
    loaders_dict = {}

    for i, (key, (values, context, datetimes)) in enumerate(data_dict.items()):
        
        #subsetting
        subset = subsets[i]
        if subset_mode is None:
            subset_mode = idx_mode
        subset_indices = None
        if subset != 1:
            if save and (not reshuffle):
                subset_indices = list(torch.load(subset_dir + f"{key}_subset.pt", weights_only=False))
            else:
                subset_indices = get_subset_indices(dataset.dates, dataset.individuals, lags, horizon, subset, subset_mode)
            if save:
                if os.path.exists(subset_dir):
                    shutil.rmtree(subset_dir)
                os.makedirs(subset_dir)
                torch.save(subset_indices, subset_dir + f"{key}_subset.pt")
        
        #train loader
        if key == "train":
            if values.shape[0]==1 and batch_size>1:
                weight=batch_size
            else:
                weight=1
            dataset = TimeSeriesDataset(values, datetimes, context, lags, horizon)
            dataset.set_sampler(idx_mode = idx_mode,
                return_all_individuals = return_all_indiv, context_by_individuals = context_by_indiv, remove_cte=remove_train_cte,
                weight=weight, subset_indices=subset_indices, subset_mode=subset_mode)
            if standard_stats is not None:
                dataset.normalize(standard_stats)
            local_collate_fn = lambda x: collate_fn(x)#, remove_cte=remove_train_cte)
            loaders_dict[key] = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=local_collate_fn)
        
        #eval loaders
        else:
            if random_eval:
                idx_mode = "random"
            else:
                idx_mode = "all"
            dataset = TimeSeriesDataset(values, datetimes, context, lags, horizon)
            dataset.set_sampler(idx_mode = idx_mode,
                return_all_individuals = return_all_indiv, context_by_individuals = context_by_indiv, remove_cte=remove_eval_cte,
                weight=weight, subset_indices=subset_indices, subset_mode=subset_mode)       
            if standard_stats is not None:
                dataset.normalize(standard_stats)
            local_collate_fn = lambda x: collate_fn(x)#, remove_cte=remove_eval_cte)
            loaders_dict[key] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_eval, collate_fn=local_collate_fn)
       
    return loaders_dict


def collate_fn(data):#, remove_cte=False):
    """
       data: is a list of tuples with (input, (context), target)
    """
    inputs, contexts, targets, indivs, dates = zip(*data)

    inputs = torch.cat(inputs, dim=0)   # shape: (bs*individuals, dim, lookback)

    if contexts[0] is not None:
        contexts = torch.cat(contexts, dim=0)
    targets = torch.cat(targets, dim=0)   # shape: (bs*individuals, dim, horiz)

    # if remove_cte: #remove constant windows
    #     stds = inputs.std(dim=-1) #(bs * indiv, dim)
    #     non_constant_mask = (stds > 0).any(dim=1)  # (bs * indiv)
    #     inputs, targets = inputs[non_constant_mask], targets[non_constant_mask]
    #     if contexts is not None:
    #         contexts = contexts[non_constant_mask]

    return inputs, contexts, targets, indivs, dates
    
    
def aggregate_loaders_dict(loaders_dicts, lags, horizon, splits, batch_size, shuffle_eval=False, random_eval=False):
    """aggregates loaders of different individuals. Expects same dates."""
    loaders_dict = {}
    keys = list(loaders_dicts[0].keys())
    idx_mode, context_by_indiv, return_all_indiv = splits["idx_mode"], splits["context_by_individuals"], splits["return_all_individuals"]
    remove_train_cte, remove_test_cte = splits["remove_train_cte"], splits["remove_eval_cte"]

    for key in keys:
        if key =="train":
            remove_cte = remove_train_cte
            local_collate_fn = lambda x: collate_fn(x)#, remove_cte=remove_cte)
            shuffle = True
            idx_mode_ = idx_mode
            effective_bs = batch_size

        else:
            remove_cte = remove_test_cte
            local_collate_fn = lambda x: collate_fn(x)#, remove_cte=remove_cte)
            shuffle = shuffle_eval
            if random_eval:
                idx_mode_ = "random"
            else:
                idx_mode_ = "all"
            effective_bs = batch_size

        datetimes = loaders_dicts[0][key].dataset.datetimes
        if context_by_indiv:
            context_list = []
        else:
            context = loaders_dicts[0][key].dataset.context
        values_list = []
        for new_dict in loaders_dicts:
            values = new_dict[key].dataset.values
            values_list.append(values)
            if context_by_indiv:
                context = new_dict[key].dataset.context
                context_list.append(context)
        if context_by_indiv:
            if context_list[0] is None:
                context = None
            else:
                context = torch.cat(context_list, dim=0)
        extended_dataset = TimeSeriesDataset(torch.cat(values_list, dim=0), datetimes, context, lags, horizon)
        extended_dataset.set_sampler(idx_mode = idx_mode_,
            return_all_individuals = return_all_indiv, context_by_individuals = context_by_indiv, remove_cte=remove_cte)
            #weight=weight, subset_indices=subset_indices, subset_mode=subset_mode)        
        extended_loader = DataLoader(extended_dataset, batch_size=effective_bs, shuffle=shuffle, collate_fn=local_collate_fn)
        loaders_dict[key] = extended_loader
    
    return loaders_dict


def get_sizes(loaders_dict, str_info=False):
    """get data size from loaders"""
    loader = next(iter(loaders_dict.values()))
    X, c, y, indiv, date = next(iter(loader)) # (indiv, dim, lags),  #(nc, dim, horizon),  #(indiv, dim, horizon)
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


def format_individual_splits(splits, subsets):
    """formats provided splits and subsets args for 1 indiv's split"""
    splits_ = copy.deepcopy(splits)
    splits_["indiv_split"] = None
    subsets_ = copy.deepcopy(subsets)
    if len(subsets_["sizes"].split(";")) > 3:
        subsets_sizes_ = subsets_["sizes"].split(";")
        subsets_["sizes"] = ";".join([subsets_sizes_[0], subsets_sizes_[1], subsets_sizes_[4]])
    return splits_, subsets_


def fetch_training_data(data_path, splits, subsets, batch_size, lags, horizon, aggregate=True, seed=None, save=False, shuffle_eval=False, cluster_ids=None, random_eval=False, do_nodes=False):
    """returns loaders dict and stats dicts"""
    
    set_seed(seed)

    #save paths
    save_path = None
    if save:
        save_path = data_path

    if splits["clusters"] is not None:
        cluster_path = data_path + splits["clusters"] + "/"
        if save:
            save_path += splits["clusters"] + "/" 

    nodes_stats_dict = {}

    # split by clusters and optionally aggregate 
    if (splits["clusters"] is not None) and (subsets["cluster"] is None):
        cluster_names = [name for name in os.listdir(cluster_path) if name[-3:]==".pt"]
        loaders_dicts = []
        for k, cluster_name in enumerate(cluster_names):
            if save:
                split_path = save_path+cluster_name[:-3]+"splits/"
                subset_path = save_path+cluster_name[:-3]+"subsets/"
            else:
                split_path, subset_path = None, None
            cluster_path_ = cluster_path+cluster_name
            data_dict = get_dataset_splits(splits, data_path, split_path, cluster_path_, set_cluster_context=k)
            loaders_dict = get_train_loaders(data_dict, batch_size, lags, horizon,
                splits, subsets, subset_path,
                standard_stats=None, shuffle_eval=shuffle_eval, random_eval=random_eval)
            loaders_dicts.append(loaders_dict)

            node_dict = {subkey: loader.dataset.get_df() for subkey, loader in loaders_dict.items()}
            if save:
                save_path = save_path+cluster_name[:-3] + "/"
            nodes_stats_dict[f"node{k}"] = get_dataset_stats(node_dict, lags, horizon, splits["remove_train_cte"], splits["remove_eval_cte"], save_path)
        
        if aggregate:
            loaders_dict = aggregate_loaders_dict(loaders_dicts, lags, horizon, splits, batch_size,
                shuffle_eval, random_eval)
            df_dict = {key: loader.dataset.get_df() for key, loader in loaders_dict.items()}
            stats_dict = get_dataset_stats(df_dict, lags, horizon, splits["remove_train_cte"], splits["remove_eval_cte"], save_path)
        else:
            loaders_dict = {f"node{k}": loaders_dicts[k] for k in range(len(loaders_dicts))}
            stats_dict = None
        return loaders_dict, stats_dict, nodes_stats_dict

    else: #1 split

        #fetch 1 cluster path
        if subsets["cluster"] is not None:
            cluster_name = subsets["cluster"]
            cluster_path += cluster_name + ".pt"
            if save:
                split_path = save_path+cluster_name[:-3]+"splits/"
                subset_path = save_path+cluster_name[:-3]+"subsets/"
            else:
                split_path, subset_path = None, None
            data_dict = get_dataset_splits(splits, data_path, split_path, cluster_path)
            loaders_dict = get_train_loaders(data_dict, batch_size, lags, horizon,
                splits, subsets, subset_path,
                standard_stats=None, shuffle_eval=shuffle_eval, random_eval=random_eval)
        
        #fetch all (optionally from cluster_ids)
        else:
            if save:
                split_path = save_path + "splits/"
                subset_path = save_path+ "subsets/"
            else:
                split_path, subset_path = None, None
            data_dict = get_dataset_splits(splits, data_path, split_path, cluster_ids=cluster_ids) #cluster_ids: integer of indivs 
            loaders_dict = get_train_loaders(data_dict, batch_size, lags, horizon,
                splits, subsets, subset_path,
                standard_stats=None, shuffle_eval=shuffle_eval, random_eval=random_eval)
        df_dict = {key: loader.dataset.get_df() for key, loader in loaders_dict.items()}
        stats_dict = get_dataset_stats(df_dict, lags, horizon, splits["remove_train_cte"], splits["remove_eval_cte"], save_path)
        
        #individuals nodes
        if do_nodes:
            n_users = list(df_dict.values())[0].shape[-1]
            
            #format splits for individual users
            splits_, subsets_ = format_individual_splits(splits, subsets)
            # splits_ = copy.deepcopy(splits)
            # splits_["indiv_split"] = None
            # subsets_ = copy.deepcopy(subsets)
            # if len(subsets_["sizes"].split(";")) > 3:
            #     subsets_sizes_ = subsets_["sizes"].split(";")
            #     subsets_["sizes"] = ";".join([subsets_sizes_[0], subsets_sizes_[1], subsets_sizes_[4]])
            
            for node_id in range(n_users):
                data_dict_ = get_dataset_splits(splits_, data_path, split_path, cluster_ids=[node_id])
                loaders_dict_ = get_train_loaders(data_dict_, batch_size, lags, horizon,
                    splits_, subsets_, subset_path,
                    standard_stats=None, shuffle_eval=shuffle_eval)
                node_dict_ = {subkey: loader.dataset.get_df() for subkey, loader in loaders_dict_.items()}
                nodes_stats_dict[f"node{node_id}"] = get_dataset_stats(node_dict_, lags, horizon, splits_["remove_train_cte"], splits_["remove_eval_cte"], save_path)
        else:
            nodes_stats_dict = None
        return loaders_dict, stats_dict, nodes_stats_dict


def apply_standard_norm(loaders_dict, stats_dict):
    """apply standard normalization to loaders using stats_dict"""
    for key, loader in loaders_dict.items():
        loader.dataset.normalize(stats_dict[key])


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

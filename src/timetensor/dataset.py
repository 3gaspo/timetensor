import torch
import numpy as np
import os
import shutil
import copy
import pandas as pd
import warnings

from torch.utils.data import Dataset, DataLoader

from .utils import normalize, set_seed
from .analysis import get_dataset_stats


class IndexSampler:
    def __init__(self, values, context, lags, horizon, idx_mode="random", block_individuals=1, use_context=True, remove_cte=False, weight=1, subset_indices=None, subset_mode="dates", stride=1):
        """sampler to fetch indiv and date indices for dataset
        idx_mode: what idx corresponds to
        subset_mode: what subsets_indices correspond to
        """
        self.values, self.context = values, context
        self.individuals, self.dim_values, self.dates = self.values.shape
        self.contexts, self.dim_context, self.context_dates = self.context.shape
        self.lags, self.horizon = lags, horizon
        
        self.idx_mode, self.block_individuals = idx_mode, block_individuals
        self.use_context = use_context
        self.remove_cte = remove_cte
        self.weight = weight
        self.subset_indices, self.subset_mode = subset_indices, subset_mode
        self.stride = stride

        self.max_dates = self.dates - (self.lags + self.horizon) + 1
        if self.contexts == self.individuals:
            self.context_by_individuals = True
        else:
            self.context_by_individuals = False

    @property
    def max_strided_dates(self):
        """Dynamically calculates max available strided steps based on current config"""
        if self.subset_indices is not None and self.subset_mode == "dates":
            n_dates = len(self.subset_indices)
        else:
            n_dates = self.max_dates
        return (n_dates - 1) // self.stride + 1

    def _true_len(self):
        if self.idx_mode == "dates":
            return self.max_strided_dates
        elif self.idx_mode == "individuals":
            if self.subset_indices is not None and self.subset_mode == "individuals":
                return len(self.subset_indices)
            return self.individuals
        elif self.idx_mode == "all":
            if self.subset_indices is not None:
                if self.subset_mode == "dates":
                    return self.individuals * self.max_strided_dates
                elif self.subset_mode == "individuals":
                    return len(self.subset_indices) * self.max_strided_dates
                elif self.subset_mode == "all":
                    return len(self.subset_indices)
            return self.individuals * self.max_strided_dates
        elif self.idx_mode == "random":
            return 1
        else:
            raise ValueError(f"Unrecognized idx_mode: {self.idx_mode}")

    @property
    def shape(self):
        #values
        if self.subset_indices is not None and self.subset_mode == "individuals":
            n_indivs = len(self.subset_indices)
        else:
            n_indivs = self.individuals
        n_dates = self.max_strided_dates

        #context
        if self.context_dates == 1:
            n_context_dates = 1
        else:
            n_context_dates = n_dates

        return (n_indivs, self.dim_values, n_dates), (self.contexts, self.dim_context, n_context_dates)

    def __len__(self):
        return self.weight * self._true_len()

    def _get_non_cte_mask(self, indivs, date, eps=1e-8):
        """return mask of indiv with non constant lookback"""
        lookbacks = self.values[indivs, :, date: date + self.lags] #(individuals, dim_values, lags)
        mask = (lookbacks.std(dim=-1) > eps).any(dim=1)
        return mask
    
    def _get_strided_date(self, raw_step=None):
        """Returns actual date index from a raw step (0 to max_strided-1) or random if None"""
        if raw_step is None:
            raw_step = np.random.randint(self.max_strided_dates)
            
        date_idx = raw_step * self.stride
        
        if self.subset_indices is not None and self.subset_mode == "dates":
            date = self.subset_indices[date_idx]
            assert date < self.max_dates
        else:
            date = date_idx
        return date

    def _get_indivs(self, idx=None):
        """Returns list of individuals based on current mode/subset"""
        if self.subset_indices is not None and self.subset_mode == "individuals":
            if idx is not None:
                return [self.subset_indices[idx]]
            # Random selection from subset
            if self.block_individuals > 1:
                if self.block_individuals < len(self.subset_indices):
                    return list(np.random.choice(self.subset_indices, self.block_individuals))
                return self.subset_indices
            else:
                return [np.random.choice(self.subset_indices)]
        else:
            if idx is not None:
                return [idx]
            # Random selection from all
            if self.block_individuals > 1:
                if self.block_individuals < self.individuals:
                    return list(np.random.choice(self.individuals, self.block_individuals))
                return list(range(self.individuals))
            else:
                return [np.random.randint(self.individuals)]

    def __call__(self, raw_idx):
        idx = raw_idx % self._true_len()
        
        # 1. Determine initial indivs and date based on idx_mode
        if self.idx_mode == "dates":
            date = self._get_strided_date(idx)
            indivs = self._get_indivs(idx=None) # random indivs

        elif self.idx_mode == "individuals":
            indivs = self._get_indivs(idx=idx)
            date = self._get_strided_date(raw_step=None) # random date

        elif self.idx_mode == "all":
            if self.subset_indices is not None and self.subset_mode == "all":
                # Warning: stride is technically ignored here, idx selects the pair indiv, date
                raw_pair_idx = self.subset_indices[idx]
                indivs, date = [raw_pair_idx % self.individuals], raw_pair_idx // self.individuals
            else:
                if self.subset_indices is not None and self.subset_mode == "individuals":
                    n_indivs = len(self.subset_indices)
                    indiv_idx = idx % n_indivs
                    date_step = idx // n_indivs
                    indivs = self._get_indivs(idx=indiv_idx)
                else:
                    n_indivs = self.individuals
                    indiv_idx = idx % n_indivs
                    date_step = idx // n_indivs
                    indivs = [indiv_idx]
                
                date = self._get_strided_date(raw_step=date_step)

        elif self.idx_mode == "random":
            date = self._get_strided_date(raw_step=None)
            indivs = self._get_indivs(idx=None)
        else:
            raise ValueError(f"Unrecognized idx_mode: {self.idx_mode}")

        # 2. Handle constant windows
        if self.remove_cte:
            mask = self._get_non_cte_mask(indivs, date)
            remove_cte_counter = 0

            # require ALL individuals in the block to be non-cte
            while not mask.all().item():

                if self.idx_mode == "dates":
                    # keep date fixed (idx selects the date), resample indivs only
                    indivs = self._get_indivs(idx=None)

                elif self.idx_mode == "individuals":
                    # keep individual fixed (idx selects the indiv), resample date only
                    date = self._get_strided_date(raw_step=None)

                else:
                    # free resampling
                    indivs = self._get_indivs(idx=None)
                    date = self._get_strided_date(raw_step=None)

                remove_cte_counter += 1
                if remove_cte_counter > 100:
                    raise ValueError("Overflow constant windows")

                mask = self._get_non_cte_mask(indivs, date)

        # 3. Context
        if self.use_context:
            if self.context_by_individuals:
                context_idx = indivs
            else:
                context_idx = list(range(self.contexts))
        else:
            context_idx = None

        return indivs, date, context_idx


class TimeSeriesDataset(Dataset):
    """dataset of multiple individuals"""
    def __init__(self, values, datetimes=None, context=None, lags=168, horizon=24, build_context=False):   
        """
        values (N_individuals, dim_values, dates):  past target values 
        datetimes (dates): list of dates in datetime Y-m-d H:M:S format
        context (N_contexts, dim_context, dates): exogenous variates  e.g N_contexts=1 or N_contexts=N_individuals
        lags (int): size of lookback window
        horizon (int): size of target horizon
        build_context (bool): add users index as context
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

        self.build_context = build_context
        if self.build_context:
            if self.context is None:
                self.context = torch.tensor([[k] for k in range(self.values.shape[0])]).unsqueeze(dim=1)            
            if len(self.context.shape) == 1:
                self.context = self.context.unsqueeze(0)
            if len(self.context.shape) == 2:
                self.context = self.context.unsqueeze(0)
            self.contexts, self.dim_context, self.context_dates = self.context.shape
            assert self.context_dates == self.dates or self.context_dates == 1, f"error context temporal size {self.context.shape}"
        else:
            self.contexts, self.dim_context, self.context_dates = 0, 0, 0

        self.index_sampler = IndexSampler(self.values, self.context, self.lags, self.horizon) #default random sampler

    @property
    def original_shape(self):
        return (self.individuals, self.dim_values, self.dates), (self.contexts, self.dim_context, self.context_dates)

    @property
    def shape(self):
        return self.index_sampler.shape
    
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
    
    def set_sampler(self, **kwargs):
        """updates default sampler for special indexing and subsets"""
        for key, value in kwargs.items():
            if not hasattr(self.index_sampler, key):
                raise AttributeError(f"IndexSampler has no attribute '{key}'")
            if key in ["values", "lags", "horizon"]:
                raise AttributeError(f"{key} should not be changed after dataset init")
            if key == "stride": assert value>0 and value < (self.dates-self.lags-self.horizon + 1)
            setattr(self.index_sampler, key, value)

    def __getitem__(self, idx):                
        indivs, date, context_idx = self.index_sampler(idx)
        values = self.values[indivs, :, date : date + self.lags + self.horizon] # (individuals, dim_values, lags+horizon)
        inputs = values[:, :, :self.lags] # (individuals, dim, lags)
        target = values[:, :, self.lags:] # (individuals, dim, horizon)
        
        if (self.context is not None) and (context_idx is not None):
            if self.context_dates == 1:
                context = self.context[context_idx, :, 0].unsqueeze(-1) # (individuals, dim_context, 1)
            else:
                context = self.context[context_idx, :, date : date + self.lags + self.horizon] #(individuals, dim_context, lags+horizon)
        else:
            context = None
        return inputs, context, target, indivs, date


def fetch_csv(data_path, data_name, context_cols=None, drop_users=None, rename_cols=None, aggr=None, aggr_period="h"):
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
    if rename_cols is not None:
        values_df = values_df.rename(columns=rename_cols)
    else:
        values_df.columns = [f"user_{k}" for k in range(values_df.shape[1])] #range(values_df.shape[1]) 
    if drop_users:
        drop = drop_users.split(";")
        drop = [f"user_{int(idx)}" for idx in drop]
        values_df = values_df.drop(columns=drop)
        values_df.columns = [f"user_{k}" for k in range(values_df.shape[1])]

    if aggr == "sum":
        values_df = values_df.resample(aggr_period).sum()
        if context_df is not None:
            context_df = context_df.resample(aggr_period).sum()
    elif aggr:
        values_df = values_df.asfreq(aggr_period)
        if context_df is not None:
            context_df = context_df.asfreq(aggr_period)

    datetimes = list(values_df.index)

    return values_df, context_df, datetimes


def build_dataset(data_path, data_name, context_cols=None, drop_users=None, do_context=True, raw_format="csv", rename_cols=None, aggr=None, aggr_period="h"):
    """builds pytorch tensors from csv path"""
    #load csv
    if raw_format == "csv":
        values_df, context_df, datetimes = fetch_csv(data_path, data_name, context_cols, drop_users, rename_cols, aggr, aggr_period)
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
        context_pt =  torch.tensor([[k] for k in range(values_pt.shape[0])]).unsqueeze(dim=1)
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

def split_2_way(values, context, datetimes, date_split, timed_context=True):
    """returns dict of train/valid/test of provided values,context,datetimes
    """
    dates = len(datetimes)
    stop_date1 = int(date_split * dates)
    dates_idx1, dates_idx2 = list(range(stop_date1)), list(range(stop_date1, dates))
    dates1, dates2 = list(datetimes[:stop_date1]), list(datetimes[stop_date1::])    
    if context is None or (not timed_context):
        return {"train": (values[:,:,dates_idx1], context, dates1), "test1":(values[:,:,dates_idx2], context, dates2)}    
    else:
        context1 = context[: , :, dates_idx1]
        context2 = context[: , :, dates_idx2]
        return {"train": (values[:,:,dates_idx1], context1, dates1), "test1":(values[:,:,dates_idx2], context2, dates2)}       

def split_3_way(values, context, datetimes, date_splits, timed_context=True):
    """returns dict of train/valid/test of provided values,context,datetimes
    """
    dates = len(datetimes)
    stop_date1, stop_date2 = int(date_splits[0] * dates), int((date_splits[0] + date_splits[1])*dates)
    dates_idx1, dates_idx2, dates_idx3 = list(range(stop_date1)), list(range(stop_date1, stop_date2)), list(range(stop_date2, dates))
    dates1, dates2, dates3 = list(datetimes[:stop_date1]), list(datetimes[stop_date1:stop_date2]), list(datetimes[stop_date2:])

    if context is None or (not timed_context):
        return {"train": (values[:,:,dates_idx1], context, dates1), "valid1":(values[:,:,dates_idx2], context, dates2), "test1":(values[:,:,dates_idx3], context, dates3)}    
    else:
        context1 = context[: , :, dates_idx1]
        context2 = context[: , :, dates_idx2]
        context3 = context[: , :, dates_idx3]
        return {"train": (values[:,:,dates_idx1], context1, dates1), "valid1":(values[:,:,dates_idx2], context2, dates2), "test1":(values[:,:,dates_idx3], context3, dates3)}    
    
def split_4_way(values, context, datetimes, indiv_split, date_split, context_by_individuals=True, save_path=None, reshuffle=True, timed_context=True):
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
            context1 = context[indices1, :, :]
            context2 = context[indices1, :, :]
            context3 = context[indices2, :, :]
            context4 = context[indices2, :, :]
        if timed_context:
            context1 = context[: , :, dates_idx1]
            context2 = context[: , :, dates_idx2]
            context3 = context[: , :, dates_idx1]
            context4 = context[: , :, dates_idx2]
    else:
        context1, context2, context3, context4 = None, None, None, None
    return {"train":(values1, context1, dates1), "test1":(values2, context2, dates2), "test0":(values3, context3, dates1), "test2": (values4, context4, dates2)}

def split_6_way(values, context, datetimes, indiv_split, date_splits, context_by_individuals=True, save_path=None, reshuffle=True, timed_context=True):
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
            context1 = context[indices1, :, :]
            context2 = context[indices1, :, :]
            context3 = context[indices1, :, :]
            context4 = context[indices2, :, :]
            context5 = context[indices2, :, :]
            context6 = context[indices2, :, :]
        if timed_context:
            context1 = context[: , :, dates_idx1]
            context2 = context[: , :, dates_idx2]
            context3 = context[: , :, dates_idx3]
            context4 = context[: , :, dates_idx1]
            context5 = context[: , :, dates_idx2]
            context6 = context[: , :, dates_idx3]
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
    """splits data from path. If str splits, will load given split, if float will save new split
    set_cluster_context: associated provided integer value to specified cluster (via cluster_ids or cluster_path)
    """
    date_splits, indiv_split, reshuffle = splits["date_splits"], splits["indiv_split"], splits["reshuffle"]

    #load whole data
    if data is None:
        values, context, datetimes = load_data(data_path) #load dataset
    else:
        values, context, datetimes = data

    individuals, dim_values, dates = values.shape
    if individuals == 1:
        indiv_split = 1
    
    timed_context = True
    context_by_individuals = False
    if context is not None:
        contexts, dim_context, context_dates = context.shape
        if context_dates == 1:
            timed_context = False
        if contexts == individuals:
            context_by_individuals = True

    #filter values at cluster path
    if cluster_path is not None or cluster_ids is not None:
        if cluster_path is not None:
            indices = list(torch.load(cluster_path, weights_only=False))
        else:
            indices = cluster_ids
        values = values[indices]
        if context is not None and context_by_individuals:
            context = context[indices]
        if set_cluster_context is not None:
            context = torch.full((len(indices), 1, 1), set_cluster_context) # (indices, 1, 1)

    if type(date_splits) == float or type(date_splits) == int:
        date_splits = [date_splits]
    elif type(date_splits) == str:
        date_splits = date_splits.split(";")
        date_splits = [float(txt) for txt in date_splits]
    if type(indiv_split) == str:
        indiv_split = float(indiv_split)
    if date_splits is None or (type(date_splits)==list and date_splits[0]==1): #no splits
        type_split = 1
    elif len(date_splits) == 1 or (type(date_splits)==list and date_splits[0]+date_splits[1]==1): # split dates in two
        if indiv_split is None or (type(indiv_split)==list and date_splits[0]==1) or indiv_split ==  1:
            type_split = 2
        else: # and split indivs
            type_split = 4
    elif len(date_splits) >= 2: # split dates in three
        if indiv_split is None or (type(indiv_split)==list and date_splits[0]==1) or indiv_split ==  1:
            type_split = 3
        else: # and split indivs
            type_split = 6
    if type_split == 1:
        data_dict = split_1_way(values, context, datetimes)
    elif type_split == 2:
        data_dict = split_2_way(values, context, datetimes, date_splits[0], timed_context=timed_context)
    elif type_split == 3:
        data_dict = split_3_way(values, context, datetimes, date_splits, timed_context=timed_context)
    elif type_split == 4:
        data_dict = split_4_way(values, context, datetimes, indiv_split, date_splits[0], context_by_individuals, save_path, reshuffle=reshuffle, timed_context=timed_context)
    elif type_split == 6:
        data_dict = split_6_way(values, context, datetimes, indiv_split, date_splits, context_by_individuals, save_path, reshuffle=reshuffle, timed_context=timed_context)
    else:
        raise ValueError(f"Unrecognized type_split: {type_split}")

    return data_dict



def get_train_loaders(data_dict, batch_size, lags, horizon, splits, sampling, subsets, save_path=None, standard_stats=None):
    """returns dataloaders from data_dict as eventual subsets"""
    subset_mode, subsets  = subsets["mode"], subsets["sizes"]
    if subsets is not None:
        subsets = {key: float(value) for key, value in subsets.items()}
    else:
        subsets = {key: 1 for key in ["train", "valid1", "valid2", "valid3", "test1", "test2"]}
    save = (save_path is not None)
    if save:
        subset_dir = save_path + subset_mode + str(subsets) + "/"
    loaders_dict = {}

    for key, (values, context, datetimes) in data_dict.items():
        if key == "train":
            idx_mode = sampling["train_idx_mode"]
            remove_cte = sampling["remove_train_cte"]
            shuffle = sampling["shuffle_train"]
            stride = sampling["train_stride"]
            weight = sampling["train_len_multiplier"]
            blocks = sampling["train_block_individuals"]
        else:
            idx_mode = sampling["eval_idx_mode"]
            remove_cte = sampling["remove_eval_cte"]
            shuffle = sampling["shuffle_eval"]
            stride = sampling["eval_stride"]
            weight = sampling["eval_len_multiplier"]
            blocks = sampling["eval_block_individuals"]

        #dataset
        dataset = TimeSeriesDataset(values, datetimes, context, lags, horizon)

        #subsets
        subset = subsets[key]
        if subset_mode is None:
            subset_mode = idx_mode
        subset_indices = None
        if subset != 1:
            if save and (not splits["reshuffle"]):
                subset_indices = list(torch.load(subset_dir + f"{key}_subset.pt", weights_only=False))
            else:
                subset_indices = get_subset_indices(dataset.dates, dataset.individuals, lags, horizon, subset, subset_mode)
            if save:
                if os.path.exists(subset_dir):
                    shutil.rmtree(subset_dir)
                os.makedirs(subset_dir)
                torch.save(subset_indices, subset_dir + f"{key}_subset.pt")
        
        #loader
        dataset.set_sampler(idx_mode = idx_mode,
            block_individuals = blocks, use_context = sampling["use_context"], remove_cte=remove_cte,
            weight=weight, subset_indices=subset_indices, subset_mode=subset_mode, stride=stride)       
        if standard_stats is not None:
            dataset.normalize(standard_stats)
        loaders_dict[key] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=0)
       
    return loaders_dict


def collate_fn(data):
    """data: list of tuples with (input, context, target, indiv, date)"""
    inputs, contexts, targets, indivs, dates = zip(*data)
    inputs = torch.cat(inputs, dim=0)   # (bs*(individuals), dim, lookback)
    targets = torch.cat(targets, dim=0)   # (bs*(individuals), dim, horizon)
    if contexts[0] is not None:
        contexts = torch.cat(contexts, dim=0)  # (bs*(individuals), dim_context, 1*(lookback+horizon))  
    else:
        contexts = None
    flat_indivs = [indiv for item in indivs for indiv in item] # (bs*(individuals))
    flat_dates = [dates[i] for i,item in enumerate(indivs) for _ in range(len(item))] # (bs*(individuals))
    return inputs, contexts, targets, flat_indivs, flat_dates
    
    
def aggregate_loaders_dict(loaders_dicts, lags, horizon, sampling, batch_size):
    """aggregates loaders of different individuals. Expects same dates.
    loaders_dicts: list of loaders_dict
    no weighting nor subsetting"""

    keys = list(loaders_dicts[0].keys())
    loaders_dict = {}
    for key in keys:
        if key == "train":
            idx_mode = sampling["train_idx_mode"]
            remove_cte = sampling["remove_train_cte"]
            shuffle = sampling["shuffle_train"]
        else:
            idx_mode = sampling["eval_idx_mode"]
            remove_cte = sampling["remove_eval_cte"]
            shuffle = sampling["shuffle_eval"]

        datetimes = loaders_dicts[0][key].dataset.datetimes
        block_individuals = loaders_dicts[0][key].dataset.index_sampler.block_individuals
        context_by_individuals = loaders_dicts[0][key].dataset.index_sampler.context_by_individuals
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
        values = torch.cat(values_list, dim=0)
        extended_dataset = TimeSeriesDataset(values, datetimes, context, lags, horizon)
        extended_dataset.set_sampler(idx_mode = idx_mode,
            block_individuals = block_individuals, remove_cte=remove_cte)
        extended_loader = DataLoader(extended_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=0)
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

def fetch_training_data(data_path, splits, sampling, subsets, batch_size, lags, horizon, aggregate=True, seed=None, save=False, cluster_ids=None):#, do_indiv_stats=False):
    """returns loaders dict and stats dicts
    aggregate: if specified cluster, to aggregate as one loader or not
    cluster_ids: directly specify list of user ids to fetch
    """
    set_seed(seed)

    #save paths
    save_path = None
    if save:
        save_path = data_path

    if splits["clusters"] is not None:
        cluster_path = data_path + splits["clusters"] + "/"
        if save:
            save_path += splits["clusters"] + "/" 

    # split by clusters and optionally aggregate 
    if (splits["clusters"] is not None) and (subsets["cluster"] is None):
        cluster_names = [name for name in os.listdir(cluster_path) if name[-3:]==".pt"]
        loaders_dicts = []
        stats_dict = {}
        for k, cluster_name in enumerate(cluster_names):
            if save:
                split_path = save_path+cluster_name[:-3]+"splits/"
                subset_path = save_path+cluster_name[:-3]+"subsets/"
            else:
                split_path, subset_path = None, None
            cluster_path_ = cluster_path + cluster_name
            data_dict = get_dataset_splits(splits, data_path, split_path, cluster_path_, set_cluster_context=k)
            loaders_dict = get_train_loaders(data_dict, batch_size, lags, horizon,
                splits, sampling, subsets, subset_path,
                standard_stats=None)
            loaders_dicts.append(loaders_dict)

            node_dict = {subkey: loader.dataset.get_df() for subkey, loader in loaders_dict.items()}
            if save:
                save_path = save_path+cluster_name[:-3] + "/"
            if not aggregate:
                stats_dict[f"node{k}"] = get_dataset_stats(node_dict, lags, horizon, sampling, save_path)
        
        if aggregate:
            loaders_dict = aggregate_loaders_dict(loaders_dicts, lags, horizon, sampling, batch_size)
            df_dict = {key: loader.dataset.get_df() for key, loader in loaders_dict.items()}
            stats_dict = get_dataset_stats(df_dict, lags, horizon, sampling, save_path)
        else:
            loaders_dict = {f"node{k}": loaders_dicts[k] for k in range(len(loaders_dicts))}
        return loaders_dict, stats_dict

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
                splits, sampling, subsets, subset_path,
                standard_stats=None)
        
        #fetch all (optionally from cluster_ids)
        else:
            if save:
                split_path = save_path + "splits/"
                subset_path = save_path+ "subsets/"
            else:
                split_path, subset_path = None, None
            data_dict = get_dataset_splits(splits, data_path, split_path, cluster_ids=cluster_ids) #cluster_ids: integer of indivs 
            loaders_dict = get_train_loaders(data_dict, batch_size, lags, horizon,
                splits, sampling, subsets, subset_path,
                standard_stats=None)
        
        df_dict = {key: loader.dataset.get_df() for key, loader in loaders_dict.items()}
        stats_dict = get_dataset_stats(df_dict, lags, horizon, sampling, save_path)
        
        # #individuals' stats
        # if do_indiv_stats:
        #     indiv_stats_dict = {}
        #     n_users = list(df_dict.values())[0].shape[-1]
        #     for indiv in range(n_users):
        #         data_dict_ = get_dataset_splits(splits, data_path, split_path, cluster_ids=[indiv])
        #         loaders_dict_ = get_train_loaders(data_dict_, batch_size, lags, horizon,
        #             splits, sampling, subsets, subset_path,
        #             standard_stats=None)
        #         node_dict_ = {subkey: loader.dataset.get_df() for subkey, loader in loaders_dict_.items()}
        #         indiv_stats_dict[f"node{indiv}"] = get_dataset_stats(node_dict_, lags, horizon, splits["remove_train_cte"], splits["remove_eval_cte"], save_path)
        #     return loaders_dict, stats_dict, indiv_stats_dict

        return loaders_dict, stats_dict


def apply_standard_norm(loaders_dict, stats_dict):
    """apply standard normalization to loaders using stats_dict"""
    for key, loader in loaders_dict.items():
        loader.dataset.normalize(stats_dict[key])


def set_random_data(path="datasets/", lag=168, horizon=24, name="rand", prefix=""):
    """gets a random individual and random window from dataset"""
    values, context, datetimes = load_data(path, prefix)

    individuals, dim, dates = values.shape
    rand_indiv = np.random.randint(individuals)
    rand_date = np.random.randint(dates - (lag + horizon))

    inputs = values[rand_indiv, :, rand_date : rand_date+lag]
    target = values[rand_indiv, :, rand_date+lag : rand_date+lag+horizon]
    if context is not None:
        if context.shape[-1] > 1:
            if context.shape[0] == individuals:
                context = context[rand_indiv, :, rand_date : rand_date+lag+horizon]
            else:
                context = context[:, :, rand_date : rand_date+lag+horizon]
        else:
            context = context[:, :, 0].unsqueeze(0)
    
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

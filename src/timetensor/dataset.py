import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class TimeSeriesDataset(Dataset):
    """dataset of multiple individuals"""
    def __init__(self, values, datetimes, context=None, lags=48, horizon=24,
               by_date=True, return_all_individuals=True, context_by_individuals=False):#, steps=None, seed=None, shuffle=False):    
        """
        values (N_individuals, dim_values, dates):  past target values 
        datetimes (dates): list of dates in datetime Y-m-d H:M:S format
        context (N_contexts, dim_context, dates): exogenous variates  e.g N_contexts=1 or N_contexts=N_individuals
        lags (int): size of lookback window
        horizon (int): size of target horizon
        by_date (bool): access items by date and random or all individuals
        return_all_individuals (bool): return all individuals or a random
        context_by_individuals(bool):  return one context per individual or all
        """
        super(TimeSeriesDataset, self).__init__()

        self.values, self.context = values, context
        self.lags, self.horizon = lags, horizon 
            
        self.individuals, self.dim_values, self.dates = self.values.shape
        if self.context is not None:
            self.contexts, self.dim_context, _dates = self.context.shape
            assert _dates == self.dates, "not the same dates in values and context"        
        assert self.dates > self.lags + self.horizon, "not enough dates for this lag and horizon"
        
        self.datetimes = np.array(datetimes)
        self.by_date = by_date
        self.return_all_individuals, self.context_by_individuals = return_all_individuals, context_by_individuals

    @property
    def shape(self):
        if self.context is not None:
            return (self.individuals, self.dim_values, self.dates), (self.contexts, self.dim_context, self.dates)
        else:
            return (self.individuals, self.dim_values, self.dates)

    def __len__(self):
        if self.by_date:
            return self.dates - (self.lags + self.horizon)
        else:
            return self.N_individuals

    def __getitem__(self, idx):
        if self.by_date:
            if self.return_all_individuals: #1 batch = all individuals, batch of dates
                values = self.values[:, :, idx : idx + self.lags + self.horizon] # (individuals, dim_values, lags+horizon)
                if self.context is not None:
                    context = self.context[:, :, idx : idx + self.lags + self.horizon] # (contexts, dim_context, lags+horizon)
                inputs = values[:, :, :self.lags] # (individuals, dim, lags)
                target = values[:, :, self.lags:] # (individuals, dim, horizon)
            else: #1 batch = 1 individual, batch of dates
                if self.seed is not None:
                    np.random.seed(self.seed)
                indiv = np.random.randint(self.individuals)
                values = self.values[indiv, :, idx : idx + self.lags + self.horizon] # (dim_values, lags+horizon)
                if self.context is not None:
                    context = self.context[indiv, :, idx : idx + self.lags + self.horizon] # (dim_context, lags+horizon)
                inputs = values[:, :, :self.lags] # (dim, lag)
                target = values[:, :, self.lags:] # (dim, horizon)

        else: #1 batch = batch of individuals, random date
            if self.seed is not None:
                np.random.seed(self.seed)
            t = np.random.randint(self.dates - self.lags - self.horizon)
            values = self.values[idx, :, t: t + self.lags + self.horizon] # (dim_values, lags+horizon)
            if self.context is not None:
                if self.context_by_individuals:
                    context = self.context[idx, :, t: t + self.lags + self.horizon] # (dim_context, lags+horizon)
                else:
                    context = self.context[:, :, t: t + self.lags + self.horizon] # (contexts, dim_context, lags+horizon)
            inputs = values[:, :, :self.lags] # (dim, lags)
            target = values[:, :, self.lags:] # (dim, horizon)

        if self.context is not None:
            return inputs, context, target
        else:
            return inputs, target


class Subset(Dataset):
    def __init__(self, dataset, indices, mode="individuals"):
        self.dataset = dataset
        self.indices = indices
        self.mode = mode

        if self.mode == "individuals":
            self.individuals = len(indices)
            self.dates = self.dataset.dates
            if self.dataset.context is not None:
                if self.dataset.context_by_individuals:
                    self.contexts = self.individuals
                else:
                    self.contexts = self.dataset.contexts
        elif self.mode == "dates":
            self.individuals = self.dataset.individuals
            self.dates = len(indices)
            self.context = self.dataset.context
        else: #TO DO : mode="dim"
            raise ValueError(f"Unrecognized mode: {self.mode}")

    def __getitem__(self, idx):
        if (self.mode == "individuals" and self.dataset.by_date) or (self.mode == "dates" and not self.dataset.by_date):
            return self.dataset[idx]
        else:
            items =  self.dataset[self.indices[idx]]
            try:
                assert items[0].shape[2]==self.dataset.lags
            except:
                raise ValueError(f"error for subset={self.dataset.shape} at idx: {idx}={self.indices[idx]}")
        return items
    
    def __len__(self):
        if self.dataset.by_date:
            if self.mode == "individuals":
                return len(self.dataset)
            else:
                return len(self.indices) - (self.dataset.lags + self.dataset.horizon)
        else:
            if self.mode == "individuals":
                return len(self.indices)
            else:
                return len(self.dataset)

    @property
    def shape(self):
        if self.dataset.context is not None:
            return (self.individuals, self.dataset.dim_values, self.dates), (self.contexts, self.dataset.dim_context, self.dates)
        else:
            return (self.individuals, self.dataset.dim_values, self.dates)

    @property
    def values(self):
        if self.mode == "dates":
            return self.dataset.values[:, :, self.indices]
        else:
            return self.dataset.values[self.indices, :, :]


def build_dataset(fetcher, path="datasets/"):
    """uses fetcher to extract raw data and saves as values, context, datetimes"""
    values, context, datetimes = fetcher(path)
    torch.save(values, path + "values.pt")
    if context is not None:
        torch.save(context, path + "context.pt")
    torch.save(datetimes, path+ "datetimes.pt")

def load_data(path="datasets/", prefix=""):
    """loads values, context, datetimes from path (with eventual prefix)"""
    if prefix is None:
        prefix = ""
    if prefix != "":
        prefix = prefix + "_"
    values = torch.load(path + prefix + "values.pt")
    if os.path.exists(path + prefix + "context.pt"):
        context = torch.load(path + prefix + "context.pt")
    else:
        context = None
    datetimes = np.array(torch.load(path + prefix + "datetimes.pt", weights_only=False))
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



def get_subset_indices(dataset, ratio, mode=None):
    """returns subset of random indices for dataset"""
    if (mode is None and dataset.by_date) or mode=="date": #sample dates
        old_len = dataset.dates - dataset.lags - dataset.horizon
        new_len = int(old_len * ratio)
        if new_len <= dataset.lags + dataset.horizon:
            raise ValueError("Subset not big enough") 
        indices = np.random.choice(old_len, size=new_len, replace=False).tolist()

    elif mode=="individuals": #sample individuals
        new_len = int(dataset.N_individuals * ratio)
        indices = np.random.choice(dataset.N_individuals, size=new_len, replace=False).tolist()
        dataset.values = dataset.values[indices, :, :]
        if dataset.context is not None:
            dataset.context = dataset.context[indices, :, :]
        dataset.datetimes = dataset.datetimes[indices]
        dataset.N_individuals = new_len
    else:
        raise ValueError("Unrecognized mode: ", mode)
    return indices


def train_test_split(values, context, datetimes, indiv_split=None, date_split=None, seed=None, context_by_individuals=False, path=""):
    """returns dict of train/valid/test of provided values,context,datetimes
    split parameters can be in [0,1] or str path to indices
    """

    if seed is not None:
        np.random.seed(seed)

    if date_split is not None:
        if type(date_split)==str:
            dates_idx1, dates_idx2 = list(torch.load(date_split + "_split1.pt", weights_only=False)), list(torch.load(date_split + "_split2.pt", weights_only=False))
            dates1, dates2 = datetimes[dates_idx1], datetimes[dates_idx2]
        elif type(date_split)==float and date_split<1: #split dates
            dates = len(datetimes)
            stop_date = int(date_split * dates)
            dates_idx1, dates_idx2 = list(range(stop_date)), list(range(stop_date, dates))
            dates1, dates2 = list(datetimes[:stop_date]), list(datetimes[stop_date:])
            torch.save(dates_idx1, path + "date_split1.pt")
            torch.save(dates_idx2, path + "date_split2.pt")

        if indiv_split is not None: #split individuals
            if type(indiv_split)==str:
                indices1, indices2 = list(torch.load(indiv_split + "_split1.pt", weights_only=False)), list(torch.load(indiv_split + "_split2.pt", weights_only=False))
            elif type(indiv_split)==float and indiv_split<1: 
                individuals = values.shape[0]
                stop_indiv = int(indiv_split * individuals)
                indices = np.random.permutation(individuals)
                indices1, indices2 = list(indices[:stop_indiv]), list(indices[stop_indiv:])
                torch.save(indices1, path + "indiv_split1.pt")
                torch.save(indices2, path + "indiv_split2.pt")

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
                    context1 = context[:, :, :][: , :, dates_idx1]
                    context2 = context[:, :, :][: , :, dates_idx2]
                    context3 = context[:, :, :][: , :, dates_idx1]
                    context4 = context[:, :, :][: , :, dates_idx2]
            else:
                context1, context2, context3, context4 = None, None, None, None
            return {"train":(values1, context1, dates1), "valid":(values2, context2, dates2), "valid2":(values3, context3, dates1), "test": (values4, context4, dates2)}

        else:
            if context is not None:
                context1 = context[:, :, :][: , :, dates_idx1]
                context2 = context[:, :, :][: , :, dates_idx2]
            else:
                context1, context2 = None, None
            return {"train": (values[:, :, dates1], context1, dates1), "test":(values[:,:,dates2], context2, dates2)}

    elif indiv_split is not None:
        if type(indiv_split)==str:
            indices1, indices2 = list(torch.load(indiv_split + "_split1.pt", weights_only=False)), list(torch.load(indiv_split + "_split2.pt", weights_only=False))
        elif type(indiv_split)==float and indiv_split<1: 
            individuals = values.shape[0]
            stop_indiv = int(indiv_split * individuals)
            indices = np.random.permutation(individuals)
            indices1, indices2 = list(indices[:stop_indiv]), list(indices[stop_indiv:])
            torch.save(indices1, path + "indiv_split1.pt")
            torch.save(indices2, path + "indiv_split2.pt")

        values1 = values[indices1, :, :]
        values2 = values[indices2, :, :]
        if context is not None:
            if context_by_individuals:
                context1 = context[indices1, :, :]
                context2 = context[indices2, :, :]
            else:
                context1 = context[:, :, :]
                context2 = context[:, :, :]
        else:
            context1, context2 = None, None
        return {"train":(values1, context1, dates1), "test" :(values2, context2, dates2)}
    
    else:
        return {"train":(values, context, datetimes)}



def get_dataset_splits(path="datasets/", indiv_split=None, date_split=None, seed=None, save=False, context_by_individuals=False):
    values, context, datetimes = load_data(path) #load dataset
    data_dict = train_test_split(values, context, datetimes, indiv_split, date_split, seed, context_by_individuals, path) #split randomly of according to paths

    if save:
        for key, (values, context, datetimes) in data_dict.items():
            torch.save(values, path + key + "_values.pt")
            if context is not None:
                torch.save(context, path + key + "_context.pt")
            torch.save(datetimes, path + key + "_datetimes.pt")
    return data_dict



def get_train_loaders(data_dict, batch_size, lags, horizon, by_date=True, subsets=None, path=""):
    """returns dataloaders from data_dict as eventual subsets"""
    loaders_dict = {}
    for key, (values, context, datetimes) in data_dict.items():
        dataset = TimeSeriesDataset(values, datetimes, context, lags, horizon, by_date=by_date)
        subset = subsets.get(key)
        if subset is not None and (type(subset)==str or (type(subset)==float and subset<1 and subset>0)):
            if by_date:
                mode = "dates"
            else:
                mode = "individuals"
            if type(subset)==str:
                subset_indices = list(torch.load(subset, weights_only=False))
            elif type(subset)==float:
                subset_indices = get_subset_indices(dataset, subset)
                if not os.path.exists(path):
                    os.makedirs(path)
                torch.save(subset_indices, path + f"{key}_subset_indices_{subset}.pt")
            dataset = Subset(dataset, subset_indices, mode)

        if key=="train":
            loaders_dict[key] = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        else:
            loaders_dict[key] = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
       
    return loaders_dict


def collate_fn(data):
    """
       data: is a list of tuples with (input, (context), target)
    """
    if len(data[0]) == 3:
        inputs, contexts, targets = zip(*data)
    else:
        inputs, targets = zip(*data)
        contexts = None

    inputs = torch.stack(inputs) #(bs, (individuals), dim, lookback)
    inputs = inputs.view(-1, inputs.shape[-2], inputs.shape[-1]) #  (bs * (individuals), dim, lookback)

    if contexts is not None:
        contexts = torch.stack(contexts)
        contexts = contexts.view(-1, contexts.shape[-2], contexts.shape[-1]) 
    targets = torch.stack(targets)
    targets = targets.view(-1, targets.shape[-2], targets.shape[-1])

    return inputs, contexts, targets


def aggregate_loaders(loaders, context_by_individuals=False, by_date=True):
    if not context_by_individuals and by_date: #other cases tODO
        values_list = []
        for loader in loaders:
            values_list.append(loaders.dataset.values)
        extended_values = torch.stack(values_list, dim=0)

        extended_dataset = TimeSeriesDataset(extended_values, loader.dataset.datetimes, loader.dataset.context, loader.dataset.lags, loader.dataset.horizon, by_date=True)
        extended_loader = DataLoader(extended_dataset, batch_size=loader.batch_size, shuffle=loader.shuffle, collate_fn=loader.collate_fn)
        return extended_loader
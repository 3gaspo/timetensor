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

    # def set_subset(self, ratio, mode=None):
    #     if (mode is None and self.by_date) or mode=="date":
    #         if type(ratio)==float:
    #             new_len = int(self.dates * ratio)
    #             if new_len <= self.lags + self.horizon:
    #                 raise ValueError("Subset not big enough") 
    #             indices = np.random.choice(self.dates, size=new_len, replace=False).tolist()
    #         else:
    #             indices = ratio
    #             new_len = len(indices)
    #         self.values = self.values[:, :, indices]
    #         if self.context is not None:
    #             self.context = self.context[:, :, indices]
    #         self.datetimes = self.datetimes[indices]
    #         self.dates = new_len

    #     elif mode=="individuals":
    #         if type(ratio)==float:
    #             new_len = int(self.N_individuals * ratio)
    #             indices = np.random.choice(self.N_individuals, size=new_len, replace=False).tolist()
    #         else:
    #             indices = ratio
    #         self.values = self.values[indices, :, :]
    #         if self.context is not None:
    #             self.context = self.context[indices, :, :]
    #         self.datetimes = self.datetimes[indices]
    #         self.N_individuals = new_len
    #     else:
    #         raise ValueError("Unrecognized mode: ", mode)
    #     return indices

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
        return self.dataset[self.indices[idx]] #should call __get__item

    def __len__(self):
        if self.dataset.by_date:
            if self.mode == "individuals":
                return len(self.dataset)
            else:
                return len(self.indices)
        else:
            if self.mode == "individuals":
                return len(self.indices)
            else:
                return len(self.dataset)

    def shape(self):
        if self.dataset.context is not None:
            return (self.individuals, self.dataset.dim_values, self.dates), (self.contexts, self.dataset.dim_context, self.dates)
        else:
            return (self.individuals, self.dataset.dim_values, self.dates)


def get_subset_indices(dataset, ratio, mode=None):
    if (mode is None and dataset.by_date) or mode=="date":
        new_len = int(dataset.dates * ratio)
        if new_len <= dataset.lags + dataset.horizon:
            raise ValueError("Subset not big enough") 
        indices = np.random.choice(dataset.dates, size=new_len, replace=False).tolist()

    elif mode=="individuals":
        new_len = int(dataset.N_individuals * ratio)
        indices = np.random.choice(dataset.N_individuals, size=new_len, replace=False).tolist()
        dataset.values = dataset.values[indices, :, :]
        if self.context is not None:
            dataset.context = dataset.context[indices, :, :]
        dataset.datetimes = dataset.datetimes[indices]
        dataset.N_individuals = new_len
    else:
        raise ValueError("Unrecognized mode: ", mode)
    return indices


def train_test_split(values, context, datetimes, indiv_split=0.8, date_split=0.8, seed=None, context_by_individuals=False):
    """splits values and datetimes with a split among individuals and dates"""

    if seed is not None:
        np.random.seed(seed)

    if date_split is not None and date_split<1: #split dates
        dates = len(datetimes)
        stop_date = int(date_split * dates)
        dates1, dates2 = datetimes[:stop_date], datetimes[stop_date:] 

        if indiv_split is not None and indiv_split<1: #split individuals
            individuals = values.shape[0]
            stop_indiv = int(indiv_split * individuals)
            indices = np.random.permutation(individuals)
            indices1, indices2 = indices[:stop_indiv], indices[stop_indiv:]

            values1 = values[indices1, :, :stop_date]
            values2 = values[indices1, :, stop_date:]
            values3 = values[indices2, :, :stop_date]
            values4 = values[indices2, :, stop_date:]
            if context is not None:
                if context_by_individuals:
                    context1 = context[indices1, :, :stop_date]
                    context2 = context[indices1, :, stop_date:]
                    context3 = context[indices2, :, :stop_date]
                    context4 = context[indices2, :, stop_date:]
                else:
                    context1 = context[:, :, :stop_date]
                    context2 = context[:, :, stop_date:]
                    context3 = context[:, :, :stop_date]
                    context4 = context[:, :, stop_date:]
            else:
                context1, context2, context3, context4 = None, None, None, None
            return {"train":(values1, context1, dates1), "valid":(values2, context2, dates2), "valid2":(values3, context3, dates1), "test": (values4, context4, dates2)}

        else:
            if context is not None:
                context1 = context[:,:,dates1]
                context2 = context[:,:,dates2]
            else:
                context1, context2 = None, None
            return {"train": (values[:,:,dates1], context1, dates1), "test":(values[:,:,dates2], context2, dates2)}

    elif indiv_split is not None and indiv_split<1: #split individuals
        individuals = values.shape[0]
        stop_indiv = int(indiv_split * individuals)
        indices = np.random.permutation(individuals)
        indices1, indices2 = indices[:stop_indiv], indices[stop_indiv:]

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
        return {"":(values, context, datetimes)}


def build_datasets(fetcher, path="datasets/", indiv_split=0.8, date_split=0.8, seed=None):
    values, context, datetimes = fetcher(path)
    data_dict = train_test_split(values, context, datetimes, indiv_split, date_split, seed)
    for key, (values, context, datetimes) in data_dict.items():
        torch.save(values, path + key + "_values.pt")
        if context is not None:
            torch.save(context, path + key + "_context.pt")
        torch.save(datetimes, path + key + "_datetimes.pt")


def load_data(path="datasets/", prefix=""):
    if prefix is None:
        prefix = ""
    if prefix != "":
        prefix = prefix + "_"
    values = torch.load(path + prefix + "values.pt")
    if os.path.exists(path + prefix + "context.pt"):
        context = torch.load(path + prefix + "context.pt")
    else:
        context = None
    datetimes = torch.load(path + prefix + "datetimes.pt", weights_only=False)
    return values, context, datetimes

def load_example(path="datasets/", prefix=""):
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


def load_datasets(path="datasets/"):
    files = [f for f in os.listdir(path) if ".pt" in f and "subset" not in f]
    data_dict = {}
    for file in files:
        name, key = file.split(".")[0].split("_")
        if data_dict.get(name) is None:
            data_dict[name] = {}
        data_dict[name][key] = torch.load(path + file, weights_only=False)
    return data_dict


def get_train_loaders(path, batch_size, lags, horizon, valid_mode=1, by_date=True, subset=1):

    data_dict = load_datasets(path)
    
    #train loader
    values, context, datetimes = data_dict["train"]["values"], data_dict["train"].get("context"), data_dict["train"]["datetimes"]
    dataset = TimeSeriesDataset(values, datetimes, context, lags, horizon, by_date=by_date)
    if type(subset)==str or (type(subset)==float and subset < 1):
        if type(subset)==str:
            subset_indices = list(torch.load(subset, weights_only=False))
        else:
            assert subset > 0
            if os.path.exists(path + f"subset_indices_{subset}.pt"):
                subset_indices = list(torch.load(path + f"subset_indices_{subset}.pt", weights_only=False))
                #_ = dataset.set_subset(subset_indices)
            else:
                subset_indices = get_subset_indices(dataset, subset)
                #subset_indices = dataset.set_subset(subset)
                torch.save(subset_indices, path + f"subset_indices_{subset}.pt")
        if by_date:
            mode = "dates"
        else:
            mode = "individuals"
        dataset = Subset(dataset, subset_indices, mode)
    loaders_dict = {"train":  DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)}
    
    #valid loader
    if valid_mode == 2:
        validkey = "valid2"
    else:
        validkey = "valid"
    values, context, datetimes = data_dict[validkey]["values"], data_dict[validkey].get("context"), data_dict[validkey]["datetimes"]
    dataset = TimeSeriesDataset(values, datetimes, context, lags, horizon, by_date=True)
    loaders_dict["valid"] = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    #test loader
    values, context, datetimes = data_dict["test"]["values"], data_dict["test"].get("context"), data_dict["test"]["datetimes"]
    dataset = TimeSeriesDataset(values, datetimes, context, lags, horizon, by_date=True)
    loaders_dict["test"] = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

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
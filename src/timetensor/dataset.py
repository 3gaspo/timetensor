import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import copy

class TimeSeriesDataset(Dataset):
    """dataset of multiple individuals"""
    def __init__(self, values, datetimes=None, context=None, lags=336, horizon=24, by_date=True, return_all_individuals=True, context_by_individuals=False, seed=None):    
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
            self.datetimes = np.array(range(0, datetimes))
        self.datetimes = np.array(datetimes)
        self.by_date = by_date
        self.return_all_individuals, self.context_by_individuals = return_all_individuals, context_by_individuals
        self.seed = seed

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
            return self.individuals

    def __getitem__(self, idx):
        if self.by_date:
            if self.return_all_individuals: #1 batch = all individuals, batch of dates
                values = self.values[:, :, idx : idx + self.lags + self.horizon] # (individuals, dim_values, lags+horizon)
                if self.context is not None:
                    context = self.context[:, :, idx : idx + self.lags + self.horizon] # (contexts, dim_context, lags+horizon)

            else: #1 batch = 1 individual, batch of dates
                if self.seed is not None:
                    np.random.seed(self.seed)
                indiv = np.random.randint(self.individuals)
                values = self.values[indiv, :, idx : idx + self.lags + self.horizon].unsqueeze(0) # (1, dim_values, lags+horizon)
                if self.context is not None:
                    if self.context_by_individuals:
                        context = self.context[indiv, :, idx : idx + self.lags + self.horizon].unsqueeze(0) # (1, dim_context, lags+horizon)
                    else:
                        context = self.context[:, :, idx : idx + self.lags + self.horizon] # (contexts, dim_context, lags+horizon)



        else: #1 batch = batch of individuals, random date
            if self.seed is not None:
                np.random.seed(self.seed)
            t = np.random.randint(self.dates - self.lags - self.horizon)
            values = self.values[idx, :, t: t + self.lags + self.horizon].unsqueeze(0) # (1, dim_values, lags+horizon)
            if self.context is not None:
                if self.context_by_individuals:
                    context = self.context[idx, :, t: t + self.lags + self.horizon].unsqueeze(0) # (1, dim_context, lags+horizon)
                else:
                    context = self.context[:, :, t: t + self.lags + self.horizon] # (contexts, dim_context, lags+horizon)

        inputs = values[:, :, :self.lags] # (individuals, dim, lags)
        target = values[:, :, self.lags:] # (individuals, dim, horizon)

        if self.context is not None:
            return inputs, context, target
        else:
            return inputs, target


class Subset(Dataset):
    def __init__(self, dataset, indices, subset_mode="dates"):
        self.indices = indices
        self.mode = subset_mode
        self.lags, self.horizon = dataset.lags, dataset.horizon 

        if self.mode == "individuals":
            if dataset.by_date:
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
        if self.dataset.context is not None:
            return (self.individuals, self.dim_values, self.dates), (self.contexts, self.dataset.dim_context, self.dates)
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



def get_subset_indices(dataset, ratio, subset_mode=None):
    """returns subset of random indices for dataset"""
    if (subset_mode is None and dataset.by_date) or subset_mode=="dates": #sample dates
        old_len = dataset.dates - dataset.lags - dataset.horizon
        new_len = int(old_len * ratio)
        assert new_len > dataset.lags + dataset.horizon, "Not enough dates"
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


def train_test_split(values, context, datetimes, indiv_split=None, date_split=None, seed=None, context_by_individuals=False, path="", save=True):
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
            if save:
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
                if save:
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
            return {"train": (values[:, :, dates_idx1], context1, dates1), "test":(values[:,:,dates_idx2], context2, dates2)}

    elif indiv_split is not None:
        if type(indiv_split)==str:
            indices1, indices2 = list(torch.load(indiv_split + "_split1.pt", weights_only=False)), list(torch.load(indiv_split + "_split2.pt", weights_only=False))
        elif type(indiv_split)==float and indiv_split<1: 
            individuals = values.shape[0]
            stop_indiv = int(indiv_split * individuals)
            indices = np.random.permutation(individuals)
            indices1, indices2 = list(indices[:stop_indiv]), list(indices[stop_indiv:])
            if save:
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


def temporal_split(values, context, datetimes, date_split=None, seed=None, path="", save=True):
    """returns dict of train/valid/test of provided values,context,datetimes
    """
    if seed is not None:
        np.random.seed(seed)
    if date_split is not None:
        if type(date_split)==str:
            dates_idx1, dates_idx2, dates_idx3 = list(torch.load(date_split + "_split1.pt", weights_only=False)), list(torch.load(date_split + "_split2.pt", weights_only=False)), list(torch.load(date_split + "_split3.pt", weights_only=False)),
            dates1, dates2, dates3 = datetimes[dates_idx1], datetimes[dates_idx2], datetimes[dates_idx3]
        elif type(date_split[0])==float: #split dates
            dates = len(datetimes)
            stop_date1, stop_date2 = int(date_split[0] * dates), int((date_split[0] + date_split[1])*dates)
            dates_idx1, dates_idx2, dates_idx3 = list(range(stop_date1)), list(range(stop_date1, stop_date2)), list(range(stop_date2, dates))
            dates1, dates2, dates3 = list(datetimes[:stop_date1]), list(datetimes[stop_date1:stop_date2]), list(datetimes[stop_date2:])
            if save:
                torch.save(dates_idx1, path + "date_split1.pt")
                torch.save(dates_idx2, path + "date_split2.pt")
                torch.save(dates_idx3, path + "date_split3.pt")
        else:
            raise ValueError("Unrecognized data split")
        
        if context is not None:
            context1 = context[:, :, :][: , :, dates_idx1]
            context2 = context[:, :, :][: , :, dates_idx2]
            context3 = context[:, :, :][: , :, dates_idx3]
        else:
            context1, context2, context3 = None, None, None
        return {"train": (values[:,:,dates_idx1], context1, dates1), "valid":(values[:,:,dates_idx2], context2, dates2), "test":(values[:,:,dates_idx3], context3, dates3)}    
    else:
        return {"train":(values, context, datetimes)}


def get_dataset_splits(path="datasets/", indiv_split=None, date_split=None, seed=None, save=False, context_by_individuals=False):
    """splits data from path. If str splits, will load given split, if float will save new split"""
    values, context, datetimes = load_data(path) #load dataset
    if not os.path.exists(path+"splits/"):
        os.makedirs(path+"splits/")
    split_path = path+"splits/"
    data_dict = train_test_split(values, context, datetimes, indiv_split, date_split, seed, context_by_individuals, split_path, save=True) #split randomly of according to paths, saves indexes

    if save: #saves pt files
        for key, (values, context, datetimes) in data_dict.items():
            torch.save(values, split_path + key + "_values.pt")
            if context is not None:
                torch.save(context, split_path + key + "_context.pt")
            torch.save(datetimes, split_path + key + "_datetimes.pt")
    return data_dict



def get_train_loaders(data_dict, batch_size, lags, horizon, by_date=True, subsets={}, path="", subset_mode="dates"):
    """returns dataloaders from data_dict as eventual subsets"""
    loaders_dict = {}
    for key, (values, context, datetimes) in data_dict.items():
        if key=="train":
           by_date_ = by_date
        else:
            by_date_ = True

        dataset = TimeSeriesDataset(values, datetimes, context, lags, horizon, by_date=by_date_)
        if subsets is not None:
            subset = subsets.get(key)
            if subset is not None and (type(subset)==str or (type(subset)==float and subset<1 and subset>0)):
                if type(subset)==str:
                    subset_indices = list(torch.load(subset, weights_only=False))
                elif type(subset)==float:
                    subset_indices = get_subset_indices(dataset, subset, subset_mode)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    torch.save(subset_indices, path + f"{key}_subset_indices_{subset}.pt")
                dataset = Subset(dataset, subset_indices, subset_mode)

        if key=="train":
            loaders_dict[key] = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        else:
            loaders_dict[key] = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
       
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

    inputs = torch.stack(inputs) #(bs, individuals, dim, lookback)
    inputs = inputs.view(-1, inputs.shape[-2], inputs.shape[-1]) #  (bs * individuals, dim, lookback)

    if contexts is not None:
        contexts = torch.stack(contexts)
        contexts = contexts.view(-1, contexts.shape[-2], contexts.shape[-1]) 
    targets = torch.stack(targets)
    targets = targets.view(-1, targets.shape[-2], targets.shape[-1])

    return inputs, contexts, targets


def aggregate_loaders(loaders, context_by_individuals=False, by_date=True):
    if (not context_by_individuals) and by_date: #other cases tODO
        values_list = []
        for loader in loaders:
            values_list.append(loader.dataset.values)
        extended_values = torch.cat(values_list, dim=0)
        shuffle = isinstance(loader.sampler, torch.utils.data.RandomSampler)
        extended_dataset = TimeSeriesDataset(extended_values, loader.dataset.datetimes, loader.dataset.context, loader.dataset.lags, loader.dataset.horizon, by_date=True)
        extended_loader = DataLoader(extended_dataset, batch_size=loader.batch_size, shuffle=shuffle, collate_fn=loader.collate_fn)
        return extended_loader
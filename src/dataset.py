import torch
from torch.utils.data import Dataset
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
    self.contexts, self.dim_context, _dates = self.context.shape
    
    assert _dates == self.dates, "not the same dates in values and context"
    assert self.dates > self.lag + self.horizon, "not enough dates for this lag and horizon"
    
    self.datetimes = datetimes
    self.by_date = by_date
    self.return_all_individuals, self.context_by_individuals = return_all_individuals, context_by_individuals


  def shape(self):
    return (self.individuals, self.dim_values, self.dates), (self.contexts, self.dim_context, self.dates)

  def __len__(self):
    if self.by_date:
       return self.dates
    else:
       return self.N_individuals
    
  def __getitem__(self, idx):
    """    """

    if self.by_date:
        if self.return_all_individuals: #1 batch = all individuals, batch of dates
            values = self.values[:, :, idx + self.lags + self.horizon] # (individuals, dim_values, lags+horizon)
            context = self.context[:, :, idx + self.lags + self.horizon] # (contexts, dim_context, lags+horizon)
            inputs = values[:, :, :self.lag] # (individuals, dim, lag)
            target = values[:, :, self.lag:] # (individuals, dim, horizon)
        else: #1 batch = 1 individual, batch of dates
            if self.seed is not None:
                np.random.seed(self.seed)
            indiv = np.random.randint(self.individuals)
            values = self.values[indiv, :, idx + self.lags + self.horizon] # (dim_values, lags+horizon)
            context = self.context[indiv, :, idx + self.lags + self.horizon] # (dim_context, lags+horizon)
            inputs = values[:, :, :self.lag] # (dim, lag)
            target = values[:, :, self.lag:] # (dim, horizon)

    else: #1 batch = batch of individuals, random date
        if self.seed is not None:
            np.random.seed(self.seed)
        t = np.random.randint(self.dates - self.lags - self.horizon)
        values = self.values[idx, :, t + self.lags + self.horizon] # (dim_values, lags+horizon)
        if self.context_by_individuals:
            context = self.context[idx, :, t + self.lags + self.horizon] # (dim_context, lags+horizon)
        else:
            context = self.context[:, :, t + self.lags + self.horizon] # (contexts, dim_context, lags+horizon)
        inputs = values[:, :, :self.lag] # (dim, lag)
        target = values[:, :, self.lag:] # (dim, horizon)

    return inputs, context, target


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
    values = torch.load(path + prefix + "values.pt")
    if os.path.exists(path + prefix + "context.pt"):
        context = torch.load(path + prefix + "context.pt")
    else:
        context = None
    datetimes = torch.load(path + prefix + "datetimes.pt", weights_only=False)
    return values, context, datetimes

def load_example(path="datasets/", prefix=""):
    input = torch.load(path + prefix + "input.pt")
    target = torch.load(path + prefix + "target.pt")
    if os.path.exists(path + prefix + "context.pt"):
        context = torch.load(path + prefix + "context.pt")
    else:
        context = None
    indiv, date = torch.load(path + prefix + "indivdate.pt", weights_only=False)
    return input, context, target, indiv, date


def load_datasets(path="datasets/"):
    files = [f for f in os.listdir(path) if ".pt" in f]
    data_dict = {}
    for file in files:
        name, key = file.split(".")[0].split("_")
        if data_dict.get(name) is None:
            data_dict[name] = {}
        data_dict[name][key] = torch.load(path + file, weights_only=False)
    return data_dict
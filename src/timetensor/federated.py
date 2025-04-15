from .dataset import load_data, train_test_split
import numpy as np
import os
import torch

from .utils import append_in_dict

class Client:
    def __init__(self, dataloaders, model=None, id=None, params={}):
        self.dataloaders = dataloaders
        self.model = model
        self.params = params
        self.id = id
        self.available = True
    
    def set_unavailable(self):
        self.available = False
    def set_available(self):
        self.available = True

    def get_size(self):
        shape = self.dataloaders["train"].dataset.shape
        if len(shape)==2: #context
             return shape[0][0]*shape[0][2]
        else:
            return shape[0]*shape[2]

def client_split(values, context, datetimes, splits, shuffle=True, replace=False, seed=None, context_by_individuals=False, path=""):
    """splits individuals according to splits"""
    
    N = len(splits)
    if type(splits[0]) == str:
        indices_list = []
        for split_path in splits:
            indices = torch.load(split_path, weights_only=False)
            indices_list.append(indices)
    else:
        if seed is not None:
            np.random.seed(seed)
        individuals = values.shape[0]

        remaining = list(range(individuals))
        indices_list = []
        for k in range(N):
            n = int(splits[k]*individuals) #local number of individuals
            if shuffle:
                indices = np.random.choice(remaining, n, replace=False)
                if replace is False:
                    remaining = [k for k in remaining if k not in indices]
            else:
                indices = remaining[:n]
                remaining = remaining[n:]
            indices_list.append(indices)
            torch.save(indices, path + f"node_{k}_indices.pt")

    if context_by_individuals:
        if context is None:
            return {f"node_{i}":(values[indices_list[i], :, :], None, datetimes) for i in range(N)}
        else:
            return {f"node_{i}":(values[indices_list[i], :, :], context[indices_list[i], :, :], datetimes) for i in range(N)}
    else:
        if context is None:
            return {f"node_{i}":(values[indices_list[i], :, :], None, datetimes) for i in range(N)}
        else:
            return {f"node_{i}":(values[indices_list[i], :, :], context, datetimes) for i in range(N)}


#def build_split_datasets(path, splits, shuffle=True, replace=False, seed=None, context_by_individuals=False):
def get_client_splits(path, splits, shuffle=True, replace=False, seed=None, context_by_individuals=False, save=False):
    """splits is a dict with keys the splits for nodes (or path) and for each node the indiv and date splits (or paths)"""
    #data_dict = load_datasets(path)
    values, context, datetimes = load_data(path)
    node_dict =  client_split(values, context, datetimes, list(splits.keys()), shuffle, replace, seed, context_by_individuals, path)

    node_split_dict = {}
    for k, (split, (indiv_split, date_split)) in enumerate(splits.items()):
        values, context, datetimes = node_dict[f"node_{k}"]
        subpath = path + f"node_{k}/"
        if not os.path.exists(subpath):
            os.makedirs(subpath)
        node_split_dict[f"node_{k}"] = train_test_split(values, context, datetimes, indiv_split, date_split, seed, context_by_individuals, subpath)
        
        if save:
            for key, (values, context, datetimes) in node_split_dict[f"node_{k}"].items():
                torch.save(values, subpath + key + "_values.pt")
                if context is not None:
                    torch.save(context, subpath + key + "_context.pt")
                torch.save(datetimes, subpath + key + "_datetimes.pt")
    return node_split_dict

import copy

class DefaultLocalServer():
    def __init__(self, client, learner):
        """
        clients: unique id and dataloaders, can store a model
        learner: optimizer and model to train
        """
        self.client = client #initialized with no model
        self.id = client.id
        self.learner = learner #initialized with a (random) model
        self.client.model = copy.deepcopy(self.learner.model)

    def receive(self, x):
        """what to do with the received data"""
        pass
    def assign_client_weights(self, weights):
        """assigns weights to client model"""
        self.client.model.load_state_dict(weights)
    def assign_learner_weights(self, weights):
        """resets the learner optimizer to provided weights"""
        self.learner.reset_model(weights)
        self.learner.reset_optimizer()
    def get_latest_weights(self):
        return self.learner.get_weights()

    def compute_epoch(self):
        """computes one training epoch"""
        loader = self.client.dataloaders["train"]
        for X_batch, context_batch, y_batch in loader:
            loss = self.learner.compute_step(X_batch, context_batch, y_batch)
        average_eval_dict = self.learner.eval(self.client.dataloaders["valid"])
        average_eval_dict2 = self.learner.eval(self.client.dataloaders["valid2"])
        return average_eval_dict, average_eval_dict2

    def compute_round(self, E):
        """comptes E epochs"""
        valid_losses = {}
        valid_losses2 = {}
        for e in range(E):
            average_eval_dict, average_eval_dict2 = self.compute_epoch()
            append_in_dict(valid_losses, average_eval_dict)
            append_in_dict(valid_losses2, average_eval_dict2)
        return valid_losses, valid_losses2

    def send(self):
        """what to send to the server"""
        pass


class DefaultGlobalServer():
    def __init__(self, model):
        self.update = model.state_dict() #random initial model

    def send(self, nodes):
        """send update to local nodes"""
        for node in nodes:
            node.receive(self.update)

    def aggregate(self, x):
        """aggregate information of x"""
        self.update = X
        return

    def receive(self, nodes):
        """receive and aggregate information of nodes"""
        pass

from .dataset import load_data, train_test_split
import numpy as np
import os
import torch

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


class LocalServer():
    def __init__(self, client, learner):
        """
        clients: unique id and dataloaders, can store a model
        learner: optimizer and model to train
        """
        self.client = client #initialized with no model
        self.learner = learner #initialized with a (random) model
        self.received = None #information received from server

    def assign_weights(self, weights):
        self.client.model.load_state_dict(weights)
    def assign_model(self, model):
        self.assign_weights(model.state_dict())

    def assign_learner_to_client(self):
        self.assign(self.learner.model)
    def assign_server_to_learner(self, weights): #resets the learner optimizer. Requires modifications to keep inter-round states
        self.learner.reset_model(weights)
        self.learnder.reset_optimizer()

    def compute_epoch(self):
        loader = self.client["train"]
        for X_batch, context_batch, y_batch in loader:
            loss = self.learner.compute_step(X_batch, context_batch, y_batch)
        average_eval_dict self.learner.eval(self.client["valid"])
        average_eval_dict2 = self.learner.eval(self.client["valid2"])
        return average_eval_dict, average_eval_dict2

    def compute_local(self, E)
        for e in range(E):
            self.compute_epoch()
        latest_model = self.learner.get_weights()
        self.send = latest_model
    def send(self):
        return self.latest_model


class GlobalServer():
    def __init__(self):
        self.weights = weights

    def average_weights(weights, importance):
        """weights: multidim tensor (N, ...)
        importance: (N)
        """
        return torch.mean([weights[k]*importance[k] for k in range(weights)])

    def aggregate(self, client_weights, clients_importance):
        self.update = average_weights(client_weights, clients_importance)

    def send(locals):
        for local in locals:
            local.assign_server_to_learner(self, self.update)

    def receive(locals):
        client_weights = []
        for local in locals:
            client_weights.append(local.send())
            clients_importance.append(local.client.get_size())
        clients_importance = np.array(clients_importance)
        clients_importance = clients_importance / np.sum(clients_importance)
        return client_weights, clients_importance

    def round(locals, strategy="uniform"):
        #receive local updates
        client_weights, clients_importance = self.receive(locals)

        #aggregate updates
        if strategy == "size":
            aggregate(client_weights, clients_importance)
        elif strategy == "uniform":
            aggregate(client_weights, [1/len(locals) for k in range(len(locals))])
        elif type(strategy)==list:
            assert type(strategy[0]) == float
            aggregate(client_weights, strategy)
        else:
            raise ValueError("Unrecognized strategy")

        #send update
        self.send(locals)

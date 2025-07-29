from .dataset import load_data, temporal_split
import numpy as np
import os
import torch
import copy

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

    def get_weights(self):
        if self.model is not None:
            return self.model.state_dict()
        else:
            return None

def client_split(values, context, datetimes, nodes, shuffle=True, seed=None, context_by_individuals=True, path=""):
    """splits individuals according to splits"""
    
    individuals = values.shape[0]
    if nodes is None:
        N = individuals
        indices_list = [[k] for k in range(individuals)]
    elif type(nodes)==list and type(nodes[0])==float:
        N = len(nodes)
        if seed is not None:
            np.random.seed(seed)
        if shuffle:
            user_list = np.random.permutation(individuals)
        else:
            user_list = list(range(individuals))
        indices_list = []
        frac1 = 0
        for k in range(N):
            frac2 = frac1+nodes[k]
            indices = user_list[int(frac1*individuals):int(frac2*individuals)]
            indices_list.append(indices)
            frac1=frac2
            torch.save(indices, path + "indices/" + f"node_{k}_indices.pt")
    elif type(nodes) == str:
        indices_list = [torch.load(nodes+node_name, weights_only=False) for node_name in os.listdir(nodes)]
        N = len(indices_list)
    else:
        raise ValueError("Unrecognized nodes")

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


def get_client_splits(data_path, nodes, splits, shuffle=True, seed=None, context_by_individuals=True, save=False, path=""):
    """splits is a dict with keys the splits for nodes (or path) and for each node the indiv and date splits (or paths)"""
    values, context, datetimes = load_data(data_path)
    node_dict =  client_split(values, context, datetimes, nodes, shuffle, seed, context_by_individuals, path)

    node_split_dict = {}
    for node_name, (values, context, datetimes) in node_dict.items():
        node_split_dict[node_name] = temporal_split(values, context, datetimes, splits, seed, save=False)
        if save:
            subpath = path + node_name + "/"
            if not os.path.exists(subpath):
                os.makedirs(subpath)
            for key, (values, context, datetimes) in node_split_dict[node_name].items():
                torch.save(values, subpath + key + "_values.pt")
                if context is not None:
                    torch.save(context, subpath + key + "_context.pt")
                torch.save(datetimes, subpath + key + "_datetimes.pt")
    return node_split_dict


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
    def assign_client_learner(self):
        self.assign_learner_weights(self.client.model.state_dict())
    def get_latest_weights(self):
        return self.learner.get_weights()
    def get_client_weights(self):
        return self.client.model.state_dict()
        
    def compute_epoch(self):
        """computes one training epoch"""
        loader = self.client.dataloaders["train"]
        for X_batch, context_batch, y_batch in loader:
            loss = self.learner.compute_step(X_batch, context_batch, y_batch)
        average_eval_dict = self.learner.eval(self.client.dataloaders["valid"])
        return average_eval_dict

    def compute_round(self, E):
        """comptes E epochs"""
        valid_losses = {}
        for e in range(E):
            average_eval_dict = self.compute_epoch()
            append_in_dict(valid_losses, average_eval_dict)
        return valid_losses

    def send(self):
        """what to send to the server"""
        pass

    def eval(self):
        return self.learner.eval(self.client.dataloaders["test"])


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


class DefaultScheme():
    def __init__(self, server, nodes, shadow_server=None, shadow_nodes=None):
        self.server, self.shadow_server = server, shadow_server
        self.nodes, self.shadow_nodes = nodes, shadow_nodes
        self.N = len(self.nodes)
    
        self.valid_losses = {f"node_{k}": {} for k in range(self.N)}
        self.shadow_valid_losses = {f"node_{k}": {} for k in range(self.N)}
        self.global_valid_losses = {}

    def compute_round(self, E):
        pass
    def compute_scheme(self, K, E=1):
        pass


def average_nodes(nodes_dict, weights=None):
    """averages dicts on nodes"""
    main_dict = {} #{loss_name: [nodes: losses]}
    N = len(nodes_dict)
    for values_dict in nodes_dict.values(): #nodes
        for loss_name, values in values_dict.items(): #losses
            if loss_name not in main_dict:
                main_dict[loss_name] = []
            main_dict[loss_name].append(values)
    return {loss_name: np.average(np.array(values), axis=0, weights=weights) for (loss_name, values) in main_dict.items()}


def eval_nodes(nodes, weights=None):
    """evaluate a list of nodes on their local models or provided weights"""
    N = len(nodes)
    losses_dict = {} #{loss_name: [nodes: losses]}
    for k in range(N):
        if weights is not None:
            nodes[k].assign_learner_weights(weights)
        losses = nodes[k].eval()
        for loss_name in losses:
            if loss_name not in losses_dict:
                losses_dict[loss_name] = []
            losses_dict[loss_name].append(losses[loss_name])
    return losses_dict
    
def get_node_metrics(losses_dict, size_weights):#{loss_name: [nodes: losses]}
    """returns avg mean and mean(flop10) of each loss"""
    N = len(size_weights)
    m = int((9*N)/10)
    avg_losses_dict = {key: np.average(values, weights=size_weights) for (key, values) in losses_dict.items()}
    mean_losses_dict = {key: np.mean(values) for (key, values) in losses_dict.items()}
    flop_losses_dict = {key: np.mean(np.sort(values)[m:]) for (key, values) in losses_dict.items()}
    return avg_losses_dict, mean_losses_dict, flop_losses_dict
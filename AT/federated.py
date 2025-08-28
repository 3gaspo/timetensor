from .dataset import load_data, get_dataset_splits
import numpy as np
import os
import torch
import copy
import shutil

from .utils import append_in_dict

class Client:
    """node in a federated setting"""
    def __init__(self, dataloaders, model=None, id=None, params={}):
        self.dataloaders = dataloaders #train, valid, test loaders
        self.model = model #currently held model

        #optional
        self.id = id
        self.params = params
        self.available = True
    def set_unavailable(self):
        self.available = False
    def set_available(self):
        self.available = True

    def get_size(self):
        """returns dataset size (scalar)"""
        shape = self.dataloaders["train"].dataset.shape
        if len(shape)==2: #context
             return shape[0][0]*shape[0][2]  #users * dates (values)
        else:
            return shape[0]*shape[2] #users * dates

    def get_weights(self):
        """returns held model weights"""
        if self.model is not None:
            return self.model.state_dict()
        else:
            return None


def client_split(values, context, datetimes, nodes, shuffle=True, context_by_individuals=True, save_path="", reshuffle=False):
    """splits individuals according to splits"""
    
    individuals = values.shape[0]
    if nodes is None:
        nodes= [1 for _ in range(individuals)]

    split_dir = save_path + str(nodes) + "/"
    if reshuffle:
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
        if shuffle:
            user_list = np.random.permutation(individuals)
        else:
            user_list = list(range(individuals))
        indices_list = []
        
        idx = 0
        for k in range(len(nodes)):
            idx_bis = idx + nodes[k]
            if type(nodes[0])==float:
                idx_bis = int(idx_bis * individuals)
            indices = user_list[idx:idx_bis]
            indices_list.append(indices)
            idx=idx_bis
            torch.save(indices, split_dir + f"node{k}.pt")
    else:
        indices_list = [torch.load(split_dir+node, weights_only=False) for node in os.listdir(split_dir)]


    if context_by_individuals:
        if context is None:
            return {f"node_{i}":(values[indices_list[i], :, :], None, datetimes) for i in range(len(nodes))}
        else:
            return {f"node_{i}":(values[indices_list[i], :, :], context[indices_list[i], :, :], datetimes) for i in range(len(nodes))}
    else:
        if context is None:
            return {f"node_{i}":(values[indices_list[i], :, :], None, datetimes) for i in range(len(nodes))}
        else:
            return {f"node_{i}":(values[indices_list[i], :, :], context, datetimes) for i in range(len(nodes))}


##TODO read et complete


def get_client_splits(data_path, splits, indiv_split, date_splits, shuffle=True, context_by_individuals=True, save_path=None, reshuffle=True):
    """splits is a dict with keys the splits for nodes (or path) and for each node the indiv and date splits (or paths)"""
    values, context, datetimes = load_data(data_path)
    
    if not os.path.exists(data_path+"nodes/"):
        os.makedirs(data_path+"nodes/")
    if save_path is None:
        split_path = data_path+"nodes/"
    else:
        split_path = save_path
    
    if type(nodes)==str:
        nodes = nodes.split(";")
        nodes = [float(node) for node in nodes]
    if int(nodes[0])==nodes[0]: #it is a list of ints
        nodes = [int(node) for node in nodes]

    #shuffle = shuffling of idxs to split, reshuffle=redo split
    node_dict =  client_split(values, context, datetimes, nodes, shuffle, context_by_individuals, split_path, reshuffle)

    node_split_dict = {}
    for node_name, data in node_dict.items(): #data=(values, context, datetimes)
        save_path + str(splits) + "/"
        node_split_dict[node_name] = get_dataset_splits(data_path, indiv_split, date_splits, context_by_individuals, save_path, reshuffle, data)
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
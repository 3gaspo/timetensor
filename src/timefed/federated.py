from .dataset import load_data, get_dataset_splits
import numpy as np
import os
import torch
import copy
import shutil

from ..utils import append_in_dict

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
        shape = self.dataloaders["train"].dataset.shape[0]
        return shape[0]*shape[2] #users * dates

    def get_weights(self):
        """returns held model weights"""
        if self.model is not None:
            return self.model.state_dict()
        else:
            return None


def client_split(values, context, datetimes, nodes, shuffle=True, context_by_individuals=True, save_path="", reshuffle=False):
    """randomly split individuals according to nodes [ints]"""
    
    individuals = values.shape[0]
    if nodes is None:
        nodes= [1 for _ in range(individuals)]
        split_dir = save_path + "nodes_all/"
    else:
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
            return {f"node{i}":(values[indices_list[i], :, :], None, datetimes) for i in range(len(nodes))}
        else:
            return {f"node{i}":(values[indices_list[i], :, :], context[indices_list[i], :, :], datetimes) for i in range(len(nodes))}
    else:
        if context is None:
            return {f"node{i}":(values[indices_list[i], :, :], None, datetimes) for i in range(len(nodes))}
        else:
            return {f"node{i}":(values[indices_list[i], :, :], context, datetimes) for i in range(len(nodes))}


def get_client_splits(data_path, nodes, indiv_split, date_splits, shuffle=True, context_by_individuals=True, save_path=None, reshuffle=True):
    """returns random split as nodes data dict. equivalent de fetch_training data(cluster, aggregate=False)"""
    values, context, datetimes = load_data(data_path)
    
    if not os.path.exists(data_path+"nodes/"):
        os.makedirs(data_path+"nodes/")
    if save_path is None:
        split_path = data_path+"FL_clusters/"
    else:
        split_path = save_path
    
    if nodes == "None":
        nodes = None
    if nodes is not None:
        if type(nodes)==str:
            nodes = nodes.split(";")
            nodes = [float(node) for node in nodes]
        if int(nodes[0])==nodes[0]: #it is a list of ints
            nodes = [int(node) for node in nodes]

    #shuffle = shuffling of idxs to split, reshuffle=redo split
    node_dict =  client_split(values, context, datetimes, nodes, shuffle, context_by_individuals, split_path, reshuffle)

    node_split_dict = {}
    for node_name, data in node_dict.items(): #data=(values, context, datetimes)
        node_split_dict[node_name] = get_dataset_splits(data_path, indiv_split, date_splits, context_by_individuals, save_path, reshuffle, data)
    return node_split_dict



class DefaultLocalServer():
    def __init__(self, client, learner):
        """
        clients: unique id and dataloaders, can store a model
        learner: optimizer and model to train
        """
        self.client = client #has no model
        self.learner = learner #has a (potentially random) model

        self.id = client.id
        #clients saves initial model (weights and architecture)
        self.client.model = copy.deepcopy(self.learner.model)

    def assign_client_weights(self, weights):
        """assigns weights to client model"""
        self.client.model.load_state_dict(weights)
    def assign_learner_weights(self, weights):
        """resets the learner optimizer to provided weights"""
        self.learner.reset_model(weights)
        self.learner.reset_optimizer()
    def assign_client_learner(self):
        """reset learner to client's held model"""
        self.assign_learner_weights(self.client.model.state_dict())
    def assign_learner_client(self):
        """set client to learner's latest model"""
        self.assign_client_weights(self.get_latest_weights())
    def get_latest_weights(self):
        """returns latest learner's weights"""
        return self.learner.get_weights()
    def get_client_weights(self):
        """returns held clients weights"""
        return self.client.model.state_dict()
    
    def validate(self):
        return self.learner.eval(self.client.dataloaders["valid1"])
    def eval(self):
        return self.learner.eval(self.client.dataloaders["test1"])

    def compute_epoch(self):
        """computes one training epoch"""
        loader = self.client.dataloaders["train"]
        for X_batch, context_batch, y_batch, indiv, date in loader:
            #update learner model with 1 step
            loss = self.learner.compute_step(X_batch, context_batch, y_batch)
        return self.validate()

    def compute_round(self, E):
        """comptes E epochs"""
        valid_losses = {}
        for e in range(E):
            average_eval_dict = self.compute_epoch()
            append_in_dict(valid_losses, average_eval_dict)
        return valid_losses
    
    def receive(self, x):
        """what to do with the received data"""
        pass
    def send(self):
        """what to send to the server"""
        pass


class DefaultGlobalServer():
    def __init__(self, model):
        """
        model: provided (potentially random)
        updates: to be sent to local nodes
        """
        self.update = model.state_dict() #random initial model

    def receive(self, nodes):
        """receive and aggregate information of nodes
        nodes: list of LocalServers
        """
        pass
    def aggregate(self, x):
        """aggregates information of x"""
        pass
    def send(self, nodes):
        """send update to local nodes"""
        for node in nodes:
            node.receive(self.update)


class DefaultScheme():
    def __init__(self, E, K, B, server, nodes, shadow_server=None, shadow_nodes=None):
        """
        shadow_nodes: fully local mirror nodes
        shadow_server: fully central mirror server
        """
        self.server, self.shadow_server = server, shadow_server
        self.nodes, self.shadow_nodes = nodes, shadow_nodes
        self.M = len(self.nodes)
        
        self.E, self.K = E, K
        #fraction of sampled nodes at each round
        if B <= 1:
            self.B = int(B*self.M)
        else:
            self.B = B
        self.training_losses = {f"node_{k}": {} for k in range(self.M)}
        self.valid_losses = {f"node_{k}": {} for k in range(self.M)}
        self.shadow_valid_losses = {f"node_{k}": {} for k in range(self.M)}
        self.global_valid_losses = {}

    def compute_round(self, epochs):
        pass
    def compute_scheme(self):
        for k in range(self.K):
            self.compute_round(self.E)


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

def get_node_metrics(losses_dict, size_weights):#{loss_name: [nodes: losses]}
    """returns avg mean and mean(flop10) of each loss"""
    N = len(size_weights)
    m = int((9*N)/10)
    avg_losses_dict = {key: np.average(values, weights=size_weights) for (key, values) in losses_dict.items()}
    mean_losses_dict = {key: np.mean(values) for (key, values) in losses_dict.items()}
    flop_losses_dict = {key: np.mean(np.sort(values)[m:]) for (key, values) in losses_dict.items()}
    return avg_losses_dict, mean_losses_dict, flop_losses_dict

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
    
from src.timetensor.dataset import aggregate_loaders_dict
from src.timetensor.pipeline import Learner
from src.timetensor.visu import plot_multi_losses

def launch_training(client_builder, server_builder, scheme_builder, loaders_dicts, global_model,
    E, K, criterion, lr, eval_losses, device, logger,
    save_dir, save_name, retrain=True, verbose=1):
    
    losses_dir = save_dir + "losses/"
    if not os.path.exists(losses_dir):
        os.makedirs(losses_dir)

    #nodes
    M = len(loaders_dicts)
    nodes, sizes = [], []
    shadow_nodes = []
    for k in range(M):
        node = f"node{k}"
        client = Client(loaders_dicts[node], id=f'n{k}')
        shadow_client =  Client(copy.deepcopy(loaders_dicts[node]), id=f's{k}')
        learner = Learner(copy.deepcopy(global_model), criterion, lr, eval_losses, device=device)
        shadow_learner = Learner(copy.deepcopy(global_model), criterion, lr, eval_losses, device=device)
        node = client_builder(client, learner)
        shadow_node = client_builder(shadow_client, shadow_learner)
        nodes.append(node)
        shadow_nodes.append(shadow_node)
        sizes.append(client.get_size())
    size_weights = np.array(sizes) / np.sum(sizes)

    #server
    global_shadow_learner = Learner(copy.deepcopy(global_model), criterion, lr, eval_losses, device=device)
    server_client = Client(aggregate_loaders_dict(list(loaders_dicts.values())), id="server")
    shadow_server = client_builder(server_client, global_shadow_learner)
    server = server_builder(global_model)
    logger.info("Built all nodes")
    
    #scheme
    scheme = scheme_builder(server, nodes, shadow_server, shadow_nodes)
    if retrain:
        logger.info("Starting training...")
        training_losses, valid_losses, shadow_valid_losses, global_valid_losses = scheme.compute_scheme(verbose=verbose)
        logger.info(f"Finished")

        #save losses
        torch.save(training_losses, losses_dir + f"training_losses.pt")
        torch.save(valid_losses, losses_dir + f"valid_losses.pt")
        torch.save(shadow_valid_losses, losses_dir + f"shadow_losses.pt")
        torch.save(global_valid_losses, losses_dir + f"global_losses.pt")

        for k in range(M):
            path = save_dir + f"nodes/node_{k}/"
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(nodes[k].get_latest_weights(), path + "federated_model.pt")
            torch.save(shadow_nodes[k].get_latest_weights(), path + "shadow_model.pt")
        torch.save(server.update, save_dir + "global_model.pt")
        torch.save(shadow_server.get_latest_weights(), save_dir + "shadow_model.pt")


    else:
        training_losses = torch.load(losses_dir + f"training_losses.pt", weights_only=False)
        valid_losses = torch.load(losses_dir + f"valid_losses.pt", weights_only=False)
        shadow_valid_losses = torch.load(losses_dir + f"shadow_losses.pt", weights_only=False)
        global_valid_losses = torch.load(losses_dir + f"global_losses.pt", weights_only=False)    

    # avg_train_losses = average_nodes(training_losses)
    # shadow_avg_losses = average_nodes(shadow_valid_losses)
    # avg_losses = average_nodes(valid_losses)
    mean_train_losses =  average_nodes(training_losses, size_weights)
    mean_losses =  average_nodes(valid_losses, size_weights)
    shadow_mean_losses =  average_nodes(shadow_valid_losses, size_weights)

    #plots
    if M<=10:
        for k in range(M):
            path = save_dir + f"nodes/node_{k}/"
            for key in eval_losses:
                plot_multi_losses({
                    "valid": training_losses[f"node_{k}"][key], "shadow valid": shadow_valid_losses[f"node_{k}"][key]},
                    path, f"valid_{key}.pdf", f"Training {key} of {save_name}, node_{k}", x_every=E)
    for key in eval_losses:
        plot_multi_losses({
            f"mean valid {key}": mean_losses[key],
            f"global valid {key}": global_valid_losses[key]},
            save_dir + "plots/", f"mean_valid_{key}.pdf", f"Training {key} of {save_name}", x_every=None)
        plot_multi_losses({
            f"mean training {key}": mean_train_losses[key],
            f"mean shadows {key}": shadow_mean_losses[key]},
            save_dir + "plots/", f"mean_training_{key}.pdf", f"Training {key} of {save_name}", x_every=E)
    
    return server, shadow_server, nodes, shadow_nodes, size_weights

from .utils import save_results

def launch_eval(global_model, nodes, shadow_server, eval_losses, size_weights, save_dir, save_name, logger):
    M = len(nodes)
    logger.info("Computing test metrics")
    tune_losses_dict = eval_nodes(nodes)
    global_losses_dict = shadow_server.eval()
    tune_avg_loss_dict, tune_mean_losses_dict, tune_flop_losses_dict = get_node_metrics(tune_losses_dict, size_weights)

    for k in range(M):
        flat_losses_dict = eval_nodes(nodes, global_model.state_dict())
        flat_avg_loss_dict, flat_mean_losses_dict, flat_flop_losses_dict = get_node_metrics(flat_losses_dict, size_weights)

    for loss_name in eval_losses:
        save_results(global_losses_dict[loss_name], save_dir, f"{loss_name}_mean_results.json", save_name, f"Global {loss_name}")
        save_results(tune_avg_loss_dict[loss_name], save_dir, f"{loss_name}_mean_results.json", save_name, f"Tuned Uniform {loss_name}")
        save_results(tune_mean_losses_dict[loss_name], save_dir, f"{loss_name}_mean_results.json", save_name, f"Tuned Weighted {loss_name}")
        save_results(tune_flop_losses_dict[loss_name], save_dir, f"{loss_name}_mean_results.json", save_name, f"Tuned Flop10 {loss_name}")
        save_results(flat_avg_loss_dict[loss_name], save_dir, f"{loss_name}_mean_results.json", save_name, f"Flat Uniform {loss_name}")
        save_results(flat_mean_losses_dict[loss_name], save_dir, f"{loss_name}_mean_results.json", save_name, f"Flat Weighted {loss_name}")
        save_results(flat_flop_losses_dict[loss_name], save_dir, f"{loss_name}_mean_results.json", save_name, f"Flat Flop10 {loss_name}")
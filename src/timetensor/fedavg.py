import torch
import numpy as np
from .federated import DefaultGlobalServer, DefaultLocalServer, DefaultScheme
from src.timetensor.utils import append_in_dict

class LocalFedAvg(DefaultLocalServer):
    def __init__(self, client, learner):
        """
        clients: unique id and dataloaders, can store a model
        learner: optimizer and model to train
        """
        super(LocalFedAvg, self).__init__(client, learner)

    def receive(self, weights):
        self.assign_client_weights(weights)
        self.assign_learner_weights(weights)
    def send(self):
        return self.get_latest_weights()


class GlobalFedAvg(DefaultGlobalServer):
    def __init__(self, model):
        super(GlobalFedAvg, self).__init__(model)

    def receive(self, nodes, strategy="uniform"):
        client_weights = []
        clients_importance = []
        C = len(nodes)
        for node in nodes:
            client_weights.append(node.send())
            if strategy == "uniform":
                clients_importance.append(1/C)
            else:
                clients_importance.append(node.client.get_size())
        clients_importance = np.array(clients_importance)
        if strategy == "size":
            clients_importance = clients_importance / np.sum(clients_importance)    
        self.update = self.aggregate(client_weights, clients_importance)

    def aggregate(self, weights, importances):
        C = len(importances)
        averaged_weights = {}
        for key in weights[0].keys():
            raw_weights = [weights[i][key].clone().detach().cpu() for i in range(C)]
            averaged_weights[key] = sum([raw_weights[i]*importances[i] for i in range(C)])
        return averaged_weights


class FedAvgScheme(DefaultScheme):
    def __init__(self, server, nodes, shadow_server=None, shadow_nodes=None):
        super(FedAvgScheme, self).__init__(server, nodes, shadow_server, shadow_nodes)

    def compute_round(self, E):
        self.server.send(self.nodes) #send global model to nodes
        
        if self.shadow_server is not None:
            shadow_losses = self.shadow_server.compute_round(E) #to do : devrait être seulement 1 pour comparer à global averages. Mais probleme pour plot après
            append_in_dict(self.global_valid_losses, shadow_losses)
        
        for k in range(self.N):
            if self.shadow_nodes is not None:
                shadow_losses = self.shadow_nodes[k].compute_round(E)
                append_in_dict(self.shadow_valid_losses[f"node_{k}"], shadow_losses)

            losses = self.nodes[k].compute_round(E) #computes E steps of local training
            append_in_dict(self.valid_losses[f"node_{k}"], losses)
            
        self.server.receive(self.nodes) #averages updates 
    

    def compute_scheme(self, K, E=1, plus=None):
        for k in range(K):
            self.compute_round(E)
        self.server.send(self.nodes)

        if plus:
            if self.shadow_nodes is not None:
                shadow_losses = self.shadow_nodes[k].compute_round(E)
                append_in_dict(self.shadow_valid_losses[f"node_{k}"], shadow_losses)
            losses = self.nodes[k].compute_round(E) #computes E steps of local training
            append_in_dict(self.valid_losses[f"node_{k}"], losses)
        
        return self.valid_losses, self.shadow_valid_losses, self.global_valid_losses
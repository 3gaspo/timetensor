import torch
import numpy as np
from .federated import DefaultGlobalServer, DefaultLocalServer

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
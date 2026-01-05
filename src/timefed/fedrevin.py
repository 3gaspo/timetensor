import torch

from .fedavg import LocalFedAvg, FedAvgScheme
from timetensor.utils import append_in_dict


class LocalFedRevin(LocalFedAvg):
    def __init__(self, client, learner):
        """
        clients: unique id and dataloaders, can store a model
        learner: optimizer and model to train
        """
        super(LocalFedRevin, self).__init__(client, learner)

    def reset_revin(self):
        model = self.learner.model
        model.gamma.data = torch.ones(1, model.dim, 1, device=model.gamma.device)
        model.beta.data = torch.zeros(1, model.dim, 1, device=model.beta.device)


class FedRevinScheme(FedAvgScheme):
    def __init__(self, server, nodes, shadow_server=None, shadow_nodes=None, reset_revin=False):
        super(FedRevinScheme, self).__init__(server, nodes, shadow_server, shadow_nodes)
        self.reset_revin = reset_revin

    def compute_round(self, E, verbose=1):
        self.server.send(self.nodes) #send global model to nodes
        
        if self.shadow_server is not None:
            shadow_losses = self.shadow_server.compute_round(E) #to do : devrait être seulement 1 pour comparer à global averages. Mais probleme pour plot après
            append_in_dict(self.global_valid_losses, shadow_losses)
        
        for k in range(self.N):
            if self.shadow_nodes is not None:
                shadow_losses = self.shadow_nodes[k].compute_round(E)
                append_in_dict(self.shadow_valid_losses[f"node_{k}"], shadow_losses)
            
            if self.reset_revin:
                self.nodes[k].reset_revin()
            losses = self.nodes[k].compute_round(E) #computes E steps of local training
            if verbose:
                print(f"==Epoch {k+1} done==")
            append_in_dict(self.valid_losses[f"node_{k}"], losses)
            
        self.server.receive(self.nodes) #averages updates
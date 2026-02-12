import numpy as np
from .federated import DefaultGlobalServer, DefaultLocalServer, DefaultScheme
from ..utils import append_in_dict

class LocalFedAvg(DefaultLocalServer):
    def __init__(self, client, learner):
        """
        clients: unique id and dataloaders, can store a model
        learner: optimizer and model to train
        """
        super(LocalFedAvg, self).__init__(client, learner)

    def receive(self, weights): #received global model, reset client and learner
        self.assign_client_weights(weights)
        self.assign_learner_weights(weights)
    def send(self): #send updated model to global
        return self.get_latest_weights()


class GlobalFedAvg(DefaultGlobalServer):
    def __init__(self, model): #initialize with random weights
        super(GlobalFedAvg, self).__init__(model)

    def receive(self, nodes, strategy="uniform"):
        """
        nodes: list of LocalServers
        """
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

    def aggregate(self, client_weights, clients_importance):
        """averages weights using importance weighting"""
        C = len(clients_importance)
        averaged_weights = {}
        for key in client_weights[0].keys():
            raw_weights = [client_weights[i][key].clone().detach().cpu() for i in range(C)]
            averaged_weights[key] = sum([raw_weights[i]*clients_importance[i] for i in range(C)])
        return averaged_weights


class FedAvgScheme(DefaultScheme):
    """nodes and shadows are expected to have loaded same architecture"""
    def __init__(self, E, K, B, server, nodes, shadow_server=None, shadow_nodes=None, server_side="full", plus=True):
        super(FedAvgScheme, self).__init__(E, K, B, server, nodes, shadow_server, shadow_nodes)
        self.server_side, self.plus = server_side, plus

    def compute_round(self, verbose=0):
        """compute updates and aggregation on sampled nodes"""
        if self.B != self.M:
            sampled_idx = np.random.permutation(self.M)[:self.B]
        else:
            sampled_idx = list(range(self.M))  

        self.server.send(self.nodes) #synchronize global with ALL nodes
        
        if self.shadow_server is not None:
            if self.server_side == "full":
                shadow_losses = self.shadow_server.compute_round(self.E)
            elif self.server_side == "partial":
                shadow_losses = self.shadow_server.compute_round(1)
            append_in_dict(self.global_valid_losses, shadow_losses)
        
        for k in range(self.M):
            #loss of received model
            append_in_dict(self.valid_losses[f"node_{k}"], self.nodes[k].validate())
        
        for k in sampled_idx:
            #computes E steps of local training
            if self.shadow_nodes is not None:
                shadow_losses = self.shadow_nodes[k].compute_round(self.E)
                append_in_dict(self.shadow_valid_losses[f"node_{k}"], shadow_losses)
            #loss of training
            losses = self.nodes[k].compute_round(self.E) 
            if verbose:
                print(f"Node {k} done")
            append_in_dict(self.training_losses[f"node_{k}"], losses)
            
        sampled_nodes = [self.nodes[k] for k in sampled_idx]
        self.server.receive(self.nodes) #averages updates 
    

    def compute_scheme(self, verbose=1):
        """compute K rounds"""
        #fedavg
        for t in range(self.K):
            self.compute_round(verbose=max(0,verbose-1))
            if verbose:
                print(f"Round {t+1} done")
        #last global model
        self.server.send(self.nodes)

        #finetuning
        if self.plus:
            if self.shadow_server is not None:
                if self.server_side=="full":
                    shadow_losses = self.shadow_server.compute_round(self.E)
                elif self.server_side=="partial":
                    shadow_losses = self.shadow_server.compute_round(1)
                append_in_dict(self.global_valid_losses, shadow_losses)

            for k in range(self.M):
                #loss of received model
                append_in_dict(self.valid_losses[f"node_{k}"], self.nodes[k].validate())

            for k in range(self.M):
                if self.shadow_nodes is not None:
                    shadow_losses = self.shadow_nodes[k].compute_round(self.E)
                    append_in_dict(self.shadow_valid_losses[f"node_{k}"], shadow_losses)
                    
                #loss of training
                losses = self.nodes[k].compute_round(self.E)
                if verbose:
                    print(f"Node {k} done")
                append_in_dict(self.training_losses[f"node_{k}"], losses)
            if verbose:
                print(f"Last fine-tuning done")
        
            for k in range(self.M):
                self.nodes[k].assign_learner_client()
                print(f"debug assigned last model, beta:", self.nodes[k].client.model.beta.data.detach().cpu().numpy()[0][0][0])
        return self.training_losses, self.valid_losses, self.shadow_valid_losses, self.global_valid_losses
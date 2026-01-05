
from .federated import DefaultLocalServer
import torch
import copy


class LocalFedmIN(DefaultLocalServer):
    def __init__(self, client, learner, device):
        """
        clients: unique id and dataloaders, can store a model
        learner: optimizer and model to train
        """
        super(LocalFedmIN, self).__init__(client, learner)
        self.dim = client.model.dim
        self.device = device
        self.alpha = torch.ones(1, self.dim, 1, device=self.device)
        self.beta = torch.zeros(1, self.dim, 1, device=self.device)


    def receive(self, weights):
        local_weights = copy.deepcopy(weights)
        print("debug, received beta", local_weights["beta"])
        local_weights["alpha"] = self.alpha.clone().to(self.device)
        local_weights["beta"] = self.beta.clone().to(self.device)
        self.assign_client_weights(local_weights)
        self.assign_learner_weights(local_weights)

    def send(self):
        update = copy.deepcopy(self.get_latest_weights())
        print("debug, update beta", update["beta"])
        self.alpha = update["alpha"].detach().clone().to(self.device)
        self.beta = update["beta"].detach().clone().to(self.device)
        update["alpha"] = torch.ones(1, self.dim, 1, device=self.device)
        update["beta"] = torch.zeros(1, self.dim, 1, device=self.device)
        print("debug, sent beta", update["beta"])
        return update

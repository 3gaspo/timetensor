import torch

from .fedavg import LocalFedAvg


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
        model.gamma.zero_grad()
        model.beta.data = torch.zeros(1, model.dim, 1, device=model.beta.device)
        model.beta.zero_grad()
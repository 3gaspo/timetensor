import torch
import torch.nn as nn
import torch.optim as optim
from time import perf_counter
import numpy as np

from .utils import get_normal_stats, nloss, average_loss


class Learner:
    def __init__(self, model, criterion, lr, eval_losses, device=None, optimizer=None, scheduler=None, normalized_criterion=True):

        
        if criterion is None:
            self.criterion = nn.MSELoss() # mean over 1/B * 1/dim * 1/horizon
        else:
            self.criterion = criterion
        if optimizer is None:
            self.optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer
        if scheduler is not None:
            self.scheduler = scheduler(self.optimizer)
        else:
            self.scheduler = scheduler
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model = model
        self.model.to(self.device)

        self.normalized_criterion = normalized_criterion
        self.eval_losses = eval_losses

    def compute_step(self, X_batch, context_batch, y_batch, frozen_modules=None):
        """computes forward and backward on batch"""
        self.model.train()
        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
        if context_batch is not None:
            context_batch = context_batch.to(self.device)
        
        if self.normalized_criterion:
            mean, std = get_normal_stats(X_batch) # (B, dim, 1)
        
        self.optimizer.zero_grad()
        predictions = self.model(X_batch, context_batch)

        if self.normalized_criterion:
            loss = nloss(self.criterion, predictions, y_batch, mean, std)
        else:
            loss = self.criterion(predictions, y_batch)

        if frozen_modules is not None:
            for frozen_module in frozen_modules:
                frozen_module.zero_grad()

        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return loss.item()


    def eval(self, loader, verbose=0, return_all=False):
        """evaluates model on loader and returns mean loss"""    
        self.model.eval()
        losses = {}
        t1 = perf_counter()
        with torch.no_grad():
            for X_batch, context_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                if context_batch is not None:
                    context_batch = context_batch.to(self.device)
                mean, std = get_normal_stats(X_batch)
                
                predictions = self.model(X_batch, context_batch) #normalization done (or not) inside model
                for loss_name, criterion in self.eval_losses.items():
                    loss = criterion(predictions, y_batch).cpu()
                    normalized_loss = nloss(criterion, predictions, y_batch, mean, std).cpu()

                    if losses.get(loss_name) is None:
                        losses[loss_name] = []
                        losses["N"+loss_name] = []
                    losses[loss_name] += loss
                    losses["N"+loss_name] += normalized_loss
        t2 = perf_counter()
        if verbose:
            print(f"Evaluation done in {(t2-t1)/60:.3f} min")
        
        
        eval_dict = {key: torch.stack(losses[key]) for key in losses}
        if return_all:
            return eval_dict
        else:
            average_eval_dict = average_loss(eval_dict)
            return average_eval_dict



#def train_model(model, loaders_dict, lr, criterion=None, normalized_criterion=True, print_freq=50, eval_freq=10, optimizer=None, device=None, scheduler=None, verbose=1, eval_losses=None):
def train_model(learner, loaders_dict, print_freq=50, eval_freq=10, verbose=1, do_eval=True):
    """trains model in learner on loaders and returns train and valid losses"""
    
    #data
    train_loader = loaders_dict["train"]
    valid_loader = loaders_dict.get("valid")
    steps = len(train_loader)

    if verbose:
        print(f"Using device: {learner.device}")
        print(f"Number of steps (batches): {len(train_loader)}, eval_freq: {eval_freq}, print_freq: {print_freq}")

    train_losses = []
    valid_losses = {}
    t1 = perf_counter()

    #training
    step = 0
    for X_batch, context_batch, y_batch in train_loader:
        step += 1
        loss = learner.compute_step(X_batch, context_batch, y_batch)
        train_losses.append(loss) #loss of batch
        
        if do_eval and (step == 1 or step % eval_freq == 0 or step == steps):
            average_eval_dict = learner.eval(valid_loader)
            for loss_name, loss_value in average_eval_dict.items():
                if loss_name not in valid_losses:
                    valid_losses[loss_name] = []
                valid_losses[loss_name].append(loss_value)
            if step == 1 or step % print_freq == 0 or step == steps:
                print(f"Step {step} | " + " | ".join([f"valid {loss_name} : {loss_value:.4f}" for loss_name, loss_value in average_eval_dict.items()]))

    t2 = perf_counter()
    if verbose:
        print(f"Training done in {(t2-t1)/60:.3f} min")
    return train_losses, valid_losses




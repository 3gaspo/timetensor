import torch
import torch.nn as nn
import torch.optim as optim
from time import perf_counter
import numpy as np

from .utils import get_normal_stats, average_loss, append_in_dict
from .utils import unroll_windows
from .utils import normalize

class Loss():
    def __init__(self, loss, mean=None, std=None, mode=None):
        self.loss = loss #e.g nn.MSELoss()
        self.mode = mode

        self.mean = mean
        self.std = std
        self.standard_norm = (mean is not None and std is not None)

    def __call__(self, pred, y, mean=None, std=None):
        if self.standard_norm:
            pred = normalize(pred, self.mean, self.std)
            y = normalize(y, self.mean, self.std)
        if self.mode == "instance":
            assert (mean is not None and std is not None)
            pred = normalize(pred, mean, std)
            y = normalize(y, mean, std)
        elif self.mode == "relative":
            assert mean is not None
            mean = torch.where(mean != 0, mean, 1)
            pred, y = pred/mean, y/mean
        return self.loss(pred, y)




class Learner:
    def __init__(self, model, criterion, lr, eval_losses, device=None, optimizer=None, scheduler=None, do_train=True, pytorch=True):
        """
        optimizer: to be called on model.parameters() and lr
        scheduler: to be called on optimizer(model)
        """
        if criterion is None:
            self.criterion = Loss(nn.MSELoss()) # mean over 1/B * 1/dim * 1/horizon
        else:
            self.criterion = criterion
        self.eval_losses = eval_losses

        if optimizer is None:
            self.optimizer = lambda model: optim.Adam(model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer
        if scheduler is not None:
            self.scheduler = lambda optimizer: scheduler(optimizer)
        else:
            self.scheduler = scheduler

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model = model
        self.pytorch = pytorch
        if self.pytorch:
            self.model.to(self.device)
        self.do_train = do_train
        if self.pytorch and self.do_train:
            self.reset_optimizer()

    def reset_model(self, weights):
        self.model.load_state_dict(weights)
    def reset_optimizer(self):
        self.curent_optimizer = self.optimizer(self.model)
        if self.scheduler is not None:
            self.current_scheduler = self.scheduler(self.curent_optimizer)
    def get_weights(self):
        if self.pytorch:
            return self.model.state_dict()
        else:
            return self.model.reg.coef_

    def compute_step(self, X_batch, context_batch, y_batch):
        """computes forward and backward on batch"""
        assert self.model is not None and self.do_train and self.pytorch
        self.model.train()
        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
        if context_batch is not None:
            context_batch = context_batch.to(self.device)
        mean, std = get_normal_stats(X_batch) # (B, dim, 1)
        
        self.curent_optimizer.zero_grad()

        predictions = self.model(X_batch, context_batch)
        loss = self.criterion(predictions, y_batch, mean, std)

        loss.backward()
        self.curent_optimizer.step()
        if self.scheduler is not None:
            self.current_scheduler.step()

        return loss.item()

    def fit(self, loader):
        assert not self.pytorch
        Xtrain, Ytrain = unroll_windows(loader, shuffle=True)
        self.model.fit(Xtrain.cpu(), Ytrain.cpu())

    def eval(self, loader, verbose=0, return_all=False, logger=None, runs=1):
        """evaluates model on loader and returns mean loss"""
        losses = {}
        t1 = perf_counter()
        if self.pytorch:
            self.model.eval()
            with torch.no_grad():
                for run in range(runs):
                    for X_batch, context_batch, y_batch in loader:
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                        if context_batch is not None:
                            context_batch = context_batch.to(self.device)
                        mean, std = get_normal_stats(X_batch)
                        
                        predictions = self.model(X_batch, context_batch) #normalization done (or not) inside model
                        for loss_name, criterion in self.eval_losses.items():
                            loss = criterion(predictions, y_batch, mean, std).cpu() # (bs * individuals, dim, horizon)
                            if losses.get(loss_name) is None:
                                losses[loss_name] = []
                            losses[loss_name] += loss.tolist()
        else:
            Xtest, Ytest = unroll_windows(loader)
            predictions = self.model(Xtest)
            mean, std = get_normal_stats(Xtest)
            for loss_name, criterion in self.eval_losses.items():
                loss = criterion(predictions, Ytest, mean, std).cpu()
                losses[loss_name] = loss.tolist()
        t2 = perf_counter()

        if verbose:
            if logger is not None:
                logger.info(f"Evaluation done in {(t2-t1)/60:.3f} min")
            else:
                print(f"Evaluation done in {(t2-t1)/60:.3f} min")
        
        eval_dict = {key: torch.tensor(losses[key]) for key in losses} # (ndates * individuals, dim, horizon)
        if return_all:
            return eval_dict
        else:
            average_eval_dict = average_loss(eval_dict)
            return average_eval_dict



def train_model(learner, loaders_dict, epochs=1, print_freq=50, eval_freq=10, verbose=1, do_eval=True, logger=None):
    """trains model in learner on loaders and returns train and valid losses"""
    
    #data
    train_loader = loaders_dict["train"]
    valid_loader = loaders_dict.get("valid")
    valid_loader2 = loaders_dict.get("valid2")
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch

    if verbose:
        if logger is not None:
            logger.info(f"Using device: {learner.device}")
            logger.info(f"Training {epochs} epochs of {steps_per_epoch} batches ({total_steps} steps): , eval_freq: {eval_freq}, print_freq: {print_freq}")
        else:
            print(f"Using device: {learner.device}")
            print(f"Training {epochs} epochs of {steps_per_epoch} batches ({total_steps} steps): , eval_freq: {eval_freq}, print_freq: {print_freq}")

    train_losses = []
    valid_losses = {}
    valid_losses2 = {}
    t1 = perf_counter()

    #training
    step = 0
    for epoch in range(epochs):
        for X_batch, context_batch, y_batch in train_loader:
            step += 1
            loss = learner.compute_step(X_batch, context_batch, y_batch)
            train_losses.append(loss) #loss of batch
            
            if do_eval and (step == 1 or step % eval_freq == 0 or step == total_steps):

                #valid eval
                average_eval_dict = learner.eval(valid_loader)
                average_eval_dict2 = learner.eval(valid_loader2)
                append_in_dict(valid_losses, average_eval_dict)
                append_in_dict(valid_losses2, average_eval_dict2)

                if verbose and (step == 1 or step % print_freq == 0 or step == total_steps):
                    if logger is not None:
                        logger.info(f"Step {step} | " + " | ".join([f"valid1 {loss_name} : {loss_value:.4f}" for loss_name, loss_value in average_eval_dict.items()]))
                    else:
                        print(f"Step {step} | " + " | ".join([f"valid1 {loss_name} : {loss_value:.4f}" for loss_name, loss_value in average_eval_dict.items()]))

    t2 = perf_counter()
    if verbose:
        if logger is not None:
            logger.info(f"Training done in {(t2-t1)/60:.3f} min")
        else:
            print(f"Training done in {(t2-t1)/60:.3f} min")
    return train_losses, valid_losses, valid_losses2




import torch
import torch.nn as nn
import torch.optim as optim
from time import perf_counter
import numpy as np

from .utils import get_normal_stats


def nloss(loss, pred, y, mean, std):
    """returns normalized loss"""
    normal_pred = (pred - mean)/std
    normal_y = (y - mean)/std
    return loss(normal_pred, normal_y)


def compute_step(model, X_batch, y_batch, context_batch, normalized_criterion=True, frozen_modules=None):
    """computes forward and backward on batch"""
    if context_batch is not None:
        context_batch = context_batch.to(device)
    
    if normalized_criterion:
        mean, std = get_normal_stats(X_batch) # (B, dim, 1)
    
    optimizer.zero_grad()
    predictions = model(X_batch, context_batch)

    if normalized_criterion:
        loss = nloss(criterion, predictions, y_batch, mean, std)
    else:
        loss = criterion(predictions, y_batch)

    if frozen_modules is not None:
        for frozen_module in frozen_modules:
            frozen_module.zero_grad()

    loss.backward()

    return loss.item()


def train_model(model, loaders_dict, lr, criterion=None, normalized_criterion=True, print_freq=50, eval_freq=10, optimizer=None, device=None, scheduler=None, verbose=1, eval_losses=None):
    """trains model and returns model, train and valid losses"""
    
    #model
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)    

    #optimization
    if criterion is None:
        criterion = nn.MSELoss() # mean over 1/B * 1/dim * 1/horizon
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    if scheduler is not None:
        scheduler = scheduler(optimizer)
    
    #data
    train_loader = loaders_dict["train"]
    valid_loader = loaders_dict.get("valid")
    if valid_loader is not None:
        do_eval = True
        if eval_losses is None:
            eval_losses = {"MSE":nn.MSELoss(reduction="none")}
        valid_losses = {}
        for key in eval_losses:
            valid_losses[key] = []
            valid_losses["N"+key] = []
    else:
        do_eval = False
    steps = len(train_loader)

    if verbose:
        print(f"Using device: {device}")
        print(f"Number of steps (batches): {len(train_loader)}, eval_freq: {eval_freq}, print_freq: {print_freq}")

    train_losses = []

    t1 = perf_counter()

    #training
    step = 0
    model.train()
    for X_batch, context_batch, y_batch in train_loader:
        step += 1
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        loss = compute_step(model, X_batch, y_batch, context_batch, normalized_criterion)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        train_losses.append(loss) #loss of batch
        
        if do_eval and (step == 1 or step%eval_freq == 0 or step==steps):
            valid_loss = average_loss(eval_model(model, valid_loader, device, eval_losses))
            for loss_name, loss_value in valid_loss.items():
                valid_losses[loss_name].append(loss_value)
            if step == 1 or step%print_freq == 0 or step==steps:
                print(f"Step {step} | " + " | ".join([f"valid {loss_name} : {loss_value:.4f}" for loss_name, loss_value in valid_loss.items()]))
            model.train()

    t2 = perf_counter()
    model.eval()
    if verbose:
        print(f"Training done in {(t2-t1)/60:.3f} min")
    return model, train_losses, valid_losses


def eval_model(model, loader, device, eval_losses, verbose=0):
    """evaluates model on loader and returns mean loss
    if return_all, all individual losses
    """    
    losses = {}

    model.to(device)
    model.eval()
    t1 = perf_counter()
    with torch.no_grad():
        for X_batch, context_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            if context_batch is not None:
                context_batch = context_batch.to(device)
            mean, std = get_normal_stats(X_batch)
            
            predictions = model(X_batch, context_batch)
            for loss_name, criterion in eval_losses.items():
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
    return {key: torch.stack(losses[key]) for key in losses}


def average_loss(eval_losses):
    mean_losses = {}
    for loss_name, losses in eval_losses.items():
        mean_losses[loss_name] = losses.mean()
    return mean_losses

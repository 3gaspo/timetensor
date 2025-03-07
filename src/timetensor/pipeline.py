import torch
import torch.nn as nn
import torch.optim as optim
from time import perf_counter

from .utils import get_normal_stats


def nloss(loss, pred, y, mean, std):
    """returns normalized loss"""
    normal_pred = (pred - mean)/std
    normal_y = (y - mean)/std
    return loss(normal_pred, normal_y)


def train_model(model, loaders_dict, lr, criterion=None, normalized_criterion=True, n_prints=10, n_evals=100, optimizer=None, device=None, scheduler=None, verbose=1, eval_losses=None):
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
    else:
        do_eval = False
    steps = len(train_loader)
    eval_freq = max(steps // n_evals, 1)
    print_freq = max(steps // n_prints, 1)

    if verbose:
        print(f"Using device: {device}")
        print(f"Number of steps (batches) : {len(train_loader)}")

    train_losses = []
    valid_losses = []

    t1 = perf_counter()

    #training
    step = 0
    model.train()
    for X_batch, context_batch, y_batch in train_loader:
        step += 1
        X_batch, context_batch, y_batch = X_batch.to(device), context_batch.to(device), y_batch.to(device)
        
        if normalized_criterion:
            mean, std = get_normal_stats(X_batch) # (B, dim, 1)
        
        optimizer.zero_grad()
        predictions = model(X_batch, context_batch)

        if normalized_criterion:
            loss = nloss(criterion, predictions, y_batch, mean, std)
        else:
            loss = criterion(predictions, y_batch)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item()) #loss of batch
        if scheduler is not None:
            scheduler.step()
        
        if do_eval and (step == 1 or step%eval_freq == 0 or step==steps):
            valid_loss = average_loss(eval_model(model, valid_loader, device, eval_losses))
            valid_losses.append(valid_loss)
            if i%(print_freq)==0 or i==steps-1:
                print(f"Step {i}" + " | ".join([f"valid {loss_name} : {loss_value:.4f}" for loss_name, loss_value in valid_loss.items()]))
            model.train()
        i+=1

    t2 = perf_counter()
    model.eval()
    if verbose:
        print(f"Training done in {(t2-t1)/60:.3f} min")
    return model, train_losses, valid_losses


def eval_model(model, loader, device, eval_losses):
    """evaluates model on loader and returns mean loss
    if return_all, all individual losses
    """
    if eval_losses is None:
        eval_losses = {"MSE":nn.MSELoss(reduction="none")}
    
    losses = {}

    model.to(device)
    model.eval()
    with torch.no_grad():
        for X_batch, context_batch, y_batch in loader:
            X_batch, context_batch, y_batch = X_batch.to(device), context_batch.to(device), y_batch.to(device)            
            mean, std = get_normal_stats(X_batch)
            
            predictions = model(X_batch, context_batch)
            for loss_name, criterion in eval_losses.items():
                loss = criterion(predictions, y_batch).cpu()
                nloss = nloss(criterion, predictions, y_batch, mean, std).cpu()

                if losses.get(loss_name) is None:
                    losses[loss_name] = []
                losses[loss_name] += loss
                losses["N"+loss_name] += nloss

    return losses

def average_loss(eval_losses):
    mean_losses = {}
    for loss_name, losses in eval_losses.items():
        mean_losses[loss_name] = losses.mean()
    return mean_losses

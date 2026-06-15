import torch
import torch.nn as nn
import torch.optim as optim
from time import perf_counter
import os

from .utils import get_normal_stats, unroll_windows, normalize, save_results, set_seed
from .visu import plot_losses, plot_multi_losses, plot_serie, plot_named_example, plot_horizon_errors, plot_pred, plot_horizon_errors, plot_weights, plot_errors
from .dataset import set_random_data, fetch_example_data


## Losses

class Loss():
    def __init__(self, loss, mean=None, std=None, mode=None, eps=1e-8):
        self.loss = loss #e.g nn.MSELoss()
        self.mode = mode

        self.mean = mean
        self.std = std
        self.eps = eps
        self.name = None

    def __call__(self, pred, y, mean=None, std=None):
        if self.mode == "standard": #apply standard normalization
            assert (self.mean is not None and self.std is not None)
            pred = normalize(pred, self.mean, self.std, self.eps)
            y = normalize(y, self.mean, self.std, self.eps)
        elif self.mode == "denorm": #remove standard normalization
            assert (self.mean is not None and self.std is not None)
            pred = (self.std + self.eps) * pred + self.mean
            y = (self.std + self.eps) * y + self.mean
        elif self.mode == "instance": #apply instance normalization
            assert (mean is not None and std is not None)
            pred = normalize(pred, mean, std, self.eps)
            y = normalize(y, mean, std, self.eps)
        elif self.mode == "relative": #apply relative normalization
            assert (mean is not None)
            mean = torch.abs(mean) + self.eps
            pred, y = pred/mean, y/mean
        # elif self.mode == "normalize_y":
        #     assert (mean is not None and std is not None)
        #     y = normalize(y, mean, std, self.eps)
        # elif self.mode == "denormalize_pred":
        #     assert (mean is not None and std is not None)
        #     pred = pred*(std+self.eps) + mean
        return self.loss(pred, y)


def get_losses(criterion_name, mean=None, std=None, complete_evaluation=False):
    """returns criterion and relevant eval losses from specified criterion name"""
    if criterion_name == "MSE":
        criterion = Loss(nn.MSELoss())
    elif criterion_name == "sMSE":
        criterion = Loss(nn.MSELoss(), mean, std, mode ="standard")
    elif criterion_name == "nMSE":
        criterion = Loss(nn.MSELoss(), mode="instance")
    elif criterion_name == "rMSE":
        criterion = Loss(nn.MSELoss(), mode="relative")
    # elif criterion_name == "normalize_y":
    #     criterion = Loss(nn.MSELoss(), mode="normalize_y")
    # elif criterion_name == "denormalize_pred":
    #     criterion = Loss(nn.MSELoss(), mode="denormalize_pred")
    else:
        raise ValueError("Unknown criterion name")
    criterion.name = criterion_name
    # if criterion_name == "normalize_y":
    #     eval_losses = {
    #         "NMSE": Loss(nn.MSELoss(reduction="none"), mode="normalize_y"),
    #         "MSE": Loss(nn.MSELoss(reduction="none"), mode="denormalize_pred"),
    #         }
    # else:
    if complete_evaluation:
        eval_losses = {
            "MSE": Loss(nn.MSELoss(reduction="none")),
            "MAE": Loss(nn.L1Loss(reduction="none")),
            "nMSE": Loss(nn.MSELoss(reduction="none"), mode="instance"), 
            "rMSE": Loss(nn.MSELoss(reduction="none"), mode="relative")
        }
    else:
        eval_losses = {
            "MSE": Loss(nn.MSELoss(reduction="none")),
            "nMSE": Loss(nn.MSELoss(reduction="none"), mode="instance"), 
        }
    return criterion, eval_losses
    
## Learners

class TorchLearner:
    def __init__(self, model, criterion, lr, eval_losses, device=None, optimizer=None, scheduler=None):
        self.model_type = "pytorch"
        self.criterion, self.eval_losses = criterion, eval_losses

        if optimizer is None:
            self.optimizer = lambda model: optim.Adam(model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer

        self.scheduler = None
        if scheduler is not None:
            self.scheduler = lambda optimizer: scheduler(optimizer)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model = model.to(self.device)
        self.current_optimizer = None

    def reset_model(self, weights):
        self.model.load_state_dict(weights)
    def reset_optimizer(self):
        self.current_optimizer = self.optimizer(self.model)
        if self.scheduler is not None:
            self.current_scheduler = self.scheduler(self.current_optimizer)
    def get_weights(self):
        return self.model.state_dict()

    def compute_step(self, X_batch, context_batch, y_batch):
        """computes forward and backward on batch"""
        assert self.current_optimizer is not None
        self.model.train()
        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
        if context_batch is not None:
            context_batch = context_batch.to(self.device)
        mean, std = get_normal_stats(X_batch) # (B, dim, 1)
        
        self.current_optimizer.zero_grad()

        predictions = self.model(X_batch, context_batch)
        loss = self.criterion(predictions, y_batch, mean, std)

        loss.backward()
        self.current_optimizer.step()
        if self.scheduler is not None:
            self.current_scheduler.step()

        return loss.item()

    def eval(self, loader, return_mode="mean", runs=1, thresholds={}, seed=None):
        """evaluates model on loader and returns mean loss"""
        losses = {}
        exotics = {}
        counts = {}

        set_seed(seed)
        self.model.eval()
        with torch.inference_mode():
            for run in range(runs):
                for X_batch, context_batch, y_batch, indiv_batch, date_batch in loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    if context_batch is not None:
                        context_batch = context_batch.to(self.device)
                    mean, std = get_normal_stats(X_batch)
                    
                    predictions = self.model(X_batch, context_batch)
                    
                    for loss_name, criterion in self.eval_losses.items():
                        loss = criterion(predictions, y_batch, mean, std).detach() # (bs * (individuals), dim, horizon)
                        
                        aggr_loss = loss.mean(dim=(1,2)) # (bs * (individuals))
                        if thresholds.get(loss_name) is not None:
                            if loss_name not in exotics:
                                exotics[loss_name] = []
                            high_mask = aggr_loss > thresholds[loss_name]
                            high_indices = high_mask.nonzero(as_tuple=True)[0]
                            for idx in high_indices:
                                i = int(idx)
                                exotics[loss_name].append({
                                    "indiv": indiv_batch[i],
                                    "date": date_batch[i],
                                    "loss": float(aggr_loss[i].cpu())
                                })
                        if return_mode == "indiv":
                            if loss_name not in losses:
                                losses[loss_name] = {}
                            for i, indiv in enumerate(indiv_batch):
                                if indiv not in losses[loss_name]:
                                    losses[loss_name][indiv] = []
                                losses[loss_name][indiv].append(loss[i].mean().item()) # # {indiv: [ (1) x (steps * bs)]]
                        elif return_mode == "mean":
                            if loss_name not in losses:
                                losses[loss_name] = 0.0
                                counts[loss_name] = 0
                            losses[loss_name] += loss.sum(dim=0).mean().item() # (1)
                            counts[loss_name] += loss.shape[0]
                        else:
                            if loss_name not in losses:
                                losses[loss_name] = []
                            if return_mode == "all": #may cause memory issues on cpu if too many samples
                                losses[loss_name] += [l.cpu() for l in loss] # [ (dim, horizon) x (steps*bs*(individuals))]
                            elif return_mode == "steps":
                                losses[loss_name] += aggr_loss.detach().tolist() # [ (1) x (steps*bs*(individuals))]
                            elif return_mode == "dim":
                                losses[loss_name].append(loss.sum(dim=0).cpu()) # [ (dim, horizon) x steps] 
                                counts[loss_name] = counts.get(loss_name, 0) + loss.shape[0]

        for loss_name, criterion in self.eval_losses.items():           
            if return_mode == "all":
                losses[loss_name] = torch.stack(losses[loss_name], dim=0) # ((steps*bs*individuals), dim, horizon)
            elif return_mode == "steps":
                losses[loss_name] = torch.tensor(losses[loss_name]) # (steps*bs*individuals)
            elif return_mode == "indiv":
                for indiv in losses[loss_name]:
                    losses[loss_name][indiv] = torch.tensor(losses[loss_name][indiv]) # {indiv: (steps * bs)]
            elif return_mode == "dim":
                losses[loss_name] = torch.stack(losses[loss_name], dim=0).sum(dim=0) # (dim, horizon)
                losses[loss_name] /= counts[loss_name]
            elif return_mode == "mean":
                losses[loss_name] /= counts[loss_name] # (1)

        return losses, exotics
        
class ScikitLearner:
    def __init__(self, model, criterion, eval_losses):
        self.model_type = "scikit-learn"
        self.criterion, self.eval_losses = criterion, eval_losses
        self.model = model

    def get_weights(self):
        return self.model.reg.coef_

    def fit(self, loader):
        Xtrain, Ytrain = unroll_windows(loader)#, shuffle=True)
        self.model.fit(Xtrain.cpu(), Ytrain.cpu())

    def eval(self, loader, return_mode="mean", runs=1, thresholds={}):
        """evaluates model on loader and returns mean loss"""
        losses = {}
        exotics = {}

        Xtest, Ytest = unroll_windows(loader)
        predictions = self.model(Xtest)
        mean, std = get_normal_stats(Xtest)
        for loss_name, criterion in self.eval_losses.items():
            losses[loss_name] = criterion(predictions, Ytest, mean, std).cpu() # (steps, dim, horizon)
        if return_mode == "mean":
            for loss_name, criterion in self.eval_losses.items():
                losses[loss_name] = losses[loss_name].mean().item() # scalar
        elif return_mode == "dim":
            for loss_name, criterion in self.eval_losses.items():
                losses[loss_name] = losses[loss_name].mean(dim=0) # (dim, horizon)

        return losses, exotics

def load_learner(model, criterion, lr, eval_losses, device, optimizer=None, scheduler=None):
    """loads correct model learner"""
    if model.model_type == "pytorch":
        return TorchLearner(model, criterion, lr, eval_losses, device, optimizer, scheduler)
    elif model.model_type == "scikit-learn":
        return ScikitLearner(model, criterion, eval_losses)
    else:
        raise ValueError(f"Unknown model type: {model.model_type}")

def train_model(learner, loaders_dict, epochs=1, print_freq=50, eval_freq=10, verbose=1, do_eval=True, logger=None, eval_runs=1, weight_follow=None, seed=None):
    """trains model in learner on loaders and returns train and valid losses"""
    
    #data
    train_loader = loaders_dict["train"]
    do_eval = False
    valid_keys = []
    for key in loaders_dict:
        if "valid" in key:
            do_eval = True
            valid_keys.append(key)
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch

    if verbose and logger is not None:
        logger.info(f"Using device: {learner.device}")
        logger.info(f"Training {epochs} epochs of {steps_per_epoch} batches ({total_steps} steps): , eval_freq: {eval_freq}, print_freq: {print_freq}")

    train_losses = []
    valid_losses = {key: {} for key in valid_keys}
    weights_dict = {}
    t1 = perf_counter()

    set_seed(seed)
    #training
    step = 0
    for epoch in range(epochs):
        for X_batch, context_batch, y_batch, indiv, date in train_loader:
            step += 1
            loss = learner.compute_step(X_batch, context_batch, y_batch)
            train_losses.append(loss)

            if do_eval and (step == 1 or step % eval_freq == 0 or step == total_steps):
                #valid eval
                for valid_key in valid_keys:
                    valid_loss, _ = learner.eval(loaders_dict[valid_key], runs=eval_runs) #DO NOT set seed, or it will reset seed of training dataloaders as well
                    for loss_key in valid_loss:
                        if loss_key not in valid_losses[valid_key]:
                            valid_losses[valid_key][loss_key] = []
                        valid_losses[valid_key][loss_key].append(valid_loss[loss_key])
                    if valid_key == "valid1":
                        if verbose and logger is not None and (step == 1 or step % print_freq == 0 or step == total_steps):
                            logger.info(f"Step {step} | " + " | ".join([f"valid1 {loss_name} : {loss_value:.4f}" for loss_name, loss_value in valid_loss.items()]))
                #weights
                if weight_follow is not None:
                    weights = weight_follow(learner.model)
                    for weight_key in weights:
                        if weight_key not in weights_dict:
                            weights_dict[weight_key] = []
                        weights_dict[weight_key].append(weights[weight_key])

    t2 = perf_counter()
    if verbose:
        if logger is not None:
            T = t2-t1
            logger.info(f"Training done in {T/60:.3f} min")
            logger.info(f"Average time per step: {T/total_steps:.3f} s")
    return train_losses, valid_losses, weights_dict


def launch_training(model, normalization, criterion, lr, epochs, loaders_dict, eval_losses, device, save_dir, save_name, eval_freq, print_freq, logger, optimizer=None, scheduler=None, weight_follow=None, verbose=1, save=False, seed=None):
    """launches training of model"""
    model_name, criterion_name = model.model_name, criterion.name
    if not os.path.exists(save_dir + "plots/"):
        os.makedirs(save_dir + "plots/")

    learner = load_learner(model, criterion, lr, eval_losses, device, optimizer, scheduler)

    no_training = (model_name in ["persistence", "repeat", "lookback", "expected"]) and ((normalization is None) or normalization in ["None", "standard", "instance", "IN"])
    if no_training:
        if verbose:
            logger.info("No training needed")
    
    #scikit learn .fit
    elif learner.model_type == "scikit-learn":
        if verbose:
            logger.info("Starting scikit-learn fitting...")
        learner.fit(loaders_dict["train"])
        if verbose:
            logger.info("End of training")
    
    #pytorch training
    else:
        if verbose:
            logger.info(f"Starting training pytorch with lr={lr}")
        learner.reset_optimizer()
        train_losses, valid_losses, followed_weights = train_model(learner, loaders_dict, epochs=epochs, logger=logger, eval_runs=1, eval_freq=eval_freq, print_freq=print_freq, verbose=verbose,weight_follow=weight_follow, seed=seed)

        if save:
            torch.save(learner.model.state_dict(), save_dir + "trained_model.pt")
            torch.save(train_losses, save_dir + f"train_losses.pt")
            for key in valid_losses:
                torch.save(valid_losses[key], save_dir + f"{key}_losses.pt")
            torch.save(followed_weights, save_dir + f"followed_weights.pt")
        
        #plots
        for loss_name in eval_losses:
            valid_dict = {}
            for key in valid_losses:
                if valid_losses[key].get(loss_name) is not None:
                    valid_dict[key] = valid_losses[key][loss_name]
            if loss_name == criterion_name or (loss_name=="nMSE" and "nMSE" in criterion_name):
                plot_losses(train_losses, valid_dict, save_dir + "plots/", f"{loss_name}_plot.pdf", f"Training {loss_name} of {save_name}", eval_freq=eval_freq)
            else:
                if valid_dict != {}:
                    plot_multi_losses(valid_dict, save_dir + "plots/", f"{loss_name}_plot.pdf", f"Training {loss_name} of {save_name}", eval_freq=eval_freq)
        for weight_name in followed_weights:
            plot_serie(followed_weights[weight_name], save_dir + "plots/", f"{weight_name}.pdf", title=f"{weight_name} during training")
        if verbose:
            logger.info("End of training")        
    
    #weights
    plot_weights(model, save_dir + "plots/", save_name)
    # if verbose:
    #     if (normalization is not None) and (("revin" in normalization) or ("mIN" in normalization and "cmIN" not in normalization)):
    #         params = {"beta": model.beta.data.detach().cpu().numpy()[0][0][0], "alpha": model.alpha.data.detach().cpu().numpy()[0][0][0]}
    #         logger.info(f"Final modulations: {params}")
    #     elif (normalization is not None and "cmIN" in normalization):
    #         params = {f"beta_{k}": value.data.detach().cpu().numpy()[0][0][0] for k,value in enumerate(model.betas)}
    #         logger.info(f"Final modulations: {params}")
    
    return learner


def launch_eval(learner, loaders_dict, eval_losses, save_dir, save_name, complete_evaluation, save=False, results_dir=None, mode="test", denormalize_stats=None, runs=1, return_mode="dim", thresholds={}, seed=None):
    """evaluating model script"""
    if results_dir is None:
        results_dir = save_dir
    if not os.path.exists(save_dir + "plots/"):
        os.makedirs(save_dir + "plots/")

    exotics_dict = {}
    for key in loaders_dict:
        if mode == "all" or mode in key:
            losses, exotics = learner.eval(loaders_dict[key], return_mode=return_mode, runs=runs, thresholds=thresholds, seed=seed)
            exotics_dict[key] = exotics
            if save:
                torch.save(losses, save_dir + f"{key}_losses.pt")

            for loss_name in eval_losses:
                mean = losses[loss_name].mean()
                if denormalize_stats is not None: #TODO check si utile
                    mean *= denormalize_stats["train"]["std"]**2
                save_results(mean, results_dir, f"{key}_mean_results.json", save_name, f"{loss_name}")
                if complete_evaluation:
                    std = losses[loss_name].std()
                    save_results(std, results_dir, f"{key}_std_results.json", save_name, f"{loss_name}")
                    if return_mode == "all":
                        plot_errors(losses[loss_name].mean(dim=(1,2)), save_dir + "plots/", f"{key}_{loss_name}.pdf", f"{mode} 1 {loss_name} of {save_name} : {mean}")
                        plot_horizon_errors(losses[loss_name].mean(dim=(0,1)), save_dir + "plots/", f"{key}_horizon_{loss_name}.pdf", f"{mode} 1 {loss_name} of {save_name} : {mean}")
                    elif return_mode == "dim":
                        plot_horizon_errors(losses[loss_name].mean(dim=0), save_dir + "plots/", f"{key}_horizon_{loss_name}.pdf", f"{mode} {loss_name} 1 of {save_name} : {mean}")
    return exotics_dict

def launch_example(data_path, model, lags, horizon, device, save_dir, save_name, use_context=True):
    """runs model on example"""
    ex_dir = data_path + "examples/" + f"{lags}_{horizon}/"
    if not os.path.exists(ex_dir):
        set_random_data(data_path, lags, horizon, name="rand")
        plot_named_example(ex_dir, f"rand")
    dico = fetch_example_data(ex_dir)
    for data_name, data_tuple in dico.items():
        x, c, y = data_tuple[0].unsqueeze(0).to(device), data_tuple[1], data_tuple[2].unsqueeze(0).to(device)
        if not use_context:
            c = None
        if c is not None:
            c = c.unsqueeze(0).to(device)
        pred = model(x,c)
        plot_pred(x[0,0].cpu().detach().tolist(), y[0,0].cpu().detach().tolist(), pred[0,0].cpu().detach().tolist(), save_dir + "examples/", f"{data_name}_predictions.pdf", f"Example {data_name} prediction for {save_name}")
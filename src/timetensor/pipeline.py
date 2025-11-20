import torch
import torch.nn as nn
import torch.optim as optim
from time import perf_counter
import os
from tqdm import tqdm

from .utils import get_normal_stats, append_in_dict, unroll_windows, normalize, save_results
from .visu import plot_losses, plot_multi_losses, plot_serie, plot_named_example, plot_horizon_errors, plot_pred, plot_horizon_errors, plot_weights, plot_errors
from .dataset import set_random_data, fetch_example_data

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
            assert (mean is not None and std is not None)
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
    elif criterion_name == "MMSE":
        criterion = Loss(nn.MSELoss(), mean, std, mode ="standard")
    elif criterion_name == "NMSE":
        criterion = Loss(nn.MSELoss(), mode="instance")
    elif criterion_name == "RMSE":
        criterion = Loss(nn.MSELoss(), mode="relative")
    elif criterion_name == "normalize_y":
        criterion = Loss(nn.MSELoss(), mode="normalize_y")
    elif criterion_name == "denormalize_pred":
        criterion = Loss(nn.MSELoss(), mode="denormalize_pred")
    else:
        raise ValueError("Unknown criterion name")
    criterion.name = criterion_name
    if criterion_name == "normalize_y":
        eval_losses = {
            "NMSE": Loss(nn.MSELoss(reduction="none"), mode="normalize_y"),
            "MSE": Loss(nn.MSELoss(reduction="none"), mode="denormalize_pred"),
            }
    else:
        if complete_evaluation:
            eval_losses = {
                "MSE": Loss(nn.MSELoss(reduction="none")),
                "MAE": Loss(nn.L1Loss(reduction="none")),
                "NMSE": Loss(nn.MSELoss(reduction="none"), mode="instance"), 
                "RMSE": Loss(nn.MSELoss(reduction="none"), mode="relative")
            }
        else:
            eval_losses = {
                "MSE": Loss(nn.MSELoss(reduction="none")),
                "NMSE": Loss(nn.MSELoss(reduction="none"), mode="instance"), 
            }
    return criterion, eval_losses
    

class Learner:
    def __init__(self, model, criterion, lr, eval_losses, device=None, optimizer=None, scheduler=None, do_train=True, mode="pytorch"):
        """
        optimizer: to be called on model.parameters() and lr
        scheduler: to be called on optimizer(model)
        """
        if criterion is None:
            self.criterion = Loss(nn.MSELoss()) # mean over 1/B * 1/dim * 1/horizon
        else:
            self.criterion = criterion
        self.eval_losses = eval_losses

        self.mode = mode
        if self.mode=="pytorch":
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
        if not self.mode == "scikit-learn":
            self.model.to(self.device)
        self.do_train = do_train
        if self.mode == "pytorch" and self.do_train:
            self.reset_optimizer()

    def reset_model(self, weights):
        self.model.load_state_dict(weights)
    def reset_optimizer(self):
        self.current_optimizer = self.optimizer(self.model)
        if self.scheduler is not None:
            self.current_scheduler = self.scheduler(self.current_optimizer)
    def get_weights(self):
        if self.mode == "scikit-learn":
            return self.model.reg.coef_
        else:
            return self.model.state_dict()

    def compute_step(self, X_batch, context_batch, y_batch):
        """computes forward and backward on batch"""
        assert self.model is not None and self.do_train and self.mode == "pytorch"
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

    def fit(self, loader):
        assert self.mode == "scikit-learn"
        Xtrain, Ytrain = unroll_windows(loader)#, shuffle=True)
        self.model.fit(Xtrain.cpu(), Ytrain.cpu())

    def eval(self, loader, return_all=False, runs=1):
        """evaluates model on loader and returns mean loss
        return_all True: stores each step's loss (mean over batch)
        return_all False: overall mean loss
        """
        losses = {}

        #sklearn
        if self.mode == "scikit-learn":       
            Xtest, Ytest = unroll_windows(loader)
            predictions = self.model(Xtest)
            mean, std = get_normal_stats(Xtest)
            for loss_name, criterion in self.eval_losses.items():
                losses[loss_name] = criterion(predictions, Ytest, mean, std).cpu() # (steps, dim, horizon)
            if not return_all:
                for loss_name, criterion in self.eval_losses.items():
                    losses[loss_name] = losses[loss_name].mean().item() # scalar

        #pytorch
        else:
            counts = {}
            if self.mode == "pytorch":
                self.model.eval()
            with torch.inference_mode():
                for run in range(runs):
                    for X_batch, context_batch, y_batch in loader:
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                        if context_batch is not None:
                            context_batch = context_batch.to(self.device)
                        mean, std = get_normal_stats(X_batch)
                        
                        predictions = self.model(X_batch, context_batch)
                        
                        for loss_name, criterion in self.eval_losses.items():
                            loss = criterion(predictions, y_batch, mean, std).detach() # (bs * individuals, dim, horizon)
                            if return_all:
                                if loss_name not in losses:
                                    losses[loss_name] = []
                                losses[loss_name].append(loss.mean(dim=0).cpu()) # [ (dim, horizon) x steps]
                            else:
                                losses[loss_name] = losses.get(loss_name, 0) + loss.sum(dim=0).mean().item() #  scalar
                                counts[loss_name] = counts.get(loss_name, 0) + loss.shape[0]

            if return_all:
                for loss_name, criterion in self.eval_losses.items():
                    losses[loss_name] = torch.stack(losses[loss_name], dim=0) # (steps, dim, horizon)
            else:
                for loss_name, criterion in self.eval_losses.items():
                    losses[loss_name] = losses[loss_name] / counts[loss_name] # scalar
            
        return losses


def train_model(learner, loaders_dict, epochs=1, print_freq=50, eval_freq=10, verbose=1, do_eval=True, logger=None, eval_runs=1, weight_follow=None):
    """trains model in learner on loaders and returns train and valid losses"""
    
    #data
    train_loader = loaders_dict["train"]
    valid_loader1 = loaders_dict.get("valid1")
    valid_loader2 = loaders_dict.get("valid2")
    valid_loader3 = loaders_dict.get("valid3")
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
    valid_losses1 = {}
    valid_losses2 = {}
    valid_losses3 = {}
    weights = {}
    t1 = perf_counter()

    #training
    step = 0
    for epoch in range(epochs):
        #train_loader.dataset.add_epoch() #updates dataset epoch index for seeding
        for X_batch, context_batch, y_batch in train_loader:
            step += 1
            loss = learner.compute_step(X_batch, context_batch, y_batch)
            train_losses.append(loss) #loss of batch
            
            if do_eval and (step == 1 or step % eval_freq == 0 or step == total_steps):

                #valid eval
                if valid_loader1 is not None:
                    average_eval_dict1 = learner.eval(valid_loader1, runs=eval_runs)
                    append_in_dict(valid_losses1, average_eval_dict1)
                if valid_loader2 is not None:
                    average_eval_dict2 = learner.eval(valid_loader2, runs=eval_runs)
                    append_in_dict(valid_losses2, average_eval_dict2)
                if valid_loader3 is not None:
                    average_eval_dict3 = learner.eval(valid_loader3, runs=eval_runs)
                    append_in_dict(valid_losses3, average_eval_dict3)
                
                if weight_follow is not None:
                    append_in_dict(weights, weight_follow(learner.model))

                if verbose and (step == 1 or step % print_freq == 0 or step == total_steps):
                    if logger is not None:
                        logger.info(f"Step {step} | " + " | ".join([f"valid1 {loss_name} : {loss_value:.4f}" for loss_name, loss_value in average_eval_dict1.items()]))
                    else:
                        print(f"Step {step} | " + " | ".join([f"valid1 {loss_name} : {loss_value:.4f}" for loss_name, loss_value in average_eval_dict1.items()]))

    t2 = perf_counter()
    if verbose:
        if logger is not None:
            T = t2-t1
            logger.info(f"Training done in {T/60:.3f} min")
            logger.info(f"Average time per step: {T/total_steps:.3f} s")
        else:
            print(f"Training done in {(t2-t1)/60:.3f} min")
    return train_losses, valid_losses1, valid_losses2, valid_losses3, weights


def load_learner(model, normalization, criterion, lr, eval_losses, device):
    """loads correct model learner"""
    model_name = model.name
    if (model_name in ["persistence", "repeat", "lookback", "expected"]) and ((normalization is None) or (("mIN" not in normalization) and ("revin" not in normalization))):
        mode, do_train = "pytorch", False
    elif model_name == "sklinear":
        mode, do_train = "scikit-learn", True
    elif model_name == "chronos":
        mode, do_train = "pretrained", False
    else:
        mode, do_train = "pytorch", True
    learner = Learner(model, criterion, lr, eval_losses, device=device, mode=mode, do_train=do_train)
    return learner


def launch_training(model, normalization, criterion, lr, epochs, loaders_dict, eval_losses, device, save_dir, save_name, eval_freq, print_freq, logger):
    """launches training of model"""
    model_name = model.name
    criterion_name = criterion.name
    
    learner = load_learner(model, normalization, criterion, lr, eval_losses, device)

    #non trainable
    if (model_name in ["persistence", "repeat", "lookback", "expected"]) and ((normalization is None) or (("mIN" not in normalization) and ("revin" not in normalization))):
        logger.info("No training needed")
    
    #scikit learn .fit
    elif model_name == "sklinear":
        logger.info("Starting scikit-learn fitting...")
        learner.fit(loaders_dict["train"])
        logger.info("End of training")
    
    #pytorch training
    else:
        logger.info(f"Starting training pytorch with lr={lr}")
        if normalization is not None and (("revin" in normalization) or ("mIN" in normalization and "cmIN" not in normalization)):
            weight_follow = lambda model: {"beta": model.beta.data.detach().cpu().numpy()[0][0][0], "alpha": model.alpha.data.detach().cpu().numpy()[0][0][0]}
        else:
            weight_follow = None
        train_losses, valid_losses1, valid_losses2, valid_losses3, followed_weights = train_model(learner, loaders_dict, epochs=epochs, logger=logger, eval_runs=1, eval_freq=eval_freq, print_freq=print_freq, weight_follow=weight_follow)
        torch.save(learner.model.state_dict(), save_dir + "trained_model.pt")
        torch.save(train_losses, save_dir + f"train_losses.pt")
        torch.save(valid_losses1, save_dir + f"valid_losses1.pt")
        torch.save(valid_losses2, save_dir + f"valid_losses2.pt")
        torch.save(valid_losses3, save_dir + f"valid_losses3.pt")
        torch.save(followed_weights, save_dir + f"followed_weights.pt")
        
        #plots
        for loss_name in eval_losses:
            valid_dict = {"valid1": valid_losses1[loss_name]}
            if valid_losses2.get(loss_name) is not None:
                valid_dict["valid2"] = valid_losses2[loss_name]
            if valid_losses3.get(loss_name) is not None:
                valid_dict["valid3"] = valid_losses3[loss_name]
            if loss_name == criterion_name or (loss_name=="NMSE" and "NMSE" in criterion_name):
                plot_losses(train_losses, valid_dict, save_dir + "plots/", f"{loss_name}_plot.pdf", f"Training {loss_name} of {save_name}", eval_freq=eval_freq)
            else:
                plot_multi_losses(valid_dict,  save_dir + "plots/", f"{loss_name}_plot.pdf", f"Training {loss_name} of {save_name}", eval_freq=eval_freq)
        for weight_name in followed_weights:
            plot_serie(followed_weights[weight_name], save_dir + "plots/", f"{weight_name}.pdf", title=f"{weight_name} during training")
        logger.info("End of training")        
    
    #weights
    plot_weights(model, save_dir + "plots/", save_name)
    if (normalization is not None) and (("revin" in normalization) or ("mIN" in normalization and "cmIN" not in normalization)):
        params = {"beta": model.beta.data.detach().cpu().numpy()[0][0][0], "alpha": model.alpha.data.detach().cpu().numpy()[0][0][0]}
        logger.info(f"Final modulations: {params}")
    elif (normalization is not None and "cmIN" in normalization):
        params = {f"beta_{k}": value.data.detach().cpu().numpy()[0][0][0] for k,value in enumerate(model.betas)}
        logger.info(f"Final modulations: {params}")
    
    return learner


def launch_eval(learner, loaders_dict, stats_dict, eval_losses, save_dir, save_name, complete_evaluation, save=False, results_dir=None, mode="Test", denormalize=False, runs=1):
    """evaluating model script"""
    if results_dir is None:
        results_dir = save_dir
    
    losses1, losses2, losses3 = None, None, None
    if mode == "Valid":
        sub_ = "valid"
        losses1 = learner.eval(loaders_dict["valid1"], return_all=True, runs=runs) #(steps, dim, horizon)
        if save:
            torch.save(losses1, save_dir + "valid_losses1.pt")
        if loaders_dict.get("valid2") is not None:
            losses2 = learner.eval(loaders_dict["valid2"], return_all=True, runs=runs)
            if save:
                torch.save(losses2, save_dir + "valid_losses2.pt")
        if loaders_dict.get("valid3") is not None:
            losses3 = learner.eval(loaders_dict["valid3"], return_all=True, runs=runs)
            if save:
                torch.save(losses3, save_dir + "valid_losses3.pt")
    elif mode == "Test":
        sub_ = "test"
        losses1 = learner.eval(loaders_dict["test1"], return_all=True, runs=runs)
        if save:
            torch.save(losses1, save_dir + "test_losses1.pt")
        if loaders_dict.get("test2") is not None:
            losses2 = learner.eval(loaders_dict["test2"], return_all=True, runs=runs) 
            if save:
                torch.save(losses2, save_dir + "test_losses2.pt")
    else:
        raise ValueError("Unrecognized eval mode")
  
    for loss_name in eval_losses:
        if losses1 is not None:
            mean = losses1[loss_name].mean()
            if denormalize:
                mean *= stats_dict["train"]["std"]**2
            save_results(mean, results_dir, f"{sub_}1_mean_results.json", save_name, f"{mode} {loss_name}")
            if complete_evaluation:
                std = losses1[loss_name].std()
                save_results(std, save_dir, f"{sub_}1_std_results.json", save_name, f"{mode} {loss_name}")
                plot_errors(losses1[loss_name].sum(axis=1).mean(axis=1), save_dir + "plots/", f"{sub_}1_{loss_name}.pdf", f"{mode} 1 {loss_name} of {save_name} : {mean}")
                plot_horizon_errors(losses1[loss_name].sum(axis=1).mean(axis=0), save_dir + "plots/", f"test1_horizon_{loss_name}.pdf", f"{mode} {loss_name} of {save_name} : {mean}")
        if losses2 is not None:
            mean = losses2[loss_name].mean()
            if denormalize:
                mean *= stats_dict["train"]["std"]**2
            save_results(mean, results_dir, f"{sub_}2_mean_results.json", save_name, f"{mode} {loss_name}")
            if complete_evaluation:
                std = losses2[loss_name].std()
                save_results(std, save_dir, f"{sub_}2_std_results.json", save_name, f"{mode} {loss_name}")
                plot_errors(losses2[loss_name].sum(axis=1).mean(axis=1), save_dir + "plots/", f"{sub_}2_{loss_name}.pdf", f"{mode} 1 {loss_name} of {save_name} : {mean}")
                plot_horizon_errors(losses2[loss_name].sum(axis=1).mean(axis=0), save_dir + "plots/", f"{sub_}2_horizon_{loss_name}.pdf", f"{mode} {loss_name} of {save_name} : {mean}")
        if losses3 is not None:
            mean = losses3[loss_name].mean() 
            if denormalize:
                mean *= stats_dict["train"]["std"]**2
            save_results(mean, results_dir, f"{sub_}3_mean_results.json", save_name, f"{mode} {loss_name}")
            if complete_evaluation:
                std = losses3[loss_name].std()
                save_results(std, save_dir, f"{sub_}3_std_results.json", save_name, f"{mode} {loss_name}")
                plot_errors(losses3[loss_name].sum(axis=1).mean(axis=1), save_dir + "plots/", f"{sub_}3_{loss_name}.pdf", f"{mode} 1 {loss_name} of {save_name} : {mean}")
                plot_horizon_errors(losses3[loss_name].sum(axis=1).mean(axis=0), save_dir + "plots/", f"{sub_}3_horizon_{loss_name}.pdf", f"{mode} 1 {loss_name} of {save_name} : {mean}")


def launch_example(data_path, model, lags, horizon, device, save_dir, save_name):
    """runs model on example"""
    if "crev" not in model.norm_name and "cm" not in model.norm_name and "softm" not in model.norm_name:#TODO gerer ce cas
        ex_dir = data_path + "examples/" + f"{lags}_{horizon}/"
        if not os.path.exists(ex_dir):
            set_random_data(data_path, lags, horizon, name="rand")
            plot_named_example(ex_dir, f"rand")
        dico = fetch_example_data(ex_dir)
        for data_name, data_tuple in dico.items():
            x, c, y = data_tuple[0].unsqueeze(0).to(device), data_tuple[1], data_tuple[2].unsqueeze(0).to(device)
            if c is not None:
                c = c.unsqueeze(0).to(device)
            pred = model(x,c)
            plot_pred(x[0,0].cpu().detach().tolist(), y[0,0].cpu().detach().tolist(), pred[0,0].cpu().detach().tolist(), save_dir + "examples/", f"{data_name}_predictions.pdf", f"Example {data_name} prediction for {save_name}")
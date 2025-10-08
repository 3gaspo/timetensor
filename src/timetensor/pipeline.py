import torch
import torch.nn as nn
import torch.optim as optim
from time import perf_counter
import os

from .utils import get_normal_stats, average_loss, append_in_dict, unroll_windows, normalize, save_results
from .visu import plot_losses, plot_multi_losses, plot_serie, plot_named_example, plot_horizon_errors, plot_pred, plot_errors, plot_horizon_errors
from .dataset import set_random_data, fetch_example_data

class Loss():
    def __init__(self, loss, mean=None, std=None, mode=None, eps=1e-8):
        self.loss = loss #e.g nn.MSELoss()
        self.mode = mode

        self.mean = mean
        self.std = std
        self.standard_norm = (mean is not None and std is not None)
        self.eps = eps
        self.name = None

    def __call__(self, pred, y, mean=None, std=None):
        if self.standard_norm:
            pred = normalize(pred, self.mean, self.std, self.eps)
            y = normalize(y, self.mean, self.std, self.eps)
        if self.mode == "instance":
            assert (mean is not None and std is not None)
            pred = normalize(pred, mean, std, self.eps)
            y = normalize(y, mean, std, self.eps)
        elif self.mode == "relative":
            assert (mean is not None and std is not None)
            #mean = torch.where(mean != 0, mean, self.eps)
            mean = torch.abs(mean) + self.eps
            pred, y = pred/mean, y/mean
        elif self.mode == "normalize_y":
            assert (mean is not None and std is not None)
            y = normalize(y, mean, std, self.eps)
        elif self.mode == "denormalize_pred":
            assert (mean is not None and std is not None)
            pred = pred*(std+self.eps) + mean
        return self.loss(pred, y)


def get_losses(criterion_name, mean=None, std=None, complete_evaluation=False):
    """returns criterion and relevant eval losses from specified criterion name"""
    if criterion_name == "MSE":
        criterion = Loss(nn.MSELoss())
    elif criterion_name == "MMSE":
        criterion = Loss(nn.MSELoss(), mean, std)
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
        Xtrain, Ytrain = unroll_windows(loader)#, shuffle=True)
        self.model.fit(Xtrain.cpu(), Ytrain.cpu())

    def eval(self, loader, return_all=False, runs=1):
        """evaluates model on loader and returns mean loss
        return_all True: stores each step's loss (mean over batch)
        return_all False: overall mean loss
        """
        losses = {}
        #pytorch
        if self.pytorch:
            counts = {}
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
        
        #sklearn
        else:       
            Xtest, Ytest = unroll_windows(loader)
            predictions = self.model(Xtest)
            mean, std = get_normal_stats(Xtest)
            for loss_name, criterion in self.eval_losses.items():
                losses[loss_name] = criterion(predictions, Ytest, mean, std).cpu() # (steps, dim, horizon)
            if not return_all:
                for loss_name, criterion in self.eval_losses.items():
                    losses[loss_name] = losses[loss_name].mean().item() # scalar
    
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
    valid_losses3= {}
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
                if valid_loader2 is not None:
                    average_eval_dict2 = learner.eval(valid_loader2, runs=eval_runs)
                if valid_loader3 is not None:
                    average_eval_dict3 = learner.eval(valid_loader3, runs=eval_runs)
                append_in_dict(valid_losses1, average_eval_dict1)
                append_in_dict(valid_losses2, average_eval_dict2)
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


def launch_training(model, normalization, criterion, lr, epochs, loaders_dict, eval_losses, device, save_dir, save_name, eval_freq, print_freq, logger, retrain=True):
    """launches training of model"""
    model_name = model.name
    criterion_name = criterion.name
    #non trainable
    if (model_name in ["persistence", "repeat", "lookback", "expected"]) and ((normalization is None) or (("mIN" not in normalization) and ("revin" not in normalization))):
        learner = Learner(model, criterion, lr, eval_losses, device=device, do_train=False)
        logger.info("No training needed")
    
    #scikit learn .fit
    elif model_name == "sklinear":
        learner = Learner(model, criterion, lr, eval_losses, device=device, pytorch=False)
        if retrain:
            logger.info("Starting scikit-learn fitting...")
            learner.fit(loaders_dict["train"])
            logger.info("End of training")
        else:
            logger.info("No training needed")
    
    #pytorch training
    else:
        learner = Learner(model, criterion, lr, eval_losses, device=device)
        if retrain:
            logger.info(f"Starting training pytorch with lr={lr}")
            if normalization is not None and (("revin" in normalization) or ("mIN" in normalization)):
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
                valid_dict = {"valid1": valid_losses1[loss_name], "valid2": valid_losses2[loss_name], "valid3": valid_losses3[loss_name]}
                if loss_name == criterion_name or (loss_name=="NMSE" and "NMSE" in criterion_name):
                    plot_losses(train_losses, valid_dict, save_dir + "plots/", f"{loss_name}_plot.pdf", f"Training {loss_name} of {save_name}", eval_freq=eval_freq)
                else:
                    plot_multi_losses(valid_dict,  save_dir + "plots/", f"{loss_name}_plot.pdf", f"Training {loss_name} of {save_name}", eval_freq=eval_freq)
            for weight_name in followed_weights:
                plot_serie(followed_weights[weight_name], save_dir + "plots/", f"{weight_name}.pdf", title=f"{weight_name} during training")
            logger.info("End of training")        
        else:
            logger.info("No training needed")
    return learner


def launch_eval(learner, loaders_dict, eval_losses, save_dir, save_name, complete_evaluation, save=False, results_dir=None):
    """evaluating model script"""
    if results_dir is None:
        results_dir = save_dir
    test_losses1 = learner.eval(loaders_dict["test1"], return_all=True) #(steps, dim, horizon)
    test_losses2 = learner.eval(loaders_dict["test2"], return_all=True) #(steps, dim, horizon)
    if save:
        torch.save(test_losses1, save_dir + "test_losses1.pt")
        torch.save(test_losses2, save_dir + "test_losses2.pt")

    for loss_name in eval_losses:
        mean = test_losses1[loss_name].mean() 
        #std = test_losses1[loss_name].std()
        save_results(mean, results_dir, "test1_mean_results.json", save_name, f"Test {loss_name}")
        # save_results(std, save_dir, "test1_std_results.json", save_name, f"Test {loss_name}")
        if complete_evaluation:
            # plot_errors(test_losses1[loss_name].sum(axis=1).mean(axis=1), save_dir + "plots/", f"test1_{loss_name}.pdf", f"Test 1 {loss_name} of {save_name} : {mean}")
            plot_horizon_errors(test_losses1[loss_name].sum(axis=1).mean(axis=0), save_dir + "plots/", f"test1_horizon_{loss_name}.pdf", f"Test 1 {loss_name} of {save_name} : {mean}")
    for loss_name in eval_losses:
        mean = test_losses2[loss_name].mean()
        # std = test_losses2[loss_name].std()
        save_results(mean, results_dir, "test2_mean_results.json", save_name, f"Test {loss_name}")
        # save_results(std, save_dir, "test2_std_results.json", save_name, f"Test {loss_name}")
        if complete_evaluation:
            # plot_errors(test_losses2[loss_name].sum(axis=1).mean(axis=1), save_dir + "plots/", f"test2_{loss_name}.pdf", f"Test 2 {loss_name} of {save_name} : {mean}")
            plot_horizon_errors(test_losses2[loss_name].sum(axis=1).mean(axis=0), save_dir + "plots/", f"test2_horizon_{loss_name}.pdf", f"Test 2 {loss_name} of {save_name} : {mean}")

def launch_example(data_path, model, lags, horizon, device, save_dir, save_name):
    """runs model on example"""
    if model.norm_name not in ["crevin", "cmIN", "cflexrevin"]:#TODO gerer ce cas
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
            plot_pred(x[0,0].cpu().detach().numpy(), y[0,0].cpu().detach().numpy(), pred[0,0].cpu().detach().numpy(), save_dir + "examples/", f"{data_name}_predictions.pdf", f"Example {data_name} prediction for {save_name}")
import itertools
import numpy as np
import copy
from scipy.special import binom
import torch

def get_coalitions(L):
    """returns all possible subsets of L"""
    coalitions = []
    for k in range(len(L)+1):
        for coalition in itertools.combinations(L, k):
            coalitions.append(list(coalition))
    return coalitions


def get_coalition_weight(coalition, team, players):
    """returns shap importance weight"""
    remaining = players-len(team)
    coeff = int(binom(remaining,len(coalition)))
    weight = 1/((remaining+1)*coeff)
    return weight


class BackgroundDataset():
    """dataset which returns a random value when called"""
    def __init__(self, dataset, seed=None):
        self.dataset = dataset
        self.size = len(dataset)
        self.seed = seed

    def __len__(self):
        return len(self.dataset)
    
    def get_data(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        idx = np.random.randint(self.size)
        output = self.dataset[idx]
        return output[0]
    
    def get_batch(self, size):
        batch = []
        for k in range(size):
            batch.append(self.get_data())
        return batch

    def get_conditional_data(self, x, S):
        """return a random value conditioned by x_S"""
        pass
    def get_conditional_batch(self, x, S, size):
        batch = []
        for k in range(size):
            batch.append(self.get_conditional_data(x, S))
        return batch
        


class Game:
    """game of players=features"""
    def __init__(self, model, players, background):
        """players is dict {player_name, idx}"""
        self.player_names = players.keys() #name of features
        self.players = len(players)
        self.players_idx = list(players.values())
        self.players_ids = list(range(self.players))
        self.background = background
        self.model = model
    
    def get_excluding_coalitions(self, team):
        """returns list of coalitions without team [ids]"""
        included = [player for player in self.players_ids if player not in team]
        coalitions = get_coalitions(included) #very slow
        return coalitions
    
    def sample_excluding_coalitions(self, team, size, replace=True):
        """returns list of sampled coalitions without team [idxs]"""
        #coalitions = self.get_excluding_coalitions(team) #very slow
        #sampled = np.random.choice(coalitions, size=size, replace=replace)
        included = [player for player in self.players_ids if player not in team]
        sampled = [list(np.sort(np.random.choice(included, np.random.randint(0,len(included)+1), replace=replace))) for _ in range(size)] #+ [[],included.copy()]
        return sampled

    def replace(self, x, team, coalition, size, split=False):
        """returns x_S+team and x_S for coalition S, with different background replacements (split or not)"""
        remaining = np.array([player for player in self.players_ids if player not in team and player not in coalition])
        team_ids = np.array(team)
        backgroud_values = self.background.get_batch(size)
        if split:
            second_backgroud_values = self.background.get_batch(size)
        replaced_values_team = []
        replaced_values_no_team = []
        for k in range(size):
            replaced_team = copy.deepcopy(x)
            for player in remaining:
                replaced_team[self.players_idx[player]] = backgroud_values[k][self.players_idx[player]]
            replaced_values_team.append(replaced_team)

            replaced_no_team = copy.deepcopy(replaced_team)
            for player in team_ids:
                if split:
                    replaced_no_team[self.players_idx[player]] = second_backgroud_values[k][self.players_idx[player]]
                else:
                    replaced_no_team[self.players_idx[player]] = backgroud_values[k][self.players_idx[player]]
            replaced_values_no_team.append(replaced_no_team)
        return torch.concat(replaced_values_team, dim=0), torch.concat(replaced_values_no_team, dim=0)
    
    def sample_replacements(self, x, team, ncoalitions, nexamples, replace=True, aggregate=False, split=False, logger=None):
        """samples coalitions and applie replacement"""
        if aggregate:
            sampled_coalitions = [[], [player for player in self.players_idx if player not in team]]
        else:
            sampled_coalitions = self.sample_excluding_coalitions(team, ncoalitions, replace)
        replaced_values_team, replaced_values_no_team = {tuple(coalition): None for coalition in sampled_coalitions}, {tuple(coalition): None for coalition in sampled_coalitions}
        for coalition in sampled_coalitions:
            #logger.info(f"Replacing coalition {coalition}")
            replaced_team, replaced_no_team = self.replace(x, team, coalition, nexamples, split)
            replaced_values_team[tuple(coalition)] = replaced_team
            replaced_values_no_team[tuple(coalition)] = replaced_no_team
        return sampled_coalitions, replaced_values_team, replaced_values_no_team
    
    def sample_predictions(self, x, idx, team, ncoalitions, nexamples, replace=True, aggregate=False, split=False, logger=None):
        sampled_coalitions, replaced_values_team, replaced_values_no_team = self.sample_replacements(x, team, ncoalitions, nexamples, replace, aggregate, split, logger=logger)
        deltas = {tuple(coalition): None for coalition in sampled_coalitions}
        print("sampled coalitions: ", len(sampled_coalitions))
        for coalition in sampled_coalitions:
            #logger.info(f"Predicting coalition {coalition}")
            x_team, x_no_team = replaced_values_team[tuple(coalition)], replaced_values_no_team[tuple(coalition)]
            predictions_team = self.model(x_team) # (nexamples, dim, horizon)
            predictions_no_team = self.model(x_no_team) # (nexamples, dim, horizon) #idx (1, 1)
            value = torch.mean(predictions_team - predictions_no_team, dim=0)[:, idx] #(1)
            deltas[tuple(coalition)] = value
            print(value.shape)
        return sampled_coalitions, deltas

    def get_shapley_value(self, x, idx, team, ncoalitions, nexamples, replace=True, aggregate=False, split=False, return_coalitions=False, logger=None):
        sampled_coalitions, deltas = self.sample_predictions(x, idx, team, ncoalitions, nexamples, replace, aggregate, split, logger=logger)
        if return_coalitions:
            return np.mean(deltas.values()), sampled_coalitions
        return np.mean(list(deltas.values()))
    
    def get_shapley_values(self, x, idx, ncoalitions, nexamples, replace=True, aggregate=False, split=False, logger=None):
        shapley_values = {player: None for player in self.player_names}
        for k in range(self.players):
            logger.info(f"Computing shapley of {k}")
            team = [k]
            shapley_values[self.player_names[k]] = self.get_shapley_value(x, idx, team, ncoalitions, nexamples, replace, aggregate, split, logger=logger)
        return shapley_values
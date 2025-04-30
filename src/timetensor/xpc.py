from torch.utils.data import Dataset
import itertools
import numpy as np

 def get_coalitions(L):
     """returns all possible subsets of L"""
    coalitions = []
    for k in range(len(L)+1):
        for coalition in itertools.combinations(L, k):
            coalitions.append(list(coalition))
    return coalitions

class BackgroundDataset(Dataset):
    """dataset which returns a random value when called"""
    def __init__(self, dataset, seed=None):
        self.dataset = dataset
        self.size = len(dataset)
        self.seed = seed

    def get_data(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        idx = np.random.randint(self.size)
        return self.dataset(idx)
    def get_batch(self, size):
        batch = []
        for k in range(size):
            batch.append(self.get_data())
        return batch

    def get_conditional_data(self, x, S):
        """return a random value conditioned by x_S"""
        pass
    def get_conditional_batch(self, x, S):
        batch = []
        for k in range(size):
            batch.append(self.get_conditional_data(x, S))
        return batch
        


class Game:
    """game of players=features"""
    def __init__(self, player_names):
        self.player_names = player_names
        self.players = len(player_names)
        self.player_idx = list(range(self.players))
    
    def get_excluding_coalitions(self, team):
        """returns list of coalitions without team [idxs]"""
        included = [player for player in self.player_idx if player not in team]
        return get_coalitions(included)
    def sample_excluding_coalitions(self, team, size, replace=True):
        included = [player for player in self.player_idx if player not in team]
        coalitions = [list(np.sort(np.random.choice(subset, np.random.randint(0,len(subset)+1),replace=replace))) for k in range(size)]
        return coalitions


def shapley_(x, background, model, players, j, n1, n2):

    coalitions = sample_coalitions(j, players, n1) # (n1, J-|j|)
    batch = replace(x, coalitions, background, n2) # (n1, n2, *dim(x))
    predictions = model(batch.view(-1, x.shape))  # (n1 * n2, *dim(x))
    shapley_value = compute_shapley(predictions)



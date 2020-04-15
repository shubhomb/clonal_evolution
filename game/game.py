import numpy as np



class Stage_Game():
    def __init__(self, n):
        self.num_players = n
        self.m = np.zeros((n,n))
        # Note: self.m[i, j] is the payoff of i by encountering j. Thus, j's payoff is self.m[j, i]

    def add_profile(self, player, payoffs):
        self.matx[player] = payoffs

    def strictly_dominated_strategies(self):
        # a Nash equilibrium, even mixed, cannot contain any strategies that are strictly dominated
        for i in range(self.num_players):
            for j in range(self.num_players):
                if i != j:
                    if np.prod(np.less(self.matx[:,i], self.matx[:,j])) == 1:
                        # i is dominated by j

if __name__ == "__main__":

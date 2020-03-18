import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Clone():
    def __init__(self, parent, phenotype):
        self.parent = parent
        self.phenotype = phenotype

class GompertzTumor():
    def __init__(self, carrying_capacity, initial_growth_rate, num_progenitors=1):
        '''

        :param carrying_capacity: the total number of cells that can be resourced in the microenvironment (int)
        :param initial_growth_rate:
        :param num_progenitors: how many cells are afflicted by first carcinogen (int, default: 1)
        '''
        self.k = carrying_capacity
        self.b = initial_growth_rate
        self.n_0 = num_progenitors

    def population(self, t):
        '''

        :param t: time interval (int)
        :return: Gompertz population function given tumor parameters and t
        '''
        return self.n_0 * np.exp(np.log(self.k / self.n_0) * (1 - np.exp(-self.b * t)))

    def visualize_gompertz_growth(self, t):
        '''
        A helper function for seeing effect of parameter choice
        :param t: time interval (int)
        :return: None
        '''
        pops = np.apply_along_axis(self.population, axis=0, arr=np.linspace(0, t, t))
        plt.title("Gompertzian Population Growth\nk=%d b=%f N_0=%d" % (k, b, n_0))
        plt.xlabel("Time interval")
        plt.ylabel("Number of cells")
        plt.plot(np.linspace(0, t, t), pops)
        plt.grid()
        plt.show()

if __name__ == "__main__":
    k = 1000
    b = 0.3
    n_0 = 1
    tumor = GompertzTumor(k, initial_growth_rate=b, num_progenitors=n_0)
    tumor.visualize_gompertz_growth(100)

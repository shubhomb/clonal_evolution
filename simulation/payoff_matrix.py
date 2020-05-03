import pandas as pd
import numpy as np


class PayoffMatrix():
    def __init__(self, sim):
        self.sim = sim
        self.matrix = np.zeros(shape=(len(self.sim.subclones), len(self.sim.subclones)))
        self.populate_matrix(self.sim.t)

    def populate_matrix(self, t):
        treatments = self.sim.treatments[t, :]
        for i in range(len(self.sim.subclones)):
            for j in range(len(self.sim.subclones)):
                fj = np.dot(self.sim.subclones[j].alpha, treatments)
                fi = np.dot(self.sim.subclones[i].alpha, treatments)
                # print (self.sim.subclones[j].alpha)
                # print (treatments)
                self.matrix[i, j] = fj - fi

    def print_matrix(self):
        labs = [s.label for s in self.sim.subclones]
        self.pretty_matrix = pd.DataFrame(self.matrix, index=labs, columns=labs)
        print (self.pretty_matrix)

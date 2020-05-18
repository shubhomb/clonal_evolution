import numpy as np

class Subclone:
    """
        Initializes a Subclone Population.
        :attr label:    Either A, B or S
        :attr fitness:  Current fitness
        :attr prop:     Current Proportion
    """

    def __init__(self, lbl, c, alpha, prop=0.333, parent=None, birthtime=None):
        self.label = lbl
        self.fitness = 0.0
        self.prop = prop
        self.c = c
        self.parent = parent
        self.alpha = alpha
        self.bt = birthtime

    def __str__(self):
        return self.label

    def update_fitness(self, treatment):
        """
        Returns the fitness with the given environment for subclone [type]
        @ param treatment: 1d np.ndarray of shape (num_treatments) for intensity of treatment
        """
        self.fitness = 1 - self.c - np.dot(self.alpha, treatment)
        return self.fitness


    def log(self):
        print("Node: ", self.label)
        print("Birthtime: ", self.bt)
        print(f'\t \t Alpha: {self.alpha}')
        print(f'\t \t Prop: {self.prop}')
        print(f'\t \t Resistant: {self.c}')
        print(f'\t \t Fitness: {self.fitness}')

import numpy as np

class Doctor():

    def __init__(self, simulation, schedule, distro, decay=0):
        self.simulation = simulation
        self.schedule = np.argwhere(schedule > 0)
        if decay < 0 or decay > 1:
            raise ValueError("Decay should be positive in [0,1]")
        self.decay = decay
        self.distro = distro
        self.num_strats = 4
        self.num_drugs = simulation.treatments.shape[1]

    def change_treatment(self, t, treatment):
        if t not in self.schedule:
            raise ValueError("Doctor can only change at given times")
        other_times = self.schedule[np.where(self.schedule > t)]
        if other_times.shape[0] > 0:
            next_time = np.min(other_times)
        else:
            next_time = self.simulation.num_timesteps
        self.simulation.treatments[t: next_time, :] = treatment
        if self.decay > 0:
            for j in range(1,next_time - t):
                self.simulation.treatments[t + j :] *= max(0, self.decay ** j)

    def greedy_fittest(self, magnitude=1.0):
        """
        Choose candidate with greatest fitness.
        :param magnitude:
        :return:
        """
        fittest_subclone = self.simulation.subclones[np.argmax([f.fitness for f in self.simulation.subclones])]
        sus_drug = np.argmax(fittest_subclone.alpha)
        treatment = np.zeros(self.num_drugs)
        treatment[sus_drug] = magnitude
        self.change_treatment(self.simulation.t, treatment)


    def greedy_prop(self, magnitude=1.0):
        """
        Choose candidate with greatest proportion.
        :param magnitude:
        :return:
        """
        populous_subclone = self.simulation.subclones[np.argmax([f.prop for f in self.simulation.subclones])]
        sus_drug = np.argmax(populous_subclone.alpha)
        treatment = np.zeros(self.num_drugs)
        treatment[sus_drug] = magnitude
        self.change_treatment(self.simulation.t, treatment)

    def greedy_propweighted_fitness(self, magnitude=1.0):
        """
        Choose candidate with greatest product of fitness and proportion.
        :param magnitude:
        :return:
        """
        weighted_subclone = self.simulation.subclones[np.argmax([f.prop * f.fitness for f in self.simulation.subclones])]
        sus_drug = np.argmax(weighted_subclone.alpha)
        treatment = np.zeros(self.num_drugs)
        treatment[sus_drug] = magnitude
        self.change_treatment(self.simulation.t, treatment)


    def greedy_degree(self, magnitude=1.0):
        """
        Choose candidate with highest degree in graph.
        :param magnitude:
        :return:
        """
        degs = [self.simulation.graph.nxgraph.degree(sc, weight="weight") for sc in self.simulation.subclones]
        print (degs)
        affected_subclone = self.simulation.subclones[np.argmax(degs)]
        sus_drug = np.argmax(affected_subclone.alpha)
        treatment = np.zeros(self.num_drugs)
        treatment[sus_drug] = magnitude
        self.change_treatment(self.simulation.t, treatment)


    def mixed_strategy(self):
        """
        This function is an example about how to simulate a doctor strategy with randomness. The deterministic case
        corresponds to choosing the strategy defined in doctor.py.
        :param distro:
        :return:
        """
        if self.distro is None:
            raise ValueError("Doctor must have a distribution specified for mixing")
        strats = {"propweight": lambda doc: doc.greedy_propweighted_fitness(magnitude=1.0),
                  "prop": lambda doc: doc.greedy_prop(magnitude=1.0),
                  "fit": lambda doc: doc.greedy_fittest(magnitude=1.0),
                  "degree": lambda doc: doc.greedy_degree(magnitude=1.0)
                  }
        strat = np.random.choice(list(strats.keys()), 1, p=self.distro)[0]

        return strat, strats[strat]

import numpy as np

class Doctor():

    def __init__(self, simulation, schedule, decay=0, distro=None):
        self.simulation = simulation
        self.schedule = np.argwhere(schedule > 0)
        if decay < 0 or decay > 1:
            raise ValueError("Decay should be positive in [0,1]")
        self.decay = decay
        self.num_drugs = simulation.treatments.shape[1]
        if distro:
            if np.sum(self.distro) > 1:
                raise ValueError("Distribution sum should add to 1")
            self.distro = distro

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

    def choose_node(self, time):
        """
        Target maximum node and all nodes within some constant

        Returns target node
        """
        DEPTH = 0.1
        # target_node = nx.maximal_independent_set(sim.graph.nxgraph)[0]
        # all_degrees = list(self.graph.nxgraph.degree(self.graph.all_nodes, weight='weight'))
        # target_node = min(all_degrees, key=lambda item:item[1])[0]
        if time < 10:
            target_node = list(self.graph.nxgraph.nodes)[0]
        else:
            target_node = list(self.graph.nxgraph.nodes)[2]

        return target_node
    def greedy_fittest(self, magnitude=1.0):
        fittest_subclone = self.simulation.subclones[np.argmax([f.fitness for f in self.simulation.subclones])]
        sus_drug = np.argmax(fittest_subclone.alpha)
        treatment = np.zeros(self.num_drugs)
        treatment[sus_drug] = magnitude
        self.change_treatment(self.simulation.t, treatment)
        return "fit"


    def greedy_prop(self, magnitude=1.0):
        populous_subclone = self.simulation.subclones[np.argmax([f.prop for f in self.simulation.subclones])]
        sus_drug = np.argmax(populous_subclone.alpha)
        treatment = np.zeros(self.num_drugs)
        treatment[sus_drug] = magnitude
        self.change_treatment(self.simulation.t, treatment)
        return "prop"

    def greedy_propweighted_fitness(self, magnitude=1.0):
        weighted_subclone = self.simulation.subclones[np.argmax([f.prop * f.fitness for f in self.simulation.subclones])]
        sus_drug = np.argmax(weighted_subclone.alpha)
        treatment = np.zeros(self.num_drugs)
        treatment[sus_drug] = magnitude
        self.change_treatment(self.simulation.t, treatment)
        return "propweight"


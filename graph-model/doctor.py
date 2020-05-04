"""
    Contains various Doctor strategies 
"""
def change_treatment(self, t, treatment):
    self.simulation.treatments[t, :] = treatment

def greedy_proportion(self, magnitude=1.0):
    fittest_subclone = self.simulation.subclones[np.argmax([f.fitness for f in self.simulation.subclones])]
    sus_drug = np.argmax(fittest_subclone.alpha)
    treatment = np.zeros(self.num_drugs)
    treatment[sus_drug] = magnitude
    self.change_treatment(self.simulation.t + 1, treatment)
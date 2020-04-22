import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json


class Subclone:
    """
        Initializes a Subclone Population.
        :attr label:    Either A, B or S
        :attr fitness:  Current fitness
        :attr prop:     Current Proportion
    """

    def __init__(self, lbl, c, alpha, prop=0.333):
        self.label = lbl
        self.fitness = 1
        self.prop = prop
        self.c = c
        self.alpha = alpha
    def __str__(self):
        return self.label

    def update_fitness(self, treatment):
        """
        Returns the fitness with the given environment for subclone [type]
        @ param treatment: 1d np.ndarray of shape (num_treatments) for intensity of treatment
        """
        self.fitness = 1 - self.c - np.dot(self.alpha, treatment)
        return self.fitness

class Simulation:
    """
        Simulation Class contains relevant methods to simulate the evolution
        of the subclone colonies.  In this example, it is hardcoded to three
        colonies.
    """
    def __init__(self, subclones, treatments):
        """
            :attr time: Time stamp starting from t=0
            :attr subclones (list): list of Subclone objects
            :attr cs(np.ndarray): the cost of of resistance for each subclone i
            :attr alphas (np.ndarray): 2d array (num subclones x num treatments) of the added susceptibility to each drug subclone i
            :attr treatments (np.ndarray): two dimensional array total timesteps x num_treatments
        """
        self.t = 0
        self.subclones = subclones
        self.treatments = treatments
        self.num_timesteps = self.treatments.shape[0]


    def adjust_proportion(self):
        """
            Adjusts using p_t+1 = p_t * W(i)/W
            This assumes fitness are already updated / calculuated
        """
        avg_fit = self.calc_avg_fitness()
        for c in self.subclones:
            c.prop *= c.fitness / avg_fit


    def calc_avg_fitness(self):
        """
        Given fitness environment, returns average fitness.
        """
        return sum([c.prop * c.fitness for c in self.subclones])

    def run_step(self):
        for c in self.subclones:
            c.update_fitness(self.treatments[self.t, :])

        if self.t < self.num_timesteps:
            self.t += 1
        else:
            print ("Simulation complete after %d timesteps" %self.num_timesteps)
        return  self.calc_avg_fitness()


def run_sim(max_time, num_treatments, treatments, subclones, treatment_names):
    if not treatments.shape[0] == max_time and treatments.shape[1] == num_treatments:
        raise ValueError("Unexpected treatment shape, should be (timesteps, number of treatments")
    if len(treatment_names) != treatments.shape[1]:
        raise ValueError("Treatment names and number of treatments should be same")


    model = Simulation(subclones, treatments)


    log_fitness = np.zeros(shape=(MAX_TIME, len(subclones)))
    log_props = np.zeros(shape=(MAX_TIME, len(subclones)))
    log_avg_fitness = np.zeros(shape=(MAX_TIME))

    for t in range(MAX_TIME):
        avg = model.run_step()
        # Adjust Proportion
        model.adjust_proportion()
        log_fitness[t, :] = np.array([c.fitness for c in model.subclones])
        log_props[t, :] = np.array([c.prop for c in model.subclones])
        log_avg_fitness[t] = avg

    savedir = os.path.join(os.path.split(os.getcwd())[0], "data", "simulations")
    dirname = str(datetime.today().date()).replace("-", "_") + "_0"
    i = 1
    while os.path.exists(os.path.join(savedir, dirname)):
        dirname = dirname[:-2] + "_%d"%i
        i += 1
    full_dir = os.path.join(savedir, dirname)
    os.mkdir(full_dir)
    cs = np.array([sc.c for sc in subclones])
    alphas = np.zeros(shape=(len(subclones), num_treatments))
    for i in range(len(subclones)):
        alphas[i, :] = subclones[i].alpha

    if not os.path.exists(os.path.join(full_dir, "params")):
        os.mkdir(os.path.join(full_dir, "params"))
    np.save(os.path.join(full_dir, "params", "alphas.npy"), alphas)
    np.save(os.path.join(full_dir, "params", "cs.npy"), cs)
    np.save(os.path.join(full_dir, "params", "treatments.npy"), treatments)



    # Plot Resulting Curves


    plt.grid()
    x_axis = [k for k in range(MAX_TIME)]
    plt.xlabel("t")
    plt.ylabel("proportion of tumor population")
    for i in range(len(subclones)):
        plt.plot(x_axis, log_props[:, i].flatten(), label=subclones[i].label)
    plt.legend()
    title = "Proportion of tumor population over time"
    plt.title(title)
    plt.savefig(os.path.join(full_dir, "proportions.png"))
    plt.show()

    plt.grid()
    x_axis = [k for k in range(MAX_TIME)]
    plt.xlabel("t")
    plt.ylabel("fitness of tumor population")
    for i in range(len(subclones)):
        plt.plot(x_axis, log_fitness[:, i].flatten(), label=subclones[i].label)
    plt.plot (x_axis, log_avg_fitness, label="average")
    plt.legend()
    title = "Subclonal fitness over time"
    plt.title(title)
    plt.savefig(os.path.join(full_dir, "fitness.png"))
    plt.show()



    plt.grid()
    x_axis = [k for k in range(MAX_TIME)]
    plt.xlabel("t")
    plt.ylabel("dosage (arbitrary unit)")
    for i in range(treatments.shape[1]):
        plt.plot(x_axis, treatments[:, i], label=treatment_names[i])
    plt.legend()
    title = "Treatments over time"
    plt.title(title)
    plt.savefig(os.path.join(full_dir, "treatments.png"))
    plt.show()











if __name__ == "__main__":
    MAX_TIME = 100
    num_treatments = 2
    treatments = np.zeros(shape=(MAX_TIME, num_treatments))
    for t in range(MAX_TIME):
        p = np.random.uniform()
        if p < 0.5:
            treatments[t, 0] = np.random.uniform(0.5, 1.0)
        else:
            treatments[t, 1] = np.random.uniform(0.5, 1.0)

    subclones = [Subclone(lbl="A",c=0.03, alpha=[0.05, 1.1], prop=0.05),
                 Subclone(lbl="B", c=0.03, alpha=[1.1, 0.03], prop=0.05),
                 Subclone(lbl="S", c=0.0, alpha=[1.0, 1.0], prop=0.9),
                 ]
    tnames = ["Drug A", "Drug B"]
    run_sim(MAX_TIME, num_treatments, treatments, subclones, tnames)


import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from graph import Graph
from subclone import Subclone
from doctor import Doctor
from payoff_matrix import PayoffMatrix


class Simulation:
    """
        Simulation Class contains relevant methods to simulate the evolution
        of the subclone colonies.  In this example, it is hardcoded to three
        colonies.
    """
    def __init__(self, subclones, treatments, adjacency=None):
        """
            :attr time: Time stamp starting from t=0
            :attr subclones (list): list of Subclone objects
            :attr cs(np.ndarray): the cost of of resistance for each subclone i
            :attr alphas (np.ndarray): 2d array (num subclones x num treatments) of the added susceptibility to each drug subclone i
            :attr treatments (np.ndarray): two dimensional array total timesteps x num_treatments
        """
        self.t = 0
        self.subclones = subclones
        self.names = [sc.label for sc in self.subclones]
        self.treatments = treatments
        self.num_timesteps = self.treatments.shape[0]
        self.adj = adjacency
        self.graph = None


    def adjust_proportion(self):
        """
            Adjusts using p_t+1 = p_t * W(i)/W
            This assumes fitness are already updated / calculuated
        """
        avg_fit = self.calc_avg_fitness()
        for c in self.subclones:
            c.prop *= (c.fitness / avg_fit)

            # ensure proportion can't surpass 1
            c.prop = min(c.prop, 1)
            c.prop = max(c.prop, 0)


    def calc_avg_fitness(self):
        """
        Given fitness environment, returns average fitness.
        """
        return np.sum([c.prop * c.fitness for c in self.subclones])

    def run_step(self):
        for c in self.subclones:
            c.update_fitness(self.treatments[self.t, :])
        if self.t < self.num_timesteps:
            self.t += 1
        else:
            print ("Simulation complete after %d timesteps" %self.num_timesteps)
        return  self.calc_avg_fitness()


def run_sim(max_time, num_treatments, treatments, subclones, treatment_names, doc_times=None, distro=None, doc_decay=0, dirname=None, adj=None):
    '''

    :param max_time (int): The number of timesteps to consider
    :param num_treatments (int): Number of possible drugs administered
    :param treatments (np.ndarray): shape (max_time, num_treatments) for pre-scheduled treatments. Can be zeros if doctor responds in real time
    :param subclones (list): Subclonal populations present initially.
    :param treatment_names (list): names of each treatments (list length = num_treatments)
    :param doc_interval (np.ndarray): shape (max_time) with booleans representing if doctor can act at t. Default: can act at each step.
    :param doc_times (np.ndarray): shape (max_time) True or False times where doctor can act
    :param distro (np.ndarray): shape (num_treatments) distrbution of strategy profile
    :param doc_decay(float): Amount of exponential decay in treatment intensity after doctor action
    :param dirname (str): Path to save files
    :param adj(np.ndarray): shape (len(subclones), len(subclones)) specifying graph structure adjacency matrix with [0-1] edge weights
    :return:
    '''
    if not treatments.shape[0] == max_time and treatments.shape[1] == num_treatments:
        raise ValueError("Unexpected treatment shape, should be (timesteps, number of treatments")
    if len(treatment_names) != treatments.shape[1]:
        raise ValueError("Treatment names and number of treatments should be same")
    if doc_times is None:
        doc_times = np.ones(shape=num_treatments)
    stratlog = []
    model = Simulation(subclones, treatments, adj)
    graph = Graph(model)
    model.graph = graph
    doc = Doctor(model, schedule=doc_times, decay=doc_decay, distro=distro)
    if distro.shape[0] != doc.num_strats:
        raise ValueError("Define distribution over all doctor strategies")
    log_fitness = np.zeros(shape=(MAX_TIME, len(subclones)))
    log_props = np.zeros(shape=(MAX_TIME, len(subclones)))
    log_avg_fitness = np.zeros(shape=(MAX_TIME))
    avg = model.calc_avg_fitness()
    for t in range(MAX_TIME):
        log_fitness[t, :] = np.array([c.fitness for c in model.subclones])
        log_props[t, :] = np.array([c.prop for c in model.subclones])
        log_avg_fitness[t] = avg

        if doc_times[t]:
            stratname, strat = doc.mixed_strategy() # perform doctor action according to predefined strategy profile (allows randomness)
            stratlog.append(stratname)
            strat(doc)
        matx = PayoffMatrix(model)
        if t == 0 or t == 2:
            matx.print_matrix()
        del(matx)
        avg = model.run_step()
        # Adjust Proportion
        model.adjust_proportion()
        graph.update()
    graph.plot(title="test")

    savedir = os.path.join(os.path.split(os.getcwd())[0], "data", "simulations")
    if not dirname:
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

    with open(os.path.join(full_dir, "params", "params.txt"), "w+") as f:
        f.write("alpha: \n " + str(alphas) +
                " \ncs:\n" + str(cs)
                + "\ntreatments:\n"
                + str(treatments)
                + "\nstratlog: \n" + str(stratlog))



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
    plt.plot (x_axis, log_avg_fitness,  label="average")
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
    # for t in range(MAX_TIME):
    #     p = np.random.uniform()
    #     if p < 0.5:
    #         treatments[t, 0] = np.random.uniform(0.5, 1.0)
    #     else:
    #         treatments[t, 1] = np.random.uniform(0.5, 1.0)

    subclones = [Subclone(lbl="A",c=0.03, alpha=[0.1, 2.2], prop=0.05),
                 Subclone(lbl="B", c=0.03, alpha=[1.1, 0.05], prop=0.05),
                 Subclone(lbl="S", c=0.01, alpha=[0.8, 0.7], prop=0.9),
                 ]
    tnames = ["Drug A", "Drug B"]

    # Let doctor prescribe every 5 time intervals
    dt = np.zeros(MAX_TIME)
    dt[::10] = 1
    np.random.seed(0)
    magnitude = 0
    # define a strategy profile for doctor to decide actionsE online
    distro = np.array([0, 0.5, 0, 0.5])
    adjacency_matx = np.array([[1, 0.3, 0],
                               [0.3, 1, 0],
                               [0, 0, 1]])
    run_sim(MAX_TIME, num_treatments, treatments, subclones, tnames, doc_times=dt,
            distro=distro, doc_decay=0.98, dirname="graph_test", adj=adjacency_matx)


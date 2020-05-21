import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

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


def run_sim(max_time, num_treatments, treatments, subclones, treatment_names, save, doc_times=None, distro=None, doc_decay=0, dirname=None, adj=None):
    '''

    :param max_time (int): The number of timesteps to consider
    :param num_treatments (int): Number of possible drugs administered
    :param treatments (np.ndarray): shape (max_time, num_treatments) for pre-scheduled treatments. Can be zeros if doctor responds in real time
    :param subclones (list): Subclonal populations present initially.
    :param treatment_names (list): names of each treatments (list length = num_treatments)
    :param save(bool): true if you want to save these results
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
        if t == max_time - 1:
            print ('Payoff Matrix: ', t)
            matx.print_matrix()
        del(matx)
        avg = model.run_step()
        # Adjust Proportion
        model.adjust_proportion()
        graph.update()
    info = pd.DataFrame([(sc.label, np.round(sc.prop, 4), sc.fitness, sc.parent, np.round(sc.alpha, 2), np.round(sc.c, 2)) for sc in subclones], columns=["label", "prop", "fitness", "parent", "alpha", "c"])
    full_dir = None
    if save:
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
    if save:
        if not os.path.exists(os.path.join(full_dir, "params")):
            os.mkdir(os.path.join(full_dir, "params"))
        np.save(os.path.join(full_dir, "params", "alphas.npy"), alphas)
        np.save(os.path.join(full_dir, "params", "cs.npy"), cs)
        np.save(os.path.join(full_dir, "params", "treatments.npy"), treatments)

        with open(os.path.join(full_dir, "params", "params.txt"), "w+") as f:
            f.write(info.to_string())
            f.write("\ntreatments:\n"
                    + str(treatments)
                    + "\nstratlog: \n" + str(stratlog))

    if save:
        graph.plot(title="Subclonal Graph", savefile=os.path.join(full_dir, "graph.png"))
    else:
        graph.plot(title="Subclonal Graph", savefile=None)

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
    if save:
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
    if save:
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
    if save:
        plt.savefig(os.path.join(full_dir, "treatments.png"))
    plt.show()

def generate_random_subclones(n_init, max_subclones, seed, num_treatments, eps=0.0001, thresh=0.9):
    np.random.seed(seed)
    subclones = []
    p_left = 1
    num_children = {}
    for j in range(n_init):
        lbl = chr(j + 65)
        c = np.random.uniform()
        alpha = np.random.uniform(size=num_treatments)
        prop = max(0.0, np.random.uniform(eps,p_left))
        p_left -= prop
        subclones.append(Subclone(lbl=lbl, c=c, alpha=alpha, prop=prop))
        num_children[lbl] = 0

    for j in range(max_subclones):
        props = np.array([sc.prop for sc in subclones])
        props = props / np.sum(props)
        parent = np.random.choice(subclones, p=props) # weight by parent proportion
        c = max(0, min(1, np.random.uniform(low=-0.1, high=0.1) + parent.c))
        alpha = np.random.uniform(low=-0.1, high=0.1, size=num_treatments) + parent.alpha
        alpha[alpha < 0] = 0
        alpha[alpha > 1] = 1
        prop = max(0.0, np.random.uniform(eps,p_left))
        p_left -= prop
        lbl = parent.label + "." + chr(num_children[parent.label] + 97)
        num_children[parent.label] += 1
        kid = Subclone(lbl=lbl, c=c, alpha=alpha, prop=prop)
        kid.parent = parent
        subclones.append(kid)
        num_children[lbl] = 0

    adjacency_matx = np.zeros((len(subclones), len(subclones)))
    for j in range(len(subclones)):
        for k in range(len(subclones)):
            phtype_j = np.hstack([subclones[j].c, subclones[j].alpha])
            phtype_k = np.hstack([subclones[k].c, subclones[k].alpha])
            sim = phtype_j.dot(phtype_k) / (np.linalg.norm(phtype_j) * np.linalg.norm(phtype_k))
            if subclones[j].parent == subclones[k] and sim > thresh:
                adjacency_matx[k][j] = sim
                adjacency_matx[j][k] = sim

    return subclones, adjacency_matx

if __name__ == "__main__":
    MAX_TIME = 100
    num_treatments = 2
    treatments = np.zeros(shape=(MAX_TIME, num_treatments))
    tnames = ["Drug A", "Drug B"]
    seed = 0
    # Let doctor prescribe every 5 time intervals
    dt = np.zeros(MAX_TIME)
    dt[::10] = 1
    np.random.seed(seed)

    subclones, adjacency_matrix = generate_random_subclones(n_init=3, max_subclones=5, seed=seed, num_treatments=num_treatments)
    names = [sc.label for sc in subclones]
    print ("ADJACENCY")
    print (pd.DataFrame(adjacency_matrix, columns=names, index=names))

    distro = np.array([0, 0, 0, 0])
    run_sim(MAX_TIME, num_treatments, treatments, subclones, tnames, save=True, doc_times=dt,
            distro=distro, doc_decay=1.0, dirname="zero_fit_nostrat_more_subclones_nodecay_seed%d"%seed, adj=adjacency_matrix)




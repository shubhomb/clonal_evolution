"""
    This file contains functions necessary to simulate the entire evolution
    of a finite number of subclone colonies using a graph framework. 

    First, we set up an environment containing the names of subclone colonies,
    the adjacency matrix describing the relationship, the alpha values denoting
    their immunity and their current state of relative proportion.

    Next, we pass this environment to the simulation engine with a few additional
    parameters and we can watch the network evolve.  

"""

import numpy as np
import json
from doctor import *
from graph import *

def parse_json(path_to_file):
    """Return subclone environment as json dictionary"""
    with open(path_to_file) as f:
        data = json.load(f)
    return data

class Environment():
    """
        This is used as an input to contruct the graph.
    """
    def __init__(self, names, matrix, alpha, prop):
        self.names      = names  # Names of Subclone Colony (List)
        self.relations  = matrix  # Adj Matrix representing relations
        self.alpha      = alpha
        self.prop       = prop

    def log(self):
        print('Logging Environment:')
        items = [f' -->  {d}' for d in zip(self.names, self.alpha, self.prop)]
        for i in items:
            print(i)
        print(20*'-*')
    
    def get_env_data(self):
        return [d for d in zip(self.names, self.alpha, self.prop)]    


class Simulation():
    def __init__(self, env, graph, MAX_TIME, debug=False):
        self.env = env
        self.graph = graph
        self.MAX_TIME = MAX_TIME
        self.debug = debug
        
    def printsim(self):
        self.env.log()
        self.graph.log()
        self.graph.nxgraph
        print(self.MAX_TIME)


    def evolve(self):
        """ Takes in graph and evolves graph """                
        pass
    


if __name__ == "__main__":
    """
        Begins simulation
    """
    MAX_TIME = 100
    num_treatments = 2
    treatments = np.zeros(shape=(MAX_TIME, num_treatments))

    # Eventually put this into a json environment object
    names = ['drugA', 'drugB', 'drugC', 'drugD']
    relations = np.array([
                    [1, 0.3, 0.4, 0.6],
                    [1, 0.3, 0.4, 0.6],
                    [1, 0.3, 0.4, 0.6],
                    [1, 0.3, 0.4, 0.6]])
    alphas = [0.3, 0.2, 0.3, 0.2]
    props = [0.25, 0.25, 0.25, 0.25]

    # Make environment
    env = Environment(names, relations, alphas, props)
    
    graph = Graph(env)

    sim = Simulation(env, graph, MAX_TIME)
    for t in range(MAX_TIME):
        sim.evolve()
    



    # Let doctor prescribe every 5 time intervals
    doctor_num_treatments = 5 
    
    
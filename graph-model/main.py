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
import doctor
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

    def update_fitness(self):
        """
            Updates each node's fitness and recalculates average fitness
        """
        total = 0
        for node in self.graph.nxgraph.nodes:
            node.update_fitness()
            total += node.fitness
        self.graph.avgfitness =  total / (len(self.graph.nxgraph.nodes))

    def update_proportion(self):
        """
            Update each colony's proportions AFTER update_fitness(self)
            is run.
        """
        for node in self.graph.nxgraph.nodes:
            node.colony.prop *= (node.fitness ) / (self.graph.avgfitness)

    def doctor(self):
        """ 
        Target maximum node and all nodes within some constant
        
        Returns target node
        """
        DEPTH = 1
        # target_node = nx.maximal_independent_set(sim.graph.nxgraph)[0]

        all_degrees = list(self.graph.nxgraph.degree(self.graph.all_nodes, weight='weight'))

        print(all_degrees)
        target_node = max(all_degrees, key=lambda item:item[1])[0]

        return target_node


    def evolve(self):
        """ Takes in graph and evolves graph using doctor strategy
        """                
        target_node = self.doctor()
        print(f'Target Node: {target_node.colony.name}')
        self.graph.apply_medicine(target_node, 0.9, debug=True)
        

    def log(self):
        print('Model parameters:')
        for node in self.graph.nxgraph.nodes():
            node.log()
    

if __name__ == "__main__":
    """
        Begins simulation
    """
    MAX_TIME = 4
    num_treatments = 2
    treatments = np.zeros(shape=(MAX_TIME, num_treatments))

    # Eventually put this into a json environment object
    names = ['drugA', 'drugB', 'drugC', 'drugD']
    relations = np.array([
                    [1, 0.3, 0, 0],
                    [0.3, 1, 0.2, 0],
                    [0, 0.2, 1, 0.8],
                    [0, 0, 0.8, 1]])
    alphas = [0.3, 0.2, 0.3, 0.2]
    props = [0.25, 0.25, 0.25, 0.25]

    # Make environment
    env = Environment(names, relations, alphas, props)
    
    graph = Graph(env)
    

    sim = Simulation(env, graph, MAX_TIME)

    # Model parameters at time t = 0
    print('-'*10 + f'SIMULATION TIME 0' + '-'*10)
    sim.graph.plot(0, fitness=True)
    sim.log()

    for t in range(1, MAX_TIME):
        print('-'*10 + f'SIMULATION TIME {t}' + '-'*10)
        
        sim.evolve() # Evolve using specified Doctor's strategy
        sim.update_fitness()  #Update fitness   
        sim.update_proportion() # Update proportion  MUST BE AFTER FITNESS 
        # gives visual
        sim.graph.plot(t, fitness=True)
        
        sim.log() # Log data

    # Let doctor prescribe every 5 time intervals
    doctor_num_treatments = 5 
    
    
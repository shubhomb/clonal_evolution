''' Set up framework for graph model. Contains Subclone, Node and Graph class
    The graph model attempts to describe the similarity between subclone populations. 
    Nodes represent population colonies and contain information on the colony size and fitness.
    

    Graph:

    ----Node-----                        ----Node-----
    |           |                       |            |
    | Colony    |-----------------------|   Colony   |
    | birthtime |                       | birthtime  |
    | edges     |                       |   edges    |
    |-----------|                       |------------|


'''
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

class Colony():
    """
        Colony object is inside node object
        Each colony contains:
            - Name 
            - Relation [relation with 1, relation with 2, ... ]
            - Alpha 
            - Relative Proportion
    """
    def __init__(self, name, relation, alpha, prop):
        self.name = name       #'drugA' for example 
        self.relation = relation    
        self.alpha = alpha
        self.prop = prop
        self.resistant_to_med = False # [True] if resistant to current drug
    
    def get_colony_info(self):
        return (self.name, self.relation, self.alpha, self.prop)
    
    def update_alpha(self, newalpha):
        self.alpha = newalpha
    
    def update_prop(self, newprop):
        self.prop = newprop


class Node():
    """
        Nodes contain the following:
            - Colony Object describing subclone colony 
            - Birthtime 
            - Edges to other nodes
    """
    def __init__(self, colony, birth_time, edges):
        # TODO add assertions to verify edges input
        self.colony = colony # Colony Info
        self.birth_time = birth_time #birth_time
        self.edges = edges  # [(node, edge_weight), (node2, edge_weight), ...]
        self.fitness = 0

    def update_fitness(self):
        # TODO
        # W(v) = 1 - (cost of resis) - alpha*d_A(t) + (1-PA)X(T)
        
        cost_of_resis = 0.1
        alpha_medicine = self.colony.alpha*(1 - self.colony.resistant_to_med)
        third_term = 0.3

        fitness = max(0, 1 - cost_of_resis - alpha_medicine)
        self.fitness = fitness
    
    def get_node_info(self):
        """ Return attributes """
        return (self.colony, self.birth_time, self.edges)
    
    def update_colony(self, newalpha, newprop):
        self.colony.alpha = newalpha
        self.colony.prop = newprop
    
    def debug(self):
        return self.colony.name
    
    def log(self):
        print(f'Node: {self.debug()}')
        print(f'\t Birthtime: {self.birth_time}')
        # print(f'\t Edges' + "*-"*10)
        # for edge in self.edges:
            # print(f'\t --> {edge[1].colony.name} {edge[1]} with weight {edge[0]}')
        print(f'\t \t Alpha: {self.colony.alpha}')
        print(f'\t \t Prop: {self.colony.prop}')
        print(f'\t \t Resistant: {self.colony.resistant_to_med}')
        print(f'\t \t Fitness: {self.fitness}')
        
    

        

class Graph():
    """
        The graph abstraction is the traditional graph with nodes and edges.
        In particular, this contains functions capable of accessing a node,
        its neighbors, removing the contained object, and storing a new object
        in its place.
        
        We have two representations

        - A networkx graph instance

        -   We represent the graph as an adjacency list.
            The graph is a mapping such that: map[node] = [list of neighbor nodes]
            where [list of neighbor nodes] contains element tuples (edgeweight, node)

        We choose this design for quick lookup for doctor treatment.
    """

    def __init__(self, env):
        """
            Given environment object consisting of:
                - names  (list): names of subclone colony
                - relations (matrix): adj matrix representing relations
                - alpha  (list) : of corresponding alpha constants
                - prop  (list) : of corresponding initial proportions

            Has attributes:
                - Point map (dict): mapping from node to its neighbors
                - all_nodes (list): list of all nodes
                - all_edges  (list): list of all edgs
                - networkx_graph (networkx graph):
                - avg fitness
                - label_dict (map) : Maps node to its name (for plotting)

        Initializes a graph instance
        """
        self.pointmap = {}  # map[node] = [(node, weight), (node2, eweight), ..]
        self.all_nodes = []  # [Node1, Node2, ... ]
        self.all_edges = []  # [(node1, node2, weight), ...] -- For printing
        self.nxgraph = nx.Graph()
        self.avgfitness = 0

        # Plotting purposes
        self.label_dict = {}  # Maps node to name (for plotting purposes)

        #  Initialize all nodes
        for (name, neigh, alpha, prop) in zip(env.names, env.relations, env.alpha, env.prop):
            new_colony = Colony(name, neigh, alpha, prop)
            new_node = Node(new_colony, 0, [(None, None)])
            self.pointmap[new_node] = [(None, None)]
            self.all_nodes.append(new_node)
            self.label_dict[new_node] = name

        # Add edges -- hacky way -- can make cleaner:
        ptr = 0  # current node exploring
        for (name, neigh) in zip(env.names, env.relations):
            curr = 0  # pointer to each neighbor
            neighbor_list = []  # tuples (weight, node) list
            # Iterate through all neighbors and add nonzero ones to list
            for neighbor_weight in neigh:
                if neighbor_weight > 0 and curr != ptr:
                    neighbor_list.append((neighbor_weight, self.all_nodes[curr]))
                    self.all_edges.append((self.all_nodes[ptr], self.all_nodes[curr], neighbor_weight))
                curr += 1
            self.pointmap[self.all_nodes[ptr]] = neighbor_list
            self.all_nodes[ptr].edges = neighbor_list

            ptr += 1

        self.nxgraph.add_nodes_from(self.all_nodes)
        self.nxgraph.add_weighted_edges_from(self.all_edges)

    def get_data(self):
        """
            Returns pandas dataframe at current timestep
        """
        
        all = []
        for node in self.nxgraph.nodes:
            dic = {
                    'name': node.colony.name,
                    'fitness': float(node.fitness),
                    'prop': float(node.colony.prop),
                  }
            all.append(dic)
        return pd.DataFrame(all)
            
            



    def log(self):
        """ Prints debug information  """
        for node in self.pointmap:
            print(f'Node: {node.debug()} obj: {node}: ')
            for neighbor, weight in self.pointmap[node]:
                print(f'\t {neighbor} with weight {weight}')

    def get_networkx_graph(self):
        G = nx.Graph()
        G.add_nodes_from(self.all_nodes)
        G.add_weighted_edges_from(self.all_edges)
        return G




    def plot(self, title, fitness=False):
        import matplotlib.pyplot as plt
        G = self.get_networkx_graph()
        pos = nx.spring_layout(G)
        # Plot graph with labels as specified by label_dict
        nx.draw(G, pos, labels=self.label_dict, with_labels=True)
        
        # Create edge label Dictionary to label edges:
        edge_labels = nx.get_edge_attributes(G,'weight')
        

        nx.draw_networkx_edge_labels(G, pos, labels = edge_labels)

        # -------- Draw fitness labels above -----

        pos_higher = {}
        y_offset = 0.07 # Might have to play around with
        for k, v in pos.items():
            pos_higher[k] = (v[0], v[1] + y_offset)        

        fit = {n: f'fitness: {n.fitness}' for n in self.all_nodes}
        nx.draw_networkx_labels(G, pos=pos_higher, font_size=10, font_color='black', labels=fit)
        plt.savefig(f'plots/Plot time {title}.png')
        plt.close()





    def apply_medicine(self, target_node, depth, debug=False):
        """
            Applies treatment to each node within depth [depth] by
            updating the dA parameter for each node .
        """
        if debug:
            print('Applying medicine to nodes....')
        target_nodes = nx.ego_graph(self.nxgraph, target_node, depth, center=True, undirected=True, distance='weight')
        if debug:
            print(f'Considering a total of {len(target_nodes)} nodes: ')
        for curr in target_nodes:
            if debug:
                print(f'Targeting: {curr.colony.name}')
        

        for node in self.nxgraph.nodes:
            if node not in target_nodes:
                node.colony.resistant_to_med = True
                if debug:
                    print(f'resistant: {node.colony.name}')
            else:
                node.colony.resistant_to_med = False
                if debug:
                    print(f'not resistant: {node.colony.name}')
        if debug:
            print(f'Updated dA parameters:')

        
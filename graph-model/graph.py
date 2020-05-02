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
        self.name = name 
        self.relation = relation
        self.alpha = alpha
        self.prop = prop

class Node():
    """
        Nodes contain the following:
            - Colony Object describing subclone colony 
            - Birthtime 
            - Edges to other nodes
    """
    def __init__(self, colony, birth_time, edges):
        # TODO add assertions to verify edges input
        self.colony = colony
        self.birth_time = birth_time
        self.edges = edges  # [(node, edge_weight), (node2, edge_weight), ...]
    def treatment(self):
        # TODO
        pass
    def debug(self):
        return self.colony.name

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

    def debug(self):
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

    def plot(self):
        import matplotlib.pyplot as plt
        G = self.get_networkx_graph()
        nx.draw(G)
        plt.savefig('Graph Plot.png')




    def __init__(self, env):
        """
            Given environment object consisting of:
                - names  (list): names of subclone colony
                - relations (matrix): adj matrix representing relations
                - alpha  (list) : of corresponding alpha constants 
                - prop  (list) : of corresponding initial proportions

        Initializes a graph instance
        """
        self.pointmap = {} # map[node] = [(node, weight), (node2, eweight), ..]
        self.all_nodes = []
        self.all_edges = [] # (node1, node2, weight) -- For printing
        #  Initialize all nodes
        for (name, neigh, alpha, prop)  in zip(env.names, env.relations, env.alpha, env.prop):
            new_colony = Colony(name, neigh, alpha, prop)
            new_node = Node(new_colony, 0, [(None, None)])
            self.pointmap[new_node] = [(None, None)]
            self.all_nodes.append(new_node)  
        # Add edges -- hacky way -- can make cleaner:
        ptr = 0 # current node exploring
        for (name, neigh)  in zip(env.names, env.relations):
            curr = 0 #pointer to each neighbor
            neighbor_list = [] #tuples (weight, node) list
            # Iterate through all neighbors and add nonzero ones to list
            for neighbor_weight in neigh:
                if neighbor_weight > 0 and curr != ptr:
                    neighbor_list.append((neighbor_weight, self.all_nodes[curr])) 
                    self.all_edges.append((self.all_nodes[ptr], self.all_nodes[curr], neighbor_weight))
                curr += 1
            self.pointmap[self.all_nodes[ptr]] = neighbor_list
            ptr+=1
        
        
        
    
        



        

        

    def add_node(self, node):
        """
            Adds a node to the graph
            :param node (Node): node to add to graph
        """
        self.pointmap[node] = node.edges


    def distance(self, x1, x2):
        """ Returns distance between two nodes (sum of shortest edge path) """
        pass


    def apply_medicine(self, target_node, depth):
        """
            TODO: 
            Applies treatment to each node within depth [depth]
        """
        raise NotImplementedError
        if depth <= 0:
            return
        for (neighbor, weight) in self.pointmap[target_node]:
            continue

        

    def nearest_neighbors(self, x1):
        """ Returns the Node and weight corresponding to the nearest neighbor """
        out_edges = x1.edges
        return min(data, key = lambda t: t[0])
                 
    def spawn_mutant(self):
        """ Adds bunch of neighboring nodes with small weight. """
        pass


    def display(self, title):
        """ Stale for d > 3 """
        if self.d == 2:
            f = plt.figure()
            if title:
                plt.title(title)
            plt.xlabel("param1")
            plt.ylabel("param2")
            for k, v in self.points.items():
                plt.scatter(v[0], v[1])
                plt.annotate(k.name, (v[0], v[1]))
            plt.show()
        elif self.d == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            if title:
                ax.set_title(title)
            ax.set_xlabel("param1")
            ax.set_ylabel("param2")
            ax.set_zlabel("param3")

            for k, v in self.points.items():
                ax.scatter(v[0], v[1], v[2], s=20)
                ax.text(v[0], v[1], v[2], k.name, "x", fontsize=5)
            plt.show()
        else:
            raise NotImplementedError("can't display graph with over 3 dimensions")





if __name__ == "__main__":
    coord1 = np.array([0, 1, 0])
    progenitor = Node("origin", 0, coord1)
    graph = Graph(dims=3)
    graph.add_node(progenitor)

    for t in range(100):
        eps = np.random.normal(coord1.shape)
        pt = np.random.choice(np.array(list(graph.points.keys())))
        newnode = Node(str(t+1), t, graph.points[pt] + eps)
        graph.add_node(newnode)
    graph.display("random")

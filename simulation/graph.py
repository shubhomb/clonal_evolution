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




class Graph():
    """
        The graph abstraction is a traditional graph with nodes and edges.
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
    def __init__(self, sim):
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
        self.nxgraph = nx.Graph()
        self.sim = sim
        self.update()

    def update(self):
        for j in range(len(self.sim.subclones)):
            weights = self.sim.adj[j, :]
            if not self.sim.subclones[j] in self.nxgraph.nodes:
                self.nxgraph.add_node(self.sim.subclones[j], name=self.sim.subclones[j].label)

            # nodes that have not been added but are added automatically without label if these w
            for k in range(len(weights)):
                if not self.sim.subclones[k] in self.nxgraph.nodes:
                    self.nxgraph.add_node(self.sim.subclones[j], name=self.sim.subclones[j].label)
                if weights[k] > 0:
                    self.nxgraph.add_edge(self.sim.subclones[j], self.sim.subclones[k], weight=weights[k])

    def get_data(self):
        """
            Returns pandas dataframe at current timestep
        """

        all = []
        for subclone in self.nxgraph.nodes:
            dic = {
                'name': subclone.label,
                'fitness': float(subclone.fitness),
                'prop': float(subclone.colony.prop),
            }
            all.append(dic)
        return pd.DataFrame(all)


    def plot(self, title, savefile=None):
        pos = nx.spring_layout(self.nxgraph, k=2/np.sqrt(len(self.sim.subclones)))
        # Plot graph with labels as specified by label_dict
        cmap = plt.cm.get_cmap("Pastel2", len(self.sim.subclones))
        colors = [cmap(i / len(self.sim.subclones) + 0.01) for i in range(len(self.sim.subclones))]
        labs = dict(zip(self.sim.subclones, self.sim.names))

        # weird nx thing makes you have to sort the label for (only) the node size argument in draw_networkx
        pops = dict(zip(self.sim.names, [sc.prop * 1000 for sc in self.sim.subclones]))
        pops = {k: pops[k] for k in sorted(pops)}


        nx.draw_networkx(self.nxgraph, pos, node_size=list(pops.values()),with_labels=True,labels=labs, width=0.5, font_size=10 , node_color=colors)

        # Create edge label Dictionary to label edges:
        edge_labels = nx.get_edge_attributes(self.nxgraph, "weight")

        # nx.draw_networkx_edge_labels(self.nxgraph, pos, labels=edge_labels, font_size=5)
        # -------- Draw fitness labels above -----

        pos_higher = {}
        y_offset = 0.04  # Might have to play around with
        for k, v in pos.items():
            pos_higher[k] = (v[0], v[1] + y_offset)
        fit = {n: "f: %.2f"%n.fitness for n in self.sim.subclones}
        plt.title(title)
        nx.draw_networkx_labels(self.nxgraph, pos=pos_higher, font_size=5, font_color='black', labels=fit)
        if savefile:
          plt.savefig(savefile)
        plt.show()
        plt.close()

    def apply_medicine(self, target_node, depth, verbose=False):
        """
            Applies treatment to each node within depth [depth] by
            updating the dA parameter for each node .
        """
        target_nodes = nx.ego_graph(self.nxgraph, target_node, depth, center=True, undirected=True, distance='weight')
        if verbose:
            print('Applying medicine to nodes....')
            print(f'Considering a total of {len(target_nodes)} nodes: ')
        for curr in target_nodes:
            if verbose:
                print(f'Targeting: {curr.label}')

''' Set up framework for graph model. Contains Subclone, Node and Graph class
'''
import numpy as np
import matplotlib.pyplot as plt

class Node():
    def __init__(self, name, birth_time, coord, parent=None):
        self.name = name
        self.birth_time = birth_time
        self.parent = parent
        self.coords = coord

class Graph():
    def __init__(self, dims):
        self.d = dims
        self.points = {}

    def add_node(self, node):
        self.points[node] = node.coords

    def display(self, title):
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

    def distance(self, x1, x2):
        return np.sqrt(np.sum(np.square(x2 - x1)))

    def nearest_neighbors(self, x1, coords):
        return np.sort([self.distance(x1, coord) for coord in coords])


    def spawn_mutant(self):
        pass




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

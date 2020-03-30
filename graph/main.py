import graph
import numpy as np
import json
import networkx as nx

def parse_json(path_to_file):
    """Return subclone environment as json dictionary"""
    with open(path_to_file) as f:
        data = json.load(f)
    return data

class PhenoSpace():
    def __init__(self, dims):
        self.d = dims
        self.environment_func = None
        self.doctor_func = None

    def distance(self, x1, x2):
        return np.sqrt(np.sum(np.square(x2 - x1)))

    def nearest_neighbors(self, x1, coords):
        return np.sort([self.distance(x1, coord) for coord in coords])



if __name__ == "__main__":
    
    env = parse_json('sub_env.json')
    
    g = graph.Graph(env)
    print(f'Graph : {g.tag}')


    subclone = graph.Node("sameer", t, parent, coord)
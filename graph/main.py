import graph
import numpy as np
import json
import networkx as nx

def parse_json(path_to_file):
    """Return subclone environment as json dictionary"""
    with open(path_to_file) as f:
        data = json.load(f)
    return data

class Simulation():
    def __init__(self, graph, progenitor, max_timesteps):
        self.t = 0
        self.max_timesteps = 0
        self.graph = graph




if __name__ == "__main__":
    
    env = parse_json('sub_env.json')
    
    g = graph.Graph(env)
    print(f'Graph : {g.tag}')


    subclone = graph.Node("sameer", t, parent, coord)
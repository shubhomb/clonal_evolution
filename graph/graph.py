''' Set up framework for graph model. Contains Subclone, Node and Graph class
'''
import numpy as np
from enum import Enum


class Subclone(Enum):
    """ 
        Enumerate different flavors of sublclone populations. 
    """
    # TODO: rename these eventually
    BLANK   = 0
    ALPHA   = 1
    BETA    = 2
    # etc.

class Node():
    def __init__(self):        
        self.subclone_type = Subclone.BLANK
        self.out_edges  = []
        self.in_edges   = []

    
class Graph():
    def __init__(self, env):
        """ 
        Initalizes graph instance with properties specified in 
        sub_env.json file 
        '"""
        # TODO: Create Graph
        self.tag = env['name']


    def mutate(self):
        pass

    def evolve(self):
        pass



    
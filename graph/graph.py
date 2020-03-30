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
    def __init__(self, name, birth_time, coord, parent=None):
        self.name = name
        self.birth_time = birth_time
        self.parent = parent
        self.coordinate = coord

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



    
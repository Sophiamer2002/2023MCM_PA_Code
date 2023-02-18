from tree import tree
import numpy.random as rd
from simulator import simulator 

class cell:
    def __init__(self, simulator: simulator, i, j):
        self.simulator = simulator
        self.x = i
        self.y = j

        typeidx = rd.randint(0, simulator.numtreetype * 2)
        self.tree = tree(simulator.treetypes[typeidx], simulator.treeinfo) if typeidx < simulator.numtreetype else None
        self.shadow = []


    def has_tree(self):
        return self.tree != None
    
    def get_numseed(self):
        return 6 * self.tree.treeinfo['ShTol']

    



import numpy.random as rd
from simulator import simulator
from .tree import tree


class cell:
    def __init__(self, simulator: simulator, i, j, isRandom=True, treetype=None):
        self.simulator = simulator
        self.x = i
        self.y = j

        if isRandom:
            typeidx = rd.randint(0, simulator.numtreetype * 2)
            self.tree = tree(simulator.treetypes[typeidx], simulator.treeinfo) if typeidx < simulator.numtreetype else None
        else:
            self.tree = tree(treetype, simulator.treeinfo)

    def has_tree(self):
        return self.tree is not None

    def remove_tree(self):
        if self.tree is not None:
            del self.tree
            self.tree = None

    # get function
    def get_numseed(self):
        return 6 * self.tree.treeinfo['ShTol']

    def get_gs(self):
        return self.tree.treeinfo['g']

    def get_s(self):
        return self.tree.treeinfo['s']

    def get_maxH(self):
        return self.tree.treeinfo['Hmax']

    def get_crownsize(self):
        return self.tree.crownsize

    def get_DrTol(self):
        return self.tree.treeinfo['DrTol']

    def get_ShTol(self):
        return self.tree.treeinfo['ShTol']

    def get_cmax(self):
        return self.tree.treeinfo['cmax']

    def get_cmin(self):
        return self.tree.treeinfo['cmin']

    def get_DDmin(self):
        return self.tree.treeinfo['DDmin']

    def get_Amax(self):
        return self.tree.treeinfo['Amax']

    def get_treetype(self):
        return self.tree.treetype

    def get_height(self):
        return self.tree.get_height()

    def get_seasonal(self):
        '''evergreen or deciduous, return a char'''
        return self.tree.treeinfo['f'][0]

    def get_diameter(self):
        return self.tree.diameter

    def get_cl(self):
        return self.tree.crownsize * self.get_height()

    def get_wtmin(self):
        return self.tree.treeinfo['Wtmin']

    def get_wtmax(self):
        return self.tree.treeinfo['Wtmax']

    def get_ShTol_seedings(self):
        return self.tree.treeinfo['ShTol_seedings']

    def get_width(self):
        # 辐射到的曼哈顿距离
        return pow(pow(self.tree.diameter, self.tree.treeinfo['a']) * self.tree.treeinfo['ff'] / self.get_height(), 0.5) - 0.5

    def get_age(self):
        return self.tree.age

    def get_GDD(self, weatherinfo):
        if self.tree.treeinfo['f'][0] == 'E':
            return weatherinfo['GDD_E']
        else:
            return weatherinfo['GDD_D']

    def get_slowyears(self):
        return self.tree.slowyears        

    # set function
    def set_crownsize(self, cs):
        self.tree.crownsize = cs

    def set_diameter(self, new_d):
        self.tree.diameter = new_d

    def set_age(self):
        self.tree.age += 1

    def increment_slowyear(self):
        self.tree.slowyears += 1

    def reset_slowyear(self):
        self.tree.slowyears = 0


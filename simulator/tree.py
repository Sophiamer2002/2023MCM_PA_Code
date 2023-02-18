import numpy.random as rd
import numpy as np

class tree:
    '''
    member--treetype:       树木种类，是一个字符串
    member--treeinfo:       树木信息，是一个字典，如
                            {
                                "f": "E5",---------------------树木种类
                                "Hmax": 50,--------------------最大高度
                                "s": 75,-----------------------异速生长率
                                "g": 350,----------------------生长率
                                "Amax": 366,-------------------最大年龄
                                "DDmin": 841,------------------
                                "DrTol": 0.23,-----------------干旱忍受度
                                "Nreq": 3,---------------------
                                "ShTol": 1,--------------------阴影忍受度
                                "ShTol_seedings": 0.05,--------
                                "Wtmin": -6,-------------------
                                "Wtmax": 5,--------------------
                                "Br": 5,-----------------------
                                "cmin": 0.09,------------------最小树冠
                                "cmax": 0.53,------------------最大树冠
                                "a": 1.5,----------------------面积指数
                                "ff": 0.45,--------------------面积常数
                            }
    member--age:            树木年龄
    member--diameter:       树木直径，以米为单位，初始直径在treeinfo中（即树高为1.37m时的直径）
    '''
    def __init__(self, treetype: str, treeinfo: dict):
        self.treetype = treetype
        self.treeinfo = treeinfo[treetype]
        self.age = 0
        self.slowyears = 0
        self.diameter = 0.03
        self.crownsize = treeinfo['cs']

    def random_init(self):
        # TODO
        self.age = rd.randint(0, self.treeinfo['Amax'])

    def get_height(self):
        # TODO
        return (1.37 + (self.treeinfo['Hmax'] - 1.37) * (1 - np.exp(-self.treeinfo['s'] * self.diameter/(self.treeinfo['Hmax'] - 1.37)))    )
    
    def get_leafarea(self):
        return pow(self.diameter, self.treeinfo['a']) * self.crownsize * self.treeinfo['ff']
    


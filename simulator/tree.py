import numpy.random as rd


class tree:
    '''
    member--treetype:       树木种类，是一个字符串
    member--treeinfo:       树木信息，是一个字典，如
                            {
                                "f": "E5",---------------------树木种类
                                "Hmax": 50,--------------------最大高度
                                "s": 75,-----------------------
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
                                "initdiameter": 0.15-----------初始直径
                            }
    member--age:            树木年龄
    member--diameter:       树木直径，以米为单位，初始直径在treeinfo中（即树高为1.37m时的直径）
    '''
    def __init__(self, treetype: str, treeinfo: dict):
        self.treetype = treetype
        self.treeinfo = treeinfo[treetype]
        self.age = 0
        self.diameter = treeinfo['initdiameter']

    def random_init(self):
        # TODO
        self.age = rd.randint(0, self.treeinfo['Amax'])

    def get_height(self):
        # TODO
        return self.diameter * self.diameter
    
    def get_growth(self):
        return self.treeinfo['g']
    


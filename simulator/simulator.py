from cell import cell
import json


class simulator:
    '''
    member--x:          模拟方格的x轴个数
    member--y:          模拟方格的y轴个数
    member--treeinfo:   树木的具体信息
    member--treetypes:  参与模拟的树木种类（一个字符串数组）
    member--numtreetype:总共参与模拟的树木个数
    '''
    def __init__(self, x, y, treetypes):
        self.x = x
        self.y = y

        # specify the tree types and their information
        with open('./tree.json', 'r') as file:
            self.treeinfo = json.loads(file)
        self.treetypes = treetypes
        for __type in treetypes:
            if __type not in self.treeinfo:
                print(f"No tree type {__type}")
                exit()
        self.numtreetype = len(treetypes)

        # initialize the x*y cells
        self.grid = [[cell(self, i, j) for i in range(y)] for j in range(x)]
        self.__construct__shadow()

        self.__warm_up()

    def __update_one_year(self, weatherinfo):
        '''
        arg--weatherinfo:   一个字典，记录了一年之内的天气信息
                    格式:   {
                        "DrI": 0.4, 
                        "GDD_E": ?,
                        "GDD_D": ?,
                        "Tw": ?,
                    }
        '''
        # update by weather info 
        # TODO

        # 计算树木生长
        self.__tree_grow()

        # 计算树木传播种子
        self.__tree_spread_seed(weatherinfo)

        # 计算树木死亡
        self.__tree_death()

        # 更新shadow信息
        self.__construct__shadow()


        return self
    
    def __warm_up(self):
        for i in range(10):
            self.__update_one_year() #TODO

    def __tree_grow(self):
        for i in range(self.x):
            for j in range(self.y):
                self
        return self

    def __tree_spread_seed(self, weatherinfo):
        for i in range(self.x):
            for j in range(self.y):
                if not self.grid[i][j].has_tree():
                    break
                numseed = self.grid[i][j].get_numseed()

        return self
    
    def __tree_death(self):
        return self
    
    def __construct__shadow(self):
        # initialize the shadow variable in grid
        for i in range(self.x):
            for j in range(self.y):
                self.grid[i][j] # TODO
    


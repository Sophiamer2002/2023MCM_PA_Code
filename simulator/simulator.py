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


    def __construct__shadow(self):
        # initialize the shadow variable in grid
        for i, j in zip(range(self.x), range(self.y)):
            self.grid[i][j] # TODO

    def __update_one_year(self, weatherinfo):
        # update by weather info 
        # TODO
        return self
    
    def __warm_up(self):
        for i in range(10):
            self.__update_one_year() #TODO

    

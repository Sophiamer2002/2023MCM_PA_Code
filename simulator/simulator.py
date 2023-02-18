from .cell import cell
import json
import numpy as np


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
            self.treeinfo = json.load(file)
        self.treetypes = treetypes
        for __type in treetypes:
            if __type not in self.treeinfo:
                print(f"No tree type {__type}")
                exit()
        self.numtreetype = len(treetypes)

        # initialize the x*y cells
        self.grid = [[cell(self, i, j) for i in range(y)] for j in range(x)]
        self.shadow = np.zeros([self.x, self.y, 50], dtype=np.float64)
        self.__construct_shadow()

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

        # 计算树木crownsize
        self.__tree_crownsize()

        # 计算树木生长，只更新diameter
        self.__tree_grow(weatherinfo)

        # 计算树木传播种子
        self.__tree_spread_seed(weatherinfo)

        # 计算树木死亡
        self.__tree_death()

        # 更新shadow信息
        self.__construct_shadow()

        return self

    def __warm_up(self):
        for i in range(10):
            self.__update_one_year()  # TODO

    def __tree_grow(self, weatherinfo):
        for i in range(self.x):
            for j in range(self.y):
                if not self.grid[i][j].has_tree():
                    continue
                bs = self.grid[i][j].get_maxH() - 137
                cs = self.grid[i][j].get_s() / bs
                D = self.grid[i][j].get_diameter()
                H = self.grid[i][j].get_height()
                Hmax = self.grid[i][j].get_maxH()
                # TODO
                delta_D_optimal = self.grid[i][j].get_gs() * D * (1 - H/Hmax)/(2 * Hmax - bs * np.exp((cs * D) * (cs * D + 2)))

                # gdd
                if self.grid[i][j].get_seasonal() == 'E':
                    GDD = weatherinfo['GDD_E']
                elif self.grid[i][j].get_seasonal() == 'D':
                    GDD = weatherinfo['GDD_D']
                GR_gdd = np.max(0, 1. - np.exp((self.grid[i][j].get_DDmin() - GDD)/750))

                # drought
                DrTol_mean = self.__get_DrTol_mean(i, j, self.grid[i][j].get_DrTol())
                GR_drought = np.sqrt(np.max(0, 1. - weatherinfo['DrI']/DrTol_mean))

                # light
                crownsize = self.grid[i][j].get_crownsize()
                GR_sh = 0
                for x in range(50):
                    Lav = 1/self.grid[i][j].get_cl() * np.exp(-0.25 * self.shadow[i][j][x])
                    Lmax = 1 - pow(4.64, (0.05 - Lav))
                    Lmin = 2.24 * (1 - pow(1.136, (0.08 - Lav)))
                    GR_shx = np.max(0, Lmax + (Lmin - Lmax) * (self.grid[i][j].get_ShTol()-1)/(9 - 1))
                    GR_sh += GR_shx
                LCP = 11.98 - (11.98 - 10.10) * (self.grid[i][j].get_ShTol - 1)/(9 - 1)
                LCP_mean = (11.98 + 10.10)/2
                GR_cs = np.min(1, (1.333 * crownsize * LCP)/(self.grid[i][j].get_cmax() * LCP_mean))                
                GR_light = GR_cs * GR_sh

                # soil
                # GR_soil = 

                delta_D = delta_D_optimal * pow(GR_drought * GR_gdd * GR_light, 1/3)
                D = D + delta_D
                self.grid[i][j].set_diameter(D)

                H_new = self.grid[i][j].get_height()                
                t_slowgr = 0.0003
                if((H_new - H) < t_slowgr):
                    self.grid[i][j].increment_slowyear()
                else:
                    self.grid[i][j].reset_slowyear()
                self.grid[i][j].set_age()

    def __tree_spread_seed(self, weatherinfo):
        for i in range(self.x):
            for j in range(self.y):
                # judge whether there is a seed and the basic conditions
                if not self.grid[i][j].has_tree():
                    continue
                # P_Tw
                if (self.grid[i][j].get_wtmin >= weatherinfo['Tw']) or (self.grid[i][j].get_wtmax <= weatherinfo['Tw']):
                    continue
                # GDD
                if (self.grid[i][j].get_DDmin()> self.grid[i][j].get_GDD(weatherinfo)):
                    continue

                numseed = self.grid[i][j].get_numseed()
                for _ in range(numseed):
                    a = np.random.randint(self.x)
                    b = np.random.randint(self.y)
                    if(self.grid[a][b].has_tree()):
                        continue

                    # Dr
                    P_dr = 1 if(weatherinfo['DrI'] <= self.__get_DrTol_mean(a, b, self.grid[i][j].get_DrTol())) else 0

                    # la
                    P_la = 1 if(np.exp(-0.25 * self.shadow[a][b][0]) >= self.grid[i][j].get_ShTol_seedings) else 0

                    if(P_dr and P_la):
                        cest = np.random(low=0, high=1)
                        pest = np.random(low=0, high=1)
                        if (cest > pest):
                            self.grid[a][b] = cell(self, a, b, isRandom=False, treetype=self.grid[i][j].get_treetype())
        return self

    def __tree_death(self):
        for i in range(self.x):
            for j in range(self.y):
                if(self.grid[i][j].has_tree()):
                    P_dist = 0.005
                    k_death = 4.605
                    P_age = k_death/self.grid[i][j].get_Amax()
                    p_slowgr = 0.368
                    k_SlowGrTime = 3
                    SlowGr = self.grid[i][j].get_slowyears()

                    P_stressed = p_slowgr if(SlowGr >= k_SlowGrTime) else 0
                    P_g = P_age + (1 - P_age) * P_stressed
                    a = np.random(low=0, high=1)
                    b = np.random(low=0, high=1)
                    P_0 = 0 if(a<P_dist) else 1
                    P_mort = np.max(P_g, P_0)
                    if(P_mort > b):
                        self.grid[i][j].remove_tree()


    def __construct_shadow(self):
        # initialize the shadow variable in grid
        self.shadow = np.zeros([self.x, self.y, 50], dtype=np.float64)

        Y, X = np.meshgrid(range(self.y), range(self.x))
        for i in range(self.x):
            for j in range(self.y):
                if not self.grid[i][j].has_tree():
                    continue
                r = self.grid[i][j].get_width()
                H = int(self.grid[i][j].get_height())
                h = int(self.grid[i][j].get_height() - self.grid[i][j].get_cl())

                temp_X = np.abs(X - i)
                temp_Y = np.abs(Y - i)
                temp = -(temp_X + temp_Y - r)
                temp = np.exp(temp) * ((temp>-1).astype(np.int64))

                self.shadow[:, :, h:H] += temp[:, :, np.newaxis]
        self.shadow[self.shadow > 1] = 1

    def __tree_crownsize(self):
        count1 = 0
        count2 = 0
        for i, j in zip(np.tile(np.arange(self.x), self.y), np.arange(self.y).repeat(self.x)):
            for a, b in zip(np.tile(np.arange(self.x), self.y), np.arange(self.y).repeat(self.x)):
                if self.grid[a][b].hastree():
                    if self.grid[a][b].get_height() > self.grid[i][j].get_height():
                        count1 += 1
                    count2 += 1
            k_lai = pow((count1/count2), 2)
            crownsize = self.grid[i][j].get_cmax() * (1-k_lai) + self.grid[i][j].get_cmin() * k_lai

        self.grid[i][j].set_crownsize(crownsize)

    def __get_DrTol_mean(self, i: int, j: int, DrTol: float):
        w_0 = 48
        w_1 = 4
        w_2 = 1
        total = w_0
        current = DrTol * w_0
        if self.grid[i-2][j].has_tree() and self.grid[i-2][j].get_age() >= 3:
            current += self.grid[i-2][j].get_DrTol() * w_2
            total += w_2
        if self.grid[i-1][j-1].has_tree() and self.grid[i-1][j-1].get_age() >= 3:
            current += self.grid[i-1][j-1].get_DrTol() * w_2
            total += w_2
        if self.grid[i][j-2].has_tree() and self.grid[i][j-2].get_age() >= 3:
            current += self.grid[i][j-2].get_DrTol() * w_2
            total += w_2    
        if self.grid[i-1][j+1].has_tree() and self.grid[i-1][j+1].get_age() >= 3:
            current += self.grid[i-1][j+1].get_DrTol() * w_2
            total += w_2
        if self.grid[i][j+2].has_tree() and self.grid[i][j+2].get_age() >= 3:
            current += self.grid[i][j+2].get_DrTol() * w_2
            total += w_2
        if self.grid[i+1][j+1].has_tree() and self.grid[i+1][j+1].get_age() >= 3:
            current += self.grid[i+1][j+1].get_DrTol() * w_2
            total += w_2
        if self.grid[i+2][j].has_tree() and self.grid[i+2][j].get_age() >= 3:
            current += self.grid[i+2][j].get_DrTol() * w_2
            total += w_2    
        if self.grid[i+1][j-1].has_tree() and self.grid[i+1][j-1].get_age() >= 3:
            current += self.grid[i+1][j-1].get_DrTol() * w_2
            total += w_2
        if self.grid[i+1][j].has_tree() and self.grid[i+1][j].get_age() >= 3:
            current += self.grid[i+1][j].get_DrTol() * w_1
            total += w_1
        if self.grid[i-1][j].has_tree() and self.grid[i-1][j].get_age() >= 3:
            current += self.grid[i-1][j].get_DrTol() * w_1
            total += w_1
        if self.grid[i][j+1].has_tree() and self.grid[i][j+1].get_age() >= 3:
            current += self.grid[i][j+1].get_DrTol() * w_1
            total += w_1
        if self.grid[i][j-1].has_tree() and self.grid[i][j-1].get_age() >= 3:
            current += self.grid[i][j-1].get_DrTol() * w_1
            total += w_1
        return (current/total)

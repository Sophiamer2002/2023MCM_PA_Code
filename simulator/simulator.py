from .cell import cell
import json
import numpy as np
import pandas as pd
import numpy.random as rd


class simulator:
    '''
    member--x:          模拟方格的x轴个数，需要大于等于10
    member--y:          模拟方格的y轴个数，需要大于等于10
    member--treeinfo:   树木的具体信息
    member--treetypes:  参与模拟的树木种类（一个字符串数组）
    member--numtreetype:总共参与模拟的树木个数
    '''
    def __init__(self, x, y, treetypes, treeinfofile='./data/tree.json'):
        self.x, self.y = x, y
        if x < 10 or y < 10:
            print("x or y is too small")
            raise ValueError

        # specify the tree types and their information
        with open(treeinfofile, 'r') as file:
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

        self.records = {
            # the following are recorded by function self.__record()
            "treetype": [],
            "treeheight": [],
            "numspecies": [],

            # the following are recorded by function self.__tree_spread_seed()
            "num_seeds": [],
            "seed_success_rate": []
        }

        # Set the probability of death due to pollution
        self.PollutionIndex = 0.05

        self.__construct_shadow()

        # self.__warm_up()
        # done by user

    # user interface
    def update(self, weatherinfo):
        '''
        arg--weatherinfo:   一个字典，记录了一年之内的天气信息
                    格式:   {
                        "DrI": 0.4,
                        "GDD_E": ?,
                        "GDD_D": ?,
                        "Tw": ?,
                    }
        '''
        self.__update_one_year(weatherinfo)

    def warm_up(self, weatherinfos: pd.DataFrame):
        '''
        arg--weatherinfos:  pandas.DataFrame，记录了若干年的天气信息
        '''
        for i in weatherinfos.index:
            self.__update_one_year(weatherinfos.loc[i], record=False)  # TODO

    # private functions starting here
    def __update_one_year(self, weatherinfo, record=True):
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
        self.__tree_crownsize(weatherinfo)

        # 计算树木生长，只更新diameter
        self.__tree_grow(weatherinfo)

        # # 计算树木传播种子
        self.__tree_spread_seed(weatherinfo, record=record)

        # # 计算树木死亡
        self.__tree_death()

        # 更新shadow信息
        self.__construct_shadow()

        if record:
            self.__record()

        return self

    def __tree_grow(self, weatherinfo):
        for i in range(self.x):
            for j in range(self.y):
                if not self.grid[i][j].has_tree():
                    continue
                Hmax = self.grid[i][j].get_maxH()
                bs = Hmax - 137
                ss = self.grid[i][j].get_s()
                cs = ss / bs
                D = self.grid[i][j].get_diameter()
                H = self.grid[i][j].get_height()
                h = H - self.grid[i][j].get_cl()
                gs_prime = 2 * 1.37 * self.grid[i][j].get_gs()/(Hmax + 1.37)

                # a random number between 0.8 and 1.2
                rand = rd.random() * 0.4 + 0.8
                # TODO may be done
                delta_D_optimal = gs_prime * D * (1 - H/Hmax)/(2 * 1.37 + 1.9*ss*D - 2*((ss*D)**2)/(5 * bs))

                # gdd
                GDD = self.grid[i][j].get_GDD(weatherinfo)
                GR_gdd = max(0, 1 - np.exp((self.grid[i][j].get_DDmin() - GDD)/750))

                # drought
                DrTol_mean = self.__get_DrTol_mean(i, j, self.grid[i][j].get_DrTol())
                GR_drought = np.sqrt(max(0, 1. - weatherinfo['DrI']/DrTol_mean))

                # light
                crownsize = self.grid[i][j].get_crownsize()
                GR_sh = 0
                for x in range(int(h), int(H) + 1):
                    # This result is according to paper Botkin et al. 1972
                    Lmax = 1 - np.exp(-4.64 * (1 - 1.3 * self.shadow[i][j][x] - 0.05))
                    Lmin = 2.24 * (1 - np.exp(-1.136 * (1 - 1.3 * self.shadow[i][j][x] - 0.08)))

                    GR_shx = max(0, Lmax + (Lmin - Lmax) * (self.grid[i][j].get_ShTol()-1)/(9 - 1))
                    GR_sh += 1/(H - h) * GR_shx

                LCP = 11.98 - (11.98 - 10.10) * (self.grid[i][j].get_ShTol() - 1)/(9 - 1)
                LCP_mean = (11.98 + 10.10)/2
                GR_cs = min(1, (1.333 * crownsize * LCP)/(self.grid[i][j].get_cmax() * LCP_mean))                
                GR_light = GR_cs * GR_sh

                # soil
                # GR_soil = 

                delta_D = delta_D_optimal * pow(GR_drought * GR_gdd * GR_light, 1/3) * rand
                D = D + delta_D
                self.grid[i][j].set_diameter(D)

                H_new = self.grid[i][j].get_height()                
                t_slowgr = 0.0003
                if((H_new - H) < t_slowgr):
                    self.grid[i][j].increment_slowyear()
                else:
                    self.grid[i][j].reset_slowyear()
                self.grid[i][j].set_age()

    def __tree_spread_seed(self, weatherinfo, record):
        Tw = weatherinfo['Tw']
        # record information
        total_seed = {species: 0 for species in self.treetypes}
        success_seed = {species: 0 for species in self.treetypes}

        for i in range(self.x):
            for j in range(self.y):
                # judge whether there is a seed and the basic conditions
                if not self.grid[i][j].has_tree():
                    continue

                # P_Tw
                # if (self.grid[i][j].get_wtmin() >= weatherinfo['Tw']) or (self.grid[i][j].get_wtmax() <= weatherinfo['Tw']):
                #     continue

                # GDD
                if (self.grid[i][j].get_DDmin()> self.grid[i][j].get_GDD(weatherinfo)):
                    continue

                numseed = self.grid[i][j].get_numseed()
                for _ in range(numseed):
                    a = rd.randint(self.x)
                    b = rd.randint(self.y)
                    if(self.grid[a][b].has_tree()):
                        continue
                    
                    total_seed[self.grid[i][j].get_treetype()] += 1
                    # consider the temperature impact here
                    # P_Tw
                    wtmin = self.grid[i][j].get_wtmin()
                    wtmax = self.grid[i][j].get_wtmax()
                    if((wtmin >= weatherinfo['Tw']) or (wtmax <= weatherinfo['Tw'])):
                        weight = 0
                        for u, v in zip(np.repeat(np.arange(-2, 3), 5), np.tile(np.arange(-2, 3), 5)):
                            distance = np.abs(u) + np.abs(v)
                            if (distance > 2) or (u == 0 and v == 0):
                                continue
                            u, v = (u+a)%self.x, (v+b)%self.y
                            if(not self.grid[u][v].has_tree()):
                                continue
                            if(self.grid[u][v].get_treetype() == self.grid[i][j].get_treetype()):
                                continue
                            weight += 0.08 if distance == 1 else 0.015
                        nowTw = weight * (wtmax + wtmin)/2 + (1 - weight) * Tw
                        if((wtmin >= nowTw) or (wtmax <= nowTw)):
                            continue


                    # Dr
                    P_dr = 1 if(weatherinfo['DrI'] <= self.__get_DrTol_mean(a, b, self.grid[i][j].get_DrTol())) else 0

                    # la
                    # P_la = 1 if(np.exp(-0.25 * self.shadow[a][b][0]) >= self.grid[i][j].get_ShTol_seedings()) else 0
                    P_la = 1 if(self.shadow[a][b][0] <= self.grid[i][j].get_ShTol_seedings()) else 0

                    if(P_dr and P_la):
                        if (rd.random() > rd.random()):
                            self.grid[a][b] = cell(self, a, b, isRandom=False, treetype=self.grid[i][j].get_treetype())
                            success_seed[self.grid[i][j].get_treetype()] += 1
            
        if record:
            self.records['num_seeds'].append(total_seed)
            success_seed_rate = {
                species: success_seed[species]/total_seed[species] if total_seed[species] != 0 else 0 
                for species in self.treetypes
            }
            self.records['seed_success_rate'].append(success_seed_rate)

        print('total_seed: ', total_seed)
        print('success_seed: ', success_seed)


    def __tree_death(self):
        for i in range(self.x):
            for j in range(self.y):
                if self.grid[i][j].has_tree():
                    P_dist = 0.005
                    k_death = 4.605
                    P_age = k_death/self.grid[i][j].get_Amax()
                    p_slowgr = 0.368
                    k_SlowGrTime = 3
                    SlowGr = self.grid[i][j].get_slowyears()

                    # P_stressed = p_slowgr if(SlowGr >= k_SlowGrTime) else 0
                    k = 2
                    P_stressed = min(1, (np.exp(k * SlowGr) - 1)/(np.exp(k * k_SlowGrTime) - 1)) * p_slowgr
                    P_g = P_age + (1 - P_age) * P_stressed
                    
                    P_0 = 0 if(rd.random() > P_dist + self.Pollution_index) else 1
                    P_mort = max(P_g, P_0)
                    if(P_mort > rd.random()):
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
                H = int(self.grid[i][j].get_height() - 0.1 * self.grid[i][j].get_cl())
                h = int(self.grid[i][j].get_height() - self.grid[i][j].get_cl())

                temp_X = np.abs(X - i)
                temp_Y = np.abs(Y - j)
                temp = -(temp_X + temp_Y - r)
                temp = np.exp(temp) * ((temp>-1).astype(np.int64))

                # self.shadow[:, :, h:H] += temp[:, :, np.newaxis]
                # 为了防止阴影过度融合，将阴影的增量减小
                self.shadow[:, :, h:H] += 1/500 * temp[:, :, np.newaxis]

        # 将阴影从上到下累加
        for i in range(48, -1, -1):
            self.shadow[:, :, i] += self.shadow[:, :, i+1]
        self.shadow[self.shadow > 1] = 1

    def __tree_crownsize(self, weatherinfo):
        DrI = weatherinfo['DrI']
        for i, j in zip(np.tile(np.arange(self.x), self.y), np.arange(self.y).repeat(self.x)):
            if not self.grid[i][j].has_tree():
                continue
            count1, count2 = 0, 0
            DrTol = self.grid[i][j].get_DrTol()

            for a, b in zip(np.repeat(np.arange(-2, 3), 5), np.tile(np.arange(-2, 3), 5)):
                if(np.abs(a) + np.abs(b) > 2):
                    continue
                weight = 1 if(np.abs(a) + np.abs(b) <= 1) else 0.5
                a, b = (i+a)%self.x, (j+b)%self.y
                if not self.grid[a][b].has_tree():
                    continue
                if self.grid[a][b].get_height() > self.grid[i][j].get_height():
                    count1 += weight
                count2 += weight
            
            # calculating the impact of drought 
            proportion = min(1, (DrI/self.__get_DrTol_mean(i, j, DrTol)) ** 2)
            k_lai = pow((count1/count2), 2) * (1 - proportion) + proportion
            crownsize = self.grid[i][j].get_cmax() * (1-k_lai) + self.grid[i][j].get_cmin() * k_lai
            self.grid[i][j].set_crownsize(crownsize)

    def __record(self):
        
        # record tree type
        self.records['treetype'].append(
            [[(Acell.get_treetype() if Acell.has_tree() else 'No tree') for Acell in Arow] for Arow in self.grid]
        )

        # record tree height
        self.records['treeheight'].append(
            [[(Acell.get_height() if Acell.has_tree() else 'No tree') for Acell in Arow] for Arow in self.grid]
        )

        # record tree species number
        numspecies = {species: 0 for species in self.treetypes}
        for i in range(self.x):
            for j in range(self.y):
                if self.grid[i][j].has_tree():
                    numspecies[self.grid[i][j].get_treetype()] += 1

        numspecies['No tree'] = self.x * self.y - sum(numspecies.values())
        self.records['numspecies'].append(numspecies)



    def __get_DrTol_mean(self, i: int, j: int, DrTol: float):
        '''
        args:
            i: the x coordinate of the cell, between 0 and self.x-1
            j: the y coordinate of the cell, between 0 and self.y-1
            DrTol: the DrTol of the tree
        '''
        w_0 = 24
        w_1 = 4
        empty_1 = 2
        w_2 = 1
        empty_2 = 0.25
        total = w_0
        current = DrTol * w_0

        i = (i - self.x) if (i >= 0.5 * self.x) else i
        j = (j - self.y) if (j >= 0.5 * self.y) else j
        
        if self.grid[i-2][j].has_tree() and (self.grid[i-2][j].get_age() >= 3):
            current += self.grid[i-2][j].get_DrTol() * w_2
            total += w_2
        else:
            total += empty_2
        if self.grid[i-1][j-1].has_tree() and (self.grid[i-1][j-1].get_age() >= 3):
            current += self.grid[i-1][j-1].get_DrTol() * w_2
            total += w_2
        else:
            total += empty_2
        if self.grid[i][j-2].has_tree() and (self.grid[i][j-2].get_age() >= 3):
            current += self.grid[i][j-2].get_DrTol() * w_2
            total += w_2
        else:
            total += empty_2    
        if self.grid[i-1][j+1].has_tree() and (self.grid[i-1][j+1].get_age() >= 3):
            current += self.grid[i-1][j+1].get_DrTol() * w_2
            total += w_2
        else:
            total += empty_2
        if self.grid[i][j+2].has_tree() and (self.grid[i][j+2].get_age() >= 3):
            current += self.grid[i][j+2].get_DrTol() * w_2
            total += w_2
        else:
            total += empty_2
        if self.grid[i+1][j+1].has_tree() and (self.grid[i+1][j+1].get_age() >= 3):
            current += self.grid[i+1][j+1].get_DrTol() * w_2
            total += w_2
        else:
            total += empty_2
        if self.grid[i+2][j].has_tree() and (self.grid[i+2][j].get_age() >= 3):
            current += self.grid[i+2][j].get_DrTol() * w_2
            total += w_2
        else:
            total += empty_2    
        if self.grid[i+1][j-1].has_tree() and (self.grid[i+1][j-1].get_age() >= 3):
            current += self.grid[i+1][j-1].get_DrTol() * w_2
            total += w_2
        else:
            total += empty_2
        if self.grid[i+1][j].has_tree() and (self.grid[i+1][j].get_age() >= 3):
            current += self.grid[i+1][j].get_DrTol() * w_1
            total += w_1
        else:
            total += empty_1
        if self.grid[i-1][j].has_tree() and (self.grid[i-1][j].get_age() >= 3):
            current += self.grid[i-1][j].get_DrTol() * w_1
            total += w_1
        else:
            total += empty_1
        if self.grid[i][j+1].has_tree() and (self.grid[i][j+1].get_age() >= 3):
            current += self.grid[i][j+1].get_DrTol() * w_1
            total += w_1
        else:
            total += empty_1
        if self.grid[i][j-1].has_tree() and (self.grid[i][j-1].get_age() >= 3):
            current += self.grid[i][j-1].get_DrTol() * w_1
            total += w_1
        else:
            total += empty_1
        return (current/total)

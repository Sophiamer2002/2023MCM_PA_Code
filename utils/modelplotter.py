import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
from simulator import simulator


class modelplotter:
    def __init__(self, simulator: simulator):

        # set font to Times New Roman, math font to stix
        rcParams['font.family'] = 'Times New Roman'
        rcParams['mathtext.fontset'] = 'stix'

        self.simulator = simulator
        self.treeindex = {treetype: (i+1) for i, treetype in enumerate(self.simulator.treeinfo.keys())}
        self.treeindex['No tree'] = 0

        # green colors and brown for none
        colors = ['#af610f', '#98f398', '#0a800a', '#1d641d', '#5d935d', '#cee7ce', '#00FFFF', '#00CCFF', '#0099FF', '#0066FF', '#0033FF', '#0000FF']
        self.tree_color = [ colors[i] for i in range(len(self.treeindex)) ]
        self.tree_cmap = plt.cm.colors.ListedColormap(self.tree_color)

        # create my cmap, using treeindex and ListedColormap

    def plot_trees(self, save=False, filename=None, title=None, showlegend=True):
        # plot trees
        _, ax = plt.subplots()
        ax.set_title(title)

        treenames = [[self.treeindex[cell.get_treetype()] if cell.has_tree() else 0 for cell in row] for row in self.simulator.grid]
        ax.imshow(treenames, cmap=self.tree_cmap, vmin=0, vmax=len(self.treeindex)-1, norm='linear')

        # legend for the tree types
        if showlegend:
            legend_elements = [
                plt.Line2D([0], [0], marker='s', color='w', 
                           label=treetype, 
                           markerfacecolor=self.tree_cmap(self.treeindex[treetype]), 
                           markersize=15) 
                for treetype in self.simulator.treetypes + ['No tree']]
            ax.legend(handles=legend_elements, loc='upper right')

        if save:
            plt.savefig(filename)
        else:
            plt.show()

    def plot_trees_by_year(self, year, save=False, filename=None, title=None, showlegend=True):
        # plot trees
        _, ax = plt.subplots()
        ax.set_title(title)

        treenames = [
            [self.treeindex[cell] for cell in row] 
            for row in (self.simulator.records["treetype"])[year]
        ]
        ax.imshow(treenames, cmap=self.tree_cmap, vmin=0, vmax=len(self.treeindex)-1, norm='linear')

        # legend for the tree types
        if showlegend:
            legend_elements = [plt.Line2D([0], [0], marker='s', color='w', 
                                        label=treetype, 
                                        markerfacecolor=self.tree_cmap(self.treeindex[treetype]), 
                                        markersize=15) for treetype in self.simulator.treetypes + ['No tree']]
            ax.legend(handles=legend_elements, loc='upper right')

        if save:
            plt.savefig(filename)
        else:
            plt.show()

    def plot_tree_num(self, title=None, save=False, filename=None, figuresize=(10, 5)):
        # plot number of treespecies every year
        df = pd.DataFrame(self.simulator.records["numspecies"])
        df.plot.area(
            color=[self.tree_color[self.treeindex[treetype]] for treetype in df.columns],
        )
        plt.title(title)
        if save:
            plt.savefig(filename)
        else:
            plt.show()

    def plot_seed_success_rate(self, title=None, save=False, filename=None, figuresize=(10, 5)):
        # plot seed success every year
        df = pd.DataFrame(self.simulator.records["seed_success_rate"])
        df.plot(
            color=[self.tree_color[self.treeindex[treetype]] for treetype in df.columns],
        )
        plt.title(title)
        if save:
            plt.savefig(filename)
        else:
            plt.show()

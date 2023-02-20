import matplotlib.pyplot as plt
from ..simulator.simulator import simulator


class modelplotter:
    def __init__(self, simulator: simulator):
        self.simulator = simulator
        self.treeindex = {treetype: (i+1) for i, treetype in enumerate(self.simulator.treeinfo.keys())}
        self.treeindex['No tree'] = 0

        # green colors and brown for none
        colors = ['#af610f', '#98f398', '#0a800a', '#1d641d', '#5d935d', '#cee7ce', '#00FFFF', '#00CCFF', '#0099FF', '#0066FF', '#0033FF', '#0000FF']
        self.tree_color = [colors[i] for i in range(len(self.treeindex)) ]
        self.tree_cmap = plt.cm.colors.ListedColormap(self.tree_color)

        # create my cmap, using treeindex and ListedColormap

    def plot_trees(self, save=False, filename=None, title=None):
        # plot trees
        _, ax = plt.subplots()
        ax.set_title(title)

        treenames = [[self.treeindex[cell.get_treetype()] if cell.has_tree() else 0 for cell in row] for row in self.simulator.grid]
        ax.imshow(treenames, cmap=self.tree_cmap, vmin=0, vmax=len(self.treeindex)-1, norm='linear')

        # legend for the tree types
        legend_elements = [plt.Line2D([0], [0], marker='s', color='w', 
                                      label=treetype, 
                                      markerfacecolor=self.tree_cmap(self.treeindex[treetype]), 
                                      markersize=15) for treetype in self.treeindex.keys()]
        ax.legend(handles=legend_elements, loc='upper left')

        if save:
            plt.savefig(filename)
        else:
            plt.show()


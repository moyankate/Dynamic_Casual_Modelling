import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class PlotGraph():
    def __init__(self, names, A):
        super().__init__()
        self.names = names
        self.A = A
        
    def creat_list(self):
        list_names = []
        i,j = np.where(self.A)
        for ite in range(len(i)):
            list_names.append((self.names[i[ite]], self.names[j[ite]]))
        return list_names
        
    def display(self):
        G = nx.DiGraph()
        G.add_edges_from(self.creat_list())
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='mediumspringgreen')
        nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black', arrowsize=30)
        nx.draw_networkx_labels(G, pos)
        plt.show()

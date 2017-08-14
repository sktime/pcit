import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def get_data_round(train, test, i):
    x = np.delete(train, i, axis=1)
    y = train[:, i]
    x_test = np.delete(test, i, axis=1)
    y_test = test[:, i]
    return x, y, x_test, y_test

def shift_data(x, y, x_test, y_test):
    x = np.delete(x, 0, axis=0)
    y = np.delete(y, y.shape[0] - 1, axis=0)
    x_test = np.delete(x_test, 0, axis=0)
    y_test = np.delete(y_test, y_test.shape[0] - 1, axis=0)
    return x, y, x_test, y_test

def log_loss_resid(estimator, predictions, y, classes, baseline = False):

    ## Add label binarizer transform

    new = np.array(())
    for i in np.unique(y):
        if i not in classes:
            new = np.append(new, i)

    classes = np.append(classes, new)
    new_probas = np.zeros(len(new))

    n = len(y)

    if not baseline:
        zero_mat = np.reshape(np.ones(n * len(new)), newshape = (n, len(new)))
        predictions = np.concatenate((predictions, zero_mat), axis = 1)

    resid = np.ones(n)

    for i in range(n):
        resid[i] = np.where(y[i] == classes)[0]
        if not baseline:
            resid[i] = predictions[i, resid[i].astype(int)]

    if baseline:
        predictions = np.append(estimator.class_prior_, new_probas)
        resid  = -np.log(np.clip(predictions[resid.astype(int)],1e-15, 1 - 1e-15))
    else:
        resid  = -np.log(np.clip(resid, 1e-15, 1 - 1e-15))

    return resid


class descendants():
    def __init__(self, skeleton):
        self.skeleton = skeleton
        self.desc = list()

    def dir_desc(self, i):
        n = self.skeleton.shape[1]
        self.desc.extend([x for x in range(n) if (self.skeleton[i, x] == 2) and (x not in self.desc)])
        return self.desc

    def all_desc(self, i):
        self.dir_desc(i)
        old_len = -1
        new_len = 0
        while old_len < new_len:
            old_len = new_len
            for q in self.desc:
                self.dir_desc(q)
            new_len = len(self.desc)
        return self.desc

    def undir_neighb(self, i):
        n = self.skeleton.shape[1]
        neighbours = [x for x in range(n) if self.skeleton[i, x] == 1]
        return neighbours


def draw_graph(skeleton, feature_names = None, title_graph = None):
    G = nx.from_numpy_matrix(skeleton)
    if feature_names is not None:
        labels = {}
        for i in range(len(feature_names)):
            labels.update({i: feature_names[i]})
        nx.relabel_nodes(G, labels, copy=False)

    graph_pos = nx.spring_layout(G)
    graph_labels = [x for x in graph_pos]
    graph_pos_up = dict()
    for i in range(len(graph_labels)):
        new_position = graph_pos[graph_labels[i]] + np.array((0,0.05))
        graph_pos_up[graph_labels[i]] = new_position


    nx.draw_networkx_nodes(G, graph_pos, node_size=1000, node_color='blue', alpha=0.3)
    nx.draw_networkx_edges(G, graph_pos, alpha=0.5, font_color = 'blue', style = 'dotted', width = 2)
    nx.draw_networkx_labels(G, graph_pos_up, font_size=15, font_family='sans-serif')
    plt.title(title_graph)

    plt.show()

def draw_graph_edgelabel(skeleton, feature_names = None):
    B = skeleton[0,0]
    G = nx.from_numpy_matrix(skeleton)

    if feature_names is not None:
        labels = {}
        for i in range(len(feature_names)):
            labels.update({i: feature_names[i]})
        nx.relabel_nodes(G, labels, copy=False)

    graph_pos = nx.shell_layout(G)
    graph_labels = [x for x in graph_pos]
    graph_pos_up = dict()

    for i in range(len(graph_labels)):
        new_position = graph_pos[graph_labels[i]] + np.array((0,0.05))
        graph_pos_up[graph_labels[i]] = new_position

    edges = G.edges()
    idx1 = [x for x in edges]

    for i in idx1:
        weight_round = G[i[0]][i[1]]['weight']
        if weight_round <= 7:
            G[i[0]][i[1]]['color'] = 'g'
        elif weight_round > (1 / 3 * B):
            G[i[0]][i[1]]['color'] = 'b'
        else:
            G[i[0]][i[1]]['color'] = 'r'

    graph_pos = nx.spring_layout(G)

    colors = [G[u][v]['color'] for u, v in edges]

    nx.draw_networkx_nodes(G, graph_pos, node_size=1000, node_color='blue', alpha=0.3)
    nx.draw_networkx_edges(G, graph_pos, alpha=0.5, edge_color=colors, style = 'dotted', width = 4)
    nx.draw_networkx_labels(G, graph_pos, font_size=15, font_family='sans-serif')
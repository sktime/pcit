import numpy as np
# import matplotlib.pyplot as plt
# import networkx as nx

def log_loss_resid(estimator, predictions, y, classes, baseline = False):
    '''
    This function calculates the log loss residuals
    ------------------------------
    Attributes:
        - estimator: fitted classifier
        - predictions: predictions (used when loss residuals are not for baseline)
        - y: observations
        - classes: training set classes
        - baseline: Flag. If true, uninformed baseline loss residuals are calculated
    '''
    
    new = np.array(())                  # Add classes that are in test set, but not in training set
    for i in np.unique(y):              
        if i not in classes:
            new = np.append(new, i)

    classes = np.append(classes, new)
    new_probas = np.zeros(len(new))

    n = len(y)

    if not baseline:                    # Add 0-predictions for the missing classes to prediction array
        zero_mat = np.reshape(np.ones(n * len(new)), newshape = (n, len(new)))
        predictions = np.concatenate((predictions, zero_mat), axis = 1)

    resid = np.ones(n)

    for i in range(n):                  # Extract predicted probabilities for true class (for log loss)
        resid[i] = np.where(y[i] == classes)[0]
        if not baseline:
            resid[i] = predictions[i, resid[i].astype(int)]

    if baseline:
        predictions = np.append(estimator.class_prior_, new_probas)
        resid  = -np.log(np.clip(predictions[resid.astype(int)],1e-15, 1 - 1e-15))
    else:
        resid  = -np.log(np.clip(resid, 1e-15, 1 - 1e-15))

    # The clipping is done to ensure finite values for the loss

    return resid

# The following functions can be used to visualize Graphs, however the require pyplot and networkx

# def draw_graph(skeleton, feature_names = None, title_graph = None):
#     '''Draw graphs from skeletons'''
#     G = nx.from_numpy_matrix(skeleton)
#     if feature_names is not None:
#         labels = {}
#         for i in range(len(feature_names)):
#             labels.update({i: feature_names[i]})
#         nx.relabel_nodes(G, labels, copy=False)

#     graph_pos = nx.spring_layout(G)
#     graph_labels = [x for x in graph_pos]
#     graph_pos_up = dict()
#     for i in range(len(graph_labels)):
#         new_position = graph_pos[graph_labels[i]] + np.array((0,0.05))
#         graph_pos_up[graph_labels[i]] = new_position

#     nx.draw_networkx_nodes(G, graph_pos, node_size=1000, node_color='blue', alpha=0.3)
#     nx.draw_networkx_edges(G, graph_pos, alpha=0.5, font_color = 'blue', style = 'dotted', width = 2)
#     nx.draw_networkx_labels(G, graph_pos_up, font_size=15, font_family='sans-serif')
#     plt.title(title_graph)

#     plt.show()

# def draw_graph_edgelabel(skeleton, feature_names = None):
#     B = skeleton[0,0]
#     G = nx.from_numpy_matrix(skeleton)

#     if feature_names is not None:
#         labels = {}
#         for i in range(len(feature_names)):
#             labels.update({i: feature_names[i]})
#         nx.relabel_nodes(G, labels, copy=False)

#     graph_pos = nx.shell_layout(G)
#     graph_labels = [x for x in graph_pos]
#     graph_pos_up = dict()

#     for i in range(len(graph_labels)):
#         new_position = graph_pos[graph_labels[i]] + np.array((0,0.05))
#         graph_pos_up[graph_labels[i]] = new_position

#     edges = G.edges()
#     idx1 = [x for x in edges]

#     for i in idx1:
#         weight_round = G[i[0]][i[1]]['weight']
#         if weight_round <= 7:
#             G[i[0]][i[1]]['color'] = 'g'
#         elif weight_round > (1 / 3 * B):
#             G[i[0]][i[1]]['color'] = 'b'
#         else:
#             G[i[0]][i[1]]['color'] = 'r'

#     graph_pos = nx.spring_layout(G)

#     colors = [G[u][v]['color'] for u, v in edges]

#     nx.draw_networkx_nodes(G, graph_pos, node_size=1000, node_color='blue', alpha=0.3)
#     nx.draw_networkx_edges(G, graph_pos, alpha=0.5, edge_color=colors, style = 'dotted', width = 4)
#     nx.draw_networkx_labels(G, graph_pos, font_size=15, font_family='sans-serif')
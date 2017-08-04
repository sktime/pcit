import numpy as np
from compare import FDRcontrol, pred_indep
import networkx as nx
import matplotlib.pyplot as plt


def find_neighbours(X, estimators = None, confidence = 0.05, feature_names = None, method = None):
    p = X.shape[1]
    skeleton = np.reshape(np.zeros(p**2), (p,p))

    for i in range(p-1):
        for j in range(i + 1, p):
            input_var = np.reshape(X[:,i], (-1,1))
            output_var = np.reshape(X[:,j], (-1,1))
            conditioning_set = np.delete(X, (i,j), 1)
            p_values_adj, which_predictable, independent, ci = pred_indep(output_var, input_var, z = conditioning_set,
                                                    confidence= confidence, estimators = estimators, method = method)
            skeleton[j,i] = np.min(p_values_adj)
            skeleton[i,j] = skeleton[j,i]

    skeleton_adj = (FDRcontrol(skeleton, confidence)[0] < confidence) * 1
    G = nx.from_numpy_matrix(skeleton_adj)
    if feature_names is not None:
        labels = {}
        for i in range(len(feature_names)):
            labels.update({i: feature_names[i]})
        nx.relabel_nodes(G, labels, copy=False)
    nx.draw_networkx(G)

    return skeleton, skeleton_adj

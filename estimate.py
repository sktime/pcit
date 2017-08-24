import numpy as np
from compare import FDRcontrol, pred_indep
from combine import MetaEstimator
import networkx as nx
import matplotlib.pyplot as plt


def find_neighbours(X, estimator = MetaEstimator(), confidence = 0.05):
    p = X.shape[1]
    skeleton = np.reshape(np.zeros(p**2), (p,p))

    for i in range(p-1):
        for j in range(i + 1, p):
            input_var = np.reshape(X[:,i], (-1,1))
            output_var = np.reshape(X[:,j], (-1,1))
            conditioning_set = np.delete(X, (i,j), 1)
            p_values_adj, which_predictable, independent, ci = pred_indep(output_var, input_var, z = conditioning_set,
                                                    confidence = confidence, estimator = estimator)
            skeleton[j,i] = np.min(p_values_adj)
            skeleton[i,j] = skeleton[j,i]
    skeleton_adj = (FDRcontrol(skeleton, confidence)[0] < confidence) * 1


    return skeleton, skeleton_adj

def mutual_independence(X, Z = None, estimators = None, method = None, confidence = 0.05):
    p = X.shape[1]

    p_values = np.ones(p)
    for i in range(p):
        p_values_adj, which_predictable, independent, ci = pred_indep(X[:,i:(i+1)],
                                np.delete(X,i,1), z=Z, confidence=confidence, estimators=estimators, method=method)
        p_values[i] = independent[1]

    p_values_adj = FDRcontrol(p_values)[0]

    independent = (all(p_values_adj > confidence), np.min(p_values_adj))

    return p_values_adj, independent
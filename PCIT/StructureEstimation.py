import numpy as np
from IndependenceTest import FDRcontrol, pred_indep

from PCIT.MetaEstimator import MetaEstimator


def find_neighbours(X, estimator = MetaEstimator(), confidence = 0.05):
    '''
    Undirected graph skeleton learning routine.
    ----------------
    Attributes:
        - X: data set for undirected graph estimation
        - estimator: object of the MetaEstimator class
        - confidence: false-discovery rate level

    Returns:
        - skeleton: Matrix (graph) with entries being the p-values for each individual test
        - skeleton_adj: Matrix (graph) with skeleton, after application of FDR control
    '''

    p = X.shape[1]
    skeleton = np.reshape(np.zeros(p**2), (p,p))

    # Loop over all subsets of X of size 2
    for i in range(p-1):
        for j in range(i + 1, p):

            input_var = np.reshape(X[:,i], (-1,1))
            output_var = np.reshape(X[:,j], (-1,1))
            conditioning_set = np.delete(X, (i,j), 1)

            # Conditional independence test conditional on all other variables
            p_values_adj, which_predictable, independent, ci = pred_indep(output_var, input_var,
                        z = conditioning_set, confidence = confidence, estimator = estimator)

            # P-value of null-hypothesis that pair is independent give all other variables
            skeleton[j,i] = independent[1]

            # Ensure symmetry
            skeleton[i,j] = skeleton[j,i]

    # Apply FDR control and make hard assignments if independent or not according to confidence level
    skeleton_adj = (FDRcontrol(skeleton, confidence)[0] < confidence) * 1

    return skeleton, skeleton_adj

def mutual_independence(X, Z = None, estimators = None, method = None, confidence = 0.05):
    '''
    Unfinished (don't use)
    Test for mutual independence in a set of variables, by testing, if there is one variable in X,
    that is not independent of all other variables, given Z.
    '''
    p = X.shape[1]

    p_values = np.ones(p)
    for i in range(p):
        p_values_adj, which_predictable, independent, ci = pred_indep(X[:,i:(i+1)],
                                np.delete(X,i,1), z=Z, confidence=confidence, estimators=estimators, method=method)
        p_values[i] = independent[1]

    p_values_adj = FDRcontrol(p_values)[0]

    independent = (all(p_values_adj > confidence), np.min(p_values_adj))

    return p_values_adj, independent
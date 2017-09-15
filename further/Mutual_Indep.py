def mutual_independence(X, Z = None, estimators = None, method = None, confidence = 0.05):
    '''
    Unfinished (don't use)
    Test for mutual independence in a set of variables, by testing, if there is one variable in X,
    that is not independent of all other variables, given Z.
    '''
    p = X.shape[1]

    p_values = np.ones(p)
    for i in range(p):
        p_values_adj, independent, ci = PCIT(X[:,i:(i+1)],
                                np.delete(X,i,1), z=Z, confidence=confidence, estimators=estimators, method=method)
        p_values[i] = independent[1]

    p_values_adj = FDRcontrol(p_values)[0]

    independent = (all(p_values_adj > confidence), np.min(p_values_adj))

    return p_values_adj, independent
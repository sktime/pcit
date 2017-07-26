import numpy as np
from scipy import stats
from sklearn import linear_model, ensemble,utils, preprocessing

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
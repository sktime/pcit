from sklearn.datasets import load_boston, load_iris
from sklearn import linear_model
from sklearn import tree, naive_bayes, ensemble, svm
from sklearn import dummy, preprocessing
from sklearn import metrics
import importlib
import support, combine, estimate, compare
import numpy as np
from scipy import stats
import time
from sklearn.multioutput import MultiOutputEstimator
import networkx as nx
import matplotlib.pyplot as plt
import pc_algorithm

## bost, iris, data, stock, synth
which = 'synth'

if which == 'bost':
    X = load_boston()['data']
    y = np.reshape(load_boston()['target'],(-1,1))
    X = np.concatenate((X,y),axis = 1)
    feature_names = ['crime', 'land_zoned', '%industry', 'riverdummy', 'nitricox', 'rooms/dwelling', 'age', 'distemploycent', 'accesshighway', 'proptaxrate','pupiltecherratio','%black','lowerstatpop','medvhome']
elif which == 'iris':
    X = load_iris()['data']
    y = np.reshape(load_iris()['target'],(-1,1))
    X = np.concatenate((X,y),axis = 1)
    feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'class']
elif which == 'data':
    with open('C:/Users/Sam/Dropbox/UniversityStuff/UCL/Project/Code/Project/data.csv', 'rt') as f:
        X = np.loadtxt(f, skiprows = 1, delimiter = ",")
elif which == 'stock':
    with open('C:/Users/Sam/Dropbox/UniversityStuff/UCL/Project/Code/Project/dataSP.csv', 'rt') as f:
        X = np.loadtxt(f, delimiter=";")
elif which == 'synth':
    n = 1000
    X0 = np.reshape(stats.norm.rvs(size = n), (-1,1))
    X1 = np.reshape(stats.norm.rvs(size = n), (-1,1))
    X2 = np.reshape(stats.norm.rvs(size = n), (-1,1))
    X3 = X0 + X1 + np.reshape(stats.norm.rvs(size = n), (-1,1))
    X4 = X3 + X2 + np.reshape(stats.norm.rvs(size = n), (-1,1))
    X5 = X0 + X4 + np.reshape(stats.norm.rvs(size = n), (-1,1))
    X6 = X5 + np.reshape(stats.norm.rvs(size = n), (-1,1))
    X = np.concatenate((X0,X1,X2,X3,X4,X5,X6), axis = 1)

pc_algorithm.find_dag(X, confidence = 0.1, whichseed=1).pc_dag()


importlib.reload(support), importlib.reload(compare), importlib.reload(combine), importlib.reload(estimate)

estimate.find_neighbours(X, confidence = 0.05, feature_names=feature_names, method = 'stacking')

estimate.find_neighbours(X, confidence = 0.05, method = 'stacking')

estimate.find_neighbours(X)

compare.pred_indep(X[:,0:1], X[:,1:15])
from sklearn.datasets import load_boston, load_iris
from sklearn import linear_model
from sklearn import tree, naive_bayes, ensemble
from sklearn import dummy, preprocessing
from sklearn import metrics
import importlib
from SupervisedGM import support, combine, estimate, compare
import numpy as np
from scipy import stats
import time
from sklearn.multioutput import MultiOutputEstimator
import networkx as nx
import matplotlib.pyplot as plt

## bost, iris, data, stock, synth
which = 'bost'

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
    n = 5000
    X1 = np.reshape(stats.norm.rvs(size = n), (-1,1))
    X2 = np.reshape(stats.norm.rvs(size = n), (-1,1))
    X3 = np.reshape(stats.norm.rvs(size = n), (-1,1))
    X4 = X1 + X2 + np.reshape(stats.norm.rvs(size = n), (-1,1))
    X5 = X4 + X3 + np.reshape(stats.norm.rvs(size = n), (-1,1))
    X6 = X1 + X5 + np.reshape(stats.norm.rvs(size = n), (-1,1))
    X7 = X6 + np.reshape(stats.norm.rvs(size = n), (-1,1))
    X = np.concatenate((X1,X2,X3,X4,X5,X6,X7), axis = 1)

importlib.reload(support), importlib.reload(compare), importlib.reload(combine), importlib.reload(estimate)
estimate.find_neighbours(X, confidence = 0.05, feature_names=feature_names, method = 'stacking')

compare.compare_methods(stats.norm.rvs(loc=10, scale=5, size = 10000),stats.norm.rvs(loc=11, scale=1, size = 10000)).loss_statistics().loss_se
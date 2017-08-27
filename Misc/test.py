import importlib

import IndependenceTest
import StructureEstimation
import Support
import numpy as np
from scipy import stats
from sklearn.datasets import load_boston, load_iris

from PCIT import MetaEstimator

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
elif which == 'wine':
    with open('C:/Users/Sam/Dropbox/UniversityStuff/UCL/Project/Data/Wine.csv', 'rt') as f:
        X = np.loadtxt(f, delimiter=";")
    feature_names = ['classes', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',
                     'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity',
                     'Hue', 'OD280 / OD315 of diluted wines', 'Proline']
elif which == 'glass':
    with open('C:/Users/Sam/Dropbox/UniversityStuff/UCL/Project/Data/glass.csv', 'rt') as f:
        X = np.loadtxt(f, delimiter=",", skiprows = 1)

importlib.reload(Support), importlib.reload(IndependenceTest), importlib.reload(MetaEstimator), importlib.reload(
    StructureEstimation)

StructureEstimation.find_neighbours(X, confidence = 0.1, estimator=MetaEstimator.MetaEstimator(method = None))


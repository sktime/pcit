from sklearn.datasets import load_boston, load_iris
from sklearn import linear_model, preprocessing
from sklearn import tree, naive_bayes, ensemble, svm
from sklearn import dummy, preprocessing
from sklearn import metrics, model_selection
import numpy as np
from scipy import stats
import time
from sklearn.multioutput import MultiOutputEstimator
import networkx as nx
import matplotlib.pyplot as plt
from compare import pred_indep
from combine import MetaEstimator
from sklearn.preprocessing import scale

with open('C:/Users/Sam/Dropbox/UniversityStuff/UCL/Project/Data/Wine.csv', 'rt') as f:
    Wine = np.loadtxt(f, delimiter=";")


n = Wine.shape[0]

X1 = Wine[:,1:2]
X2 = Wine[:,2:3]
noise = Wine[:,5:6]

n_range = [100,200,500,750,1000,2000,3000,4000,5000]
B = 500
power = []
time_sample_size = []

for sample_size in n_range:
    mistakes = 0
    tic = time.time()
    for i in range(B):
        X1_round = X1[stats.randint.rvs(low = 0, high = n, size = sample_size)]
        X2_round = X2[stats.randint.rvs(low = 0, high = n, size = sample_size)]
        noise_round = np.multiply(np.reshape((stats.uniform.rvs(size = sample_size) > 0.5) * 2 - 1,(-1,1)),
              np.sqrt(noise[stats.randint.rvs(low = 0, high = n, size = sample_size)]))
        Z = np.log(X1_round)*np.exp(X2_round) + noise_round


        X1_round = scale(X1_round)
        X2_round = scale(X2_round)
        Z = scale(Z)

        temp, temp, indep, temp = pred_indep(X1_round, X2_round, z = Z)


        mistakes += indep[0]
        print(sample_size, i)
    power.append(1 - mistakes / B)
    time_sample_size.append(time.time() - tic)


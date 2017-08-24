from compare import pred_indep
from combine import MetaEstimator
import numpy as np
from scipy import stats
import time

np.random.seed(1)

B = 500
p_noise = 5
n_range = np.arange(5,10,0.5)
power = []
time_round = []

for j in n_range:
    n = np.round(np.exp(j)).astype(int)
    mistakes = 0
    tic = time.time()
    for i in range(B):
        scale_X1 = np.sqrt(j)
        scale_noise = np.sqrt(j)
        X2 = np.reshape(stats.gamma.rvs(a = 5, scale = 1, size = n),(-1,1))
        X3 = np.reshape(stats.gamma.rvs(a = 5, scale = 1, size = n),(-1,1))
        X1 = np.reshape(stats.gamma.rvs(a = 5, scale = scale_X1, size = n),(-1,1)) + X2 + X3
        noise = np.reshape(stats.gamma.rvs(a = 5, scale = scale_noise, size = n * p_noise),(-1,p_noise))
        mistakes += pred_indep(X2,X3,z = np.concatenate((X1, noise), axis = 1),
                           estimator=MetaEstimator(method_type = 'regr', method=None))[2][0]
    power.append(1 - mistakes / B)
    time_round.append((time.time() - tic) / B)
    print(j)

power_variance = power * (np.ones(len(power)) - power) / np.sqrt(n_range)


from estimate import find_neighbours
from combine import MetaEstimator
import numpy as np
from scipy import stats
from sklearn import metrics
import time

def random_gauss(size_mat=10, sparse=0.2, n=1000, thresh = 0.1):
    mat = np.reshape(np.zeros(size_mat ** 2), (size_mat, size_mat))
    for i in range(size_mat):
        mat[i, i] = 1
    for i in range(size_mat):
        for j in range(i + 1, size_mat):
            draw = stats.uniform.rvs()
            if draw < sparse:
                pass
            else:
                mat[i, j] = stats.uniform.rvs(size=1, loc = 0, scale= min(2 - np.sum(mat[i, :]), 2 - np.sum(mat[:, j])))
                mat[j, i] = mat[i, j]
    mat = np.multiply(mat, mat > thresh)
    cov_mat = np.linalg.inv(mat)

    samples = stats.multivariate_normal.rvs(size=n, cov=cov_mat)

    return mat, samples
def smoother(arr):
    n = len(arr)
    new_arr = np.copy(arr).astype(float)
    new_arr[1] = np.mean(arr[0:3])
    for i in range(2,n-2):
        new_arr[i] = np.mean(arr[(i-2):(i+3)])
    new_arr[n-1] = np.mean(arr[n-2:n])
    return new_arr

n_list = np.round(np.exp(list(np.arange(6,10,0.1)))).astype(int)
size_mat = 10
B = 10

np.random.seed(0)

conf_mats = np.ndarray([len(n_list),2,2,B])
time1 = np.ndarray(len(n_list))

idx = 0
for n in n_list:
    conf_mat = [[0,0],[0,0]]
    time_round = 0
    for i in range(B):
        tic = time.time()
        part_cor, X = random_gauss(size_mat = size_mat, n = n, sparse = 0.20, thresh = 0.1)

        skeleton, skeleton_adj = find_neighbours(X, estimator = MetaEstimator(method = 'stacking'))

        conf_mat_round = metrics.confusion_matrix(np.reshape(skeleton_adj,(-1,1)), np.reshape(part_cor > 0, (-1,1)))
        conf_mat_round[1,1] = conf_mat_round[1,1] - size_mat
        conf_mat_round / 2


        conf_mats[idx,:,:,i] = conf_mat_round
        print(i)
    idx += 1
    print(idx)

## Plots

fdr = conf_mats[:,1,0,:] / np.sum(conf_mats[:,1,:,:], axis = 1) ## FDR

pwr = conf_mats[:,1,1,:] / np.sum(conf_mats[:,:,1,:], axis = 1) ## Power

fdr_var = np.var(fdr, axis = 1)
pwr_var = np.var(pwr, axis = 1)

fdr = np.mean(fdr, axis = 1)
pwr = np.mean(pwr, axis = 1)


plt.figure(figsize=(5,3))
plt.xscale('log')
plt.xlabel('n')
fdrline = plt.errorbar(n_list, fdr, yerr = 2.576 * fdr_var, color = 'red')
pwrline = plt.errorbar(n_list, pwr, yerr = 2.576 * pwr_var, color = 'blue')
smoothfdrline, = plt.plot(n_list, smoother(pwr),'--', color = 'blue')
plt.plot((np.min(n_list), np.max(n_list)), (0.05, 0.05), '--')
plt.title('Power curve and FDR for increasing sample size')
plt.legend([fdrline, pwrline, smoothfdrline],['FDR','Power','Power (smoothed)'])
plt.xticks([500, 1000, 2500 ,5000,10000,20000],[500, 1000, 2500,5000,10000,20000])
plt.show()
from estimate import find_neighbours
import numpy as np
from scipy import stats
from sklearn import metrics
import time
import matplotlib.pyplot as plt

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

n_list = np.round(np.exp(list(np.arange(6,10,0.1)))).astype(int)
size_mat = 10

np.random.seed(0)

conf_mats = np.ndarray([len(n_list),2,2])
time1 = np.ndarray(len(n_list))

idx = 0
B = 10
for n in n_list:
    conf_mat = [[0,0],[0,0]]
    time_round = 0
    for i in range(B):
        tic = time.time()
        part_cor, X = random_gauss(size_mat = size_mat, n = n, sparse = 0.20, thresh = 0.1)

        skeleton, skeleton_adj = find_neighbours(X, method = None, confidence=0.05)

        conf_mat_round = metrics.confusion_matrix(np.reshape(skeleton_adj,(-1,1)), np.reshape(part_cor > 0, (-1,1)))
        conf_mat_round[1,1] = conf_mat_round[1,1] - size_mat
        conf_mat_round / 2

        time_round += time.time() - tic

        conf_mat += conf_mat_round
        print(i)

    conf_mats[idx,:,:] = conf_mat / B
    time1[idx] = time_round / B
    idx += 1
    print(idx)

## Plots

def smoother(arr):
    n = len(arr)
    new_arr = np.copy(arr).astype(float)
    new_arr[1] = np.mean(arr[0:3])
    for i in range(2,n-2):
        new_arr[i] = np.mean(arr[(i-2):(i+3)])
    new_arr[n-1] = np.mean(arr[n-2:n])
    return new_arr

fdr = conf_mats[:,1,0] / np.sum(conf_mats[:,1,:], axis = 1) ## FDR

pwr = conf_mats[:,1,1] / np.sum(conf_mats[:,:,1], axis = 1) ## Power

pwr_smooth = smoother(pwr)
fdr_smooth = smoother(fdr)
time_smooth = smoother(time1)
time_smooth = time_smooth / np.max(time_smooth)

plt.figure(figsize=(5,3))
plt.plot(n_list, pwr_smooth, color='blue', lw=2, label = 'a')
plt.plot(n_list, fdr_smooth, color='red', lw=2)
plt.plot(n_list, time_smooth, color='green', lw=2)
plt.plot((np.min(n_list), np.max(n_list)), (0.05, 0.05), '--')
plt.xscale('log')
plt.xlabel('n')
plt.title('Power curve and FDR for increasing sample size')
plt.legend(['Power','FDR', 'Time'])
plt.xticks([500, 1000, 2500 ,5000,10000,20000],[500, 1000, 2500,5000,10000,20000])
plt.show()

none = np.load('05082017none.npy')
stack = np.load('05082017stacking.npy')
mplx = np.load('05082017multiplexing.npy')

fdr_none = smoother(none[:,1,0] / np.sum(none[:,1,:], axis = 1)) ## FDR
pwr_none = smoother(none[:,1,1] / np.sum(none[:,:,1], axis = 1)) ## Power

fdr_stack = smoother(stack[:,1,0] / np.sum(stack[:,1,:], axis = 1)) ## FDR
pwr_stack = smoother(stack[:,1,1] / np.sum(stack[:,:,1], axis = 1)) ## Power

fdr_mplx = smoother(mplx[:,1,0] / np.sum(mplx[:,1,:], axis = 1)) ## FDR
pwr_mplx = smoother(mplx[:,1,1] / np.sum(mplx[:,:,1], axis = 1)) ## Power

plt.figure(figsize=(14,7))
plt.plot(n_list, pwr_none, color='red', lw=2)
plt.plot(n_list, pwr_stack, color='blue', lw=2)
plt.plot(n_list, pwr_mplx, color='green', lw=2)
plt.plot(n_list, fdr_none, '--', color='red', lw=2)
plt.plot(n_list, fdr_stack, '--', color='blue', lw=2)
plt.plot(n_list, fdr_mplx, '--', color='green', lw=2)
plt.plot((np.min(n_list), np.max(n_list)), (0.05, 0.05), '--', color = 'black')
plt.xscale('log')
plt.xlabel('n')
plt.title('Power and FDR')
plt.legend(['No Ensembling','Stacking', 'Multiplexing'])
plt.xticks([500, 1000, 2500 ,5000,10000,20000],[500, 1000, 2500,5000,10000,20000])
plt.show()
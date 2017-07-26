from estimate import find_neighbours
import numpy as np
from scipy import stats
from sklearn import metrics
import time

def random_gauss(size_mat=10, sparse=0.2, n=1000, thresh = 0.2):
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

    which_discrete = (stats.uniform.rvs(size = size_mat) < 0.3) * 1

    for i in range(size_mat):
        if which_discrete[i] == 1:
            no_values = np.random.randint(5)
            new_data = np.zeros(n)
            for j in range(no_values):
                quantile = np.percentile(samples[:,i],100 * (j + 1) / (no_values + 1))
                new_data = new_data + (samples[:,i] > quantile)
            samples[:, i] = new_data

    return mat, samples

n_list = np.round(np.exp(list(np.arange(5,10,0.1)))).astype(int)
size_mat_list = range(5,20)
sparse_list = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

truetrue = np.ndarray([len(n_list),len(size_mat_list),len(sparse_list)])
truefalse = np.ndarray([len(n_list),len(size_mat_list),len(sparse_list)])
falsetrue = np.ndarray([len(n_list),len(size_mat_list),len(sparse_list)])
falsefalse = np.ndarray([len(n_list),len(size_mat_list),len(sparse_list)])
timetorun = np.ndarray([len(n_list),len(size_mat_list),len(sparse_list)])

p, q, r = 0, 0, 0
for n in n_list:
    q = 0
    for size_mat in size_mat_list:
        r = 0
        for sparse in sparse_list:
            tic = time.time()
            part_cor, X = random_gauss(size_mat = size_mat, n = n, sparse = sparse)

            skeleton, skeleton_adj = find_neighbours(X, method = 'multiplexing', confidence=0.2)

            conf_mat = metrics.confusion_matrix(np.reshape(skeleton_adj,(-1,1)), np.reshape(part_cor > 0, (-1,1)))
            conf_mat[1,1] = conf_mat[1,1] - size_mat
            conf_mat
            truetrue[p,q,r] = conf_mat[0,0]
            falsefalse[p,q,r] = conf_mat[1,1]
            truefalse[p,q,r] = conf_mat[0,1]
            falsetrue[p,q,r] = conf_mat[1,0]
            timetorun[p,q,r] = time.time() - tic
            print('FDR for n =',n,': ',conf_mat[1,0] / conf_mat[1,1])
            print('time: ', timetorun[p,q,r])
            r = r + 1
        q = q + 1
    r = r + 1
np.save('truetrue', truetrue)
np.save('falsefalse', falsefalse)
np.save('truefalse', truefalse)
np.save('falsetrue', falsetrue)
np.save('timetorun', timetorun)

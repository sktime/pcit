import numpy as np
from StructureEstimation import find_neighbours
# from Support import draw_graph_edgelabel

# with open('C:/Users/Sam/Dropbox/UniversityStuff/UCL/Project/Data/car.csv', 'rt') as f:
#    X = np.loadtxt(f, delimiter=";", skiprows = 1)

# Download auto mpg data set from https://archive.ics.uci.edu/ml/datasets/auto+mpg and store as X

feature_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year', 'origin']

n = X.shape[0]
B = 100

# Learnt structure on original data set
temp, skeleton_sum = find_neighbours(X, method ='stacking')

# Resamples
for i in range(B):
    # Resample indices (with replacement)
    idx = np.random.randint(n, size = n)

    # Learnt structure on resamples
    skeleton, skeleton_adj = find_neighbours(X[idx, :], method ='stacking', confidence = 0.1)

    # Sum over all skeletons to see how frequently a link is present
    skeleton_sum += skeleton_adj
    print(i)

# Using the graph tools requires the networkx and pyplot packages and a commented out function in Support

# draw_graph_edgelabel(skeleton_sum, feature_names)
# plt.hist(skeleton_sum)
# plt.show()
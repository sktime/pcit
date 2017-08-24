import matplotlib.pyplot as plt
import numpy as np

import estimate
import support

with open('C:/Users/Sam/Dropbox/UniversityStuff/UCL/Project/Data/car.csv', 'rt') as f:
    X = np.loadtxt(f, delimiter=";", skiprows = 1)

feature_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year', 'origin']

n = X.shape[0]
B = 100
temp, skeleton_sum = estimate.find_neighbours(X, method = 'stacking')
for i in range(B):
    idx = np.random.randint(n, size = n)
    skeleton, skeleton_adj = estimate.find_neighbours(X[idx,:], method = 'stacking', confidence = 0.1)
    skeleton_sum += skeleton_adj
    print(i)

support.draw_graph_edgelabel(skeleton_sum, feature_names)

plt.hist(skeleton_sum)
plt.show()
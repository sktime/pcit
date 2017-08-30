import time
import numpy as np
from scipy import stats
from pcit.IndependenceTest import pred_indep

np.random.seed(1)

# with open('C:/Users/Sam/Dropbox/UniversityStuff/UCL/Project/Data/Wine.csv', 'rt') as f:
#     Wine = np.loadtxt(f, delimiter=";")

# Download data set from https://archive.ics.uci.edu/ml/datasets/wine and store it as 'Wine'

n = Wine.shape[0]

# Extract data
X1 = Wine[:,1:2]
X2 = Wine[:,2:3]
noise = Wine[:,5:6]

# Sample sizes and number of resamples for test
n_range = [100,200,500,1000,2000,5000]
B = 500

power = []
time_sample_size = []

for sample_size in n_range:
    # Reset counters
    mistakes = 0
    tic = time.time()
    for i in range(B):
        # Sample with replacement from base arrays
        X1_round = X1[stats.randint.rvs(low = 0, high = n, size = sample_size)]
        X2_round = X2[stats.randint.rvs(low = 0, high = n, size = sample_size)]

        # Generate noise array
        noise_round = np.multiply(np.reshape((stats.uniform.rvs(size = sample_size) > 0.5) * 2 - 1,(-1,1)),
              np.sqrt(noise[stats.randint.rvs(low = 0, high = n, size = sample_size)]))

        # Calculate conditioning set, which makes X1_round and X2_round dependent
        Z = np.log(X1_round)*np.exp(X2_round) + noise_round

        # Independence test
        temp, indep, temp = pred_indep(X1_round, X2_round, z = Z)

        # If test made a mistake by attesting independence, update counter
        mistakes += indep[0]

        print('Sample size: ', sample_size, 'Resample round: ', i)

    power.append(1 - mistakes / B)
    time_sample_size.append((time.time() - tic) / 500)

# Calculate standard error (power follows a binomial)

SE = []
for i in range(len(n_range)):
    SE.append(np.sqrt(power[i] * (1 - power[i]) / np.sqrt(B)))
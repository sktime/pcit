from sklearn.datasets import load_boston
import pc_algorithm
import importlib

X = load_boston()['data']
n = 5
pc_algorithm.find_dag(X[:,0:n], confidence = 0.05).pc_dag()

importlib.reload(pc_algorithm)
from sklearn.datasets import load_boston
from pc_algorithm import find_dag
import importlib

X = load_boston()['data']
n = 10
find_dag(X[:,0:n], confidence = 0.9).pc_dag()

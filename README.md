# Predictive Conditional Independence Testing (PCIT)
## with applications in graphical model structure learning

## Description

This package implements a multivariate conditional independence test and an algorithm for learning directed graphs from data based on the PCIT

[arXiv preprint](https://arxiv.org/abs/1711.05869)

## Developers

[Samuel Burkart](https://github.com/SamBurkart): designated point of contact
[Franz Kiraly](https://github.com/fkiraly)

If you like to contribute, read our [contribution guide](https://github.com/alan-turing-institute/pcit/blob/master/contributing.md).

## Code Example
##### For details, see the [Examples](https://github.com/SamBurkart/pcit/blob/master/Example.ipynb)

There are 3 main functions:
- [MetaEstimator](https://github.com/SamBurkart/pcit/blob/master/pcit/MetaEstimator.py): Estimator class used for independence testing
- [PCIT](https://github.com/SamBurkart/pcit/blob/master/pcit/IndependenceTest.py): Multivariate Conditional Independence Test
- [find_neighbours](https://github.com/SamBurkart/pcit/blob/master/pcit/StructureEstimation.py): Undirected graph skeleton learning algorithm


For the following, X, Y and Z can be univariate or multivariate

```python
from sklearn.datasets import load_boston
data = load_boston()['data']
X = data[:,1:2]
Y = data[:,2:4]
Z = data[:,4:10]
```

##### Testing if X is independent of Y on a 0.01 confidence level

```python
PCIT(X, Y, confidence = 0.01)
```

The direction of the prediction is X -> Y, and as such the p-values correspond to the hypothesis that adding X does not improve the prediction of Y (one for each dimension in Y). If the parameter 'symmetric' is set to True (default), both directions are tested.
 
##### Testing if X is independent Y, conditional on Z
```python
PCIT(X, Y, z = Z)
```

##### Testing if X is independent of Y, conditional on Z, using a custom MetaEstimator, multiplexing over a manually chosen set of estimators:

```python
from sklearn.linear_model import RidgeCV, LassoCV,
                    SGDClassifier, LogisticRegression

regressors = [RidgeCV(), LassoCV()]
classifiers = [SGDClassifier(), LogisticRegression()]
custom_estim = MetaEstimator(method = 'multiplexing',
                estimators = (regressors, classifiers))

PCIT(X, Y, z = Z, estimator = custom_estim)
```

##### Learning the undirected graph with the undirected skeleton of X:

```python
X = load_boston()['data']
find_neighbours(X)
```

## Motivation

Conditional as well as multivariate independence testing are difficult problems lacking a straightforward, scalable and easy-to-use solution. This project connects the classical independence testing task to the supervised learning workflow. This has the following advantages:
- The link to the highly researched supervised learning workflow allows classical independence testing to grow its power as a side effect of the improvement in supervised learning methodology
- The sophisticated knowledge of hyperparameter-tuning in supervised prediction removes any need for hyperparameter tuning and manual choices prevalent in current methodology
- As a wrapper for the [sklearn](http://scikit-learn.org/stable/) package, the PCIT is easy to use and adjust

## Installation
Can be installed through pip

```python
pip install pcit
```

The dependencies are:
- [Scikit-learn](http://scikit-learn.org/stable/)
- [SciPy](https://scipy.org/)
- [MLXTEND](https://github.com/rasbt/mlxtend)

## Tests
Three tests can be run:

[Test_PCIT_Power](https://github.com/SamBurkart/pcit/blob/master/Tests/Test_PCIT_Power.py): Tests the power for increasing sample sizes on a difficult v-structured problem. Matlab code for same problem to compare with the "Kernel Conditional Independence Test" can be found [here](https://github.com/SamBurkart/pcit/blob/master/further/Test_KCIT_Power.m)

[Test_PCIT_Consistency](https://github.com/SamBurkart/pcit/blob/master/Tests/Test_PCIT_Consistency.py): Here the consistency under perturbations in the data is assessed.

[Test_Structure](https://github.com/SamBurkart/pcit/blob/master/Tests/Test_Structure.py): Here the power and false-discovery rate control of the graphical model structure learning algorithm are assessed

## License

MIT License

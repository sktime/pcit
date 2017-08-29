# Predictive Conditional Independence Testing (PCIT)
## with applications in graphical model structure learning

## Synopsis

This package implements a multivariate conditional independence independence test and an algorithm for learning directed graphs from data based on the PCIT

## Code Example

There are 3 main functions:
- [MetaEstimator](https://github.com/SamBurkart/pcit/blob/master/pcit/MetaEstimator.py): Estimator class used for independence testing
- [Pred_indep](https://github.com/SamBurkart/pcit/blob/master/pcit/IndependenceTest.py): Multivariate Conditional Independence Test
- [find_neighbours](https://github.com/SamBurkart/pcit/blob/master/pcit/StructureEstimation.py): Undirected graph skeleton learning algorithm

### Testing if X is independent of Y, conditional on Z

pred_indep(X, Y, z = Z)

### Testing if X is independent of Y, conditional on Z
### using a custom MetaEstimator, multiplexing over a manually chosen set of estimators:

```python
from sklearn.linear_model import RidgeCV, LassoCV,
                    SGDClassifier, LogisticRegression

regressors = [RidgeCV(), LassoCV()]
classifiers = [SGDClassifier(), LogisticRegression()]
custom_estim = MetaEstimator(method = 'multiplexing',
                estimators = (regressors, classifiers))

pred_indep(X, Y, z = Z, estimator = custom_estim)
```

### Learning the undirected graph with the undirected skeleton of X:

```python
find_neighbours(X)
```

## Motivation

A short description of the motivation behind the creation and maintenance of the project. This should explain **why** the project exists.

## Installation
Can be installed through pip

```python
pip install pcit
```

## API Reference

Depending on the size of the project, if it is small and simple enough the reference docs can be added to the README. For medium size to larger projects it is important to at least provide a link to where the API reference docs live.

## Tests

Describe and show how to run the tests with code examples.

## Contributors

Let people know how they can dive into the project, include important links to things like issue trackers, irc, twitter accounts if applicable.

## License

A short snippet describing the license (MIT, Apache, etc.)

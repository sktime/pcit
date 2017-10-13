import numpy as np
from mlxtend import classifier, regressor
from sklearn import naive_bayes, ensemble, linear_model, dummy, svm
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize

from pcit.Support import log_loss_resid


class MetaEstimator():
    '''
    This function implements the MetaEstimator class.  An estimator of the type MetaEstimator
    is a collection of methods and routines that are needed to automatically find optimal
    prediction functionals for prediction tasks. In particular, it combines automatically
    determining if the task is regression or classification, finding the optimal prediction
    functional, and implements routines to get the residuals  (which lack in sklearn)

    Functions:
        - get_estim: Fetch appropriate set of baseline estimators
        - fit: Fit the estimators and a training set
        - fit_baseline: Fit the uninformed baseline
        - predict: Predict on a test set
        - get_resid: Calculate the appropriate loss residuals for a training and test set

    ---------------------
    Attributes:
        - method: ensembling method [stacking (default), multiplexing, or None]

        - estimators: tuple with two lists of sklearn estimators, regression and classification
                        default is None, in which case predefined estimator lists are used

        - method_type: 'regr' or 'classif', default is None, which denotes automatic detection
                        if regression or classification problem

        - cutoff_categorical: if unique values in outcome are below this thre classification

    Returns:
        - Object containing fitted values, predictions, loss residuals etc (depending on function call)
    '''
    
    
    def __init__(self, method = 'stacking', estimators = None, method_type = None,
                 cutoff_categorical = 10):
        self.method = method
        self.estimators = estimators
        self.method_type = method_type
        self.cutoff_categorical = cutoff_categorical
        self.fitted = None
        self.predictions = None
        self.resid = None
        self.classes = None
        self.losses = []
        self.baseline = False
        self.isMetaEstimator = True

        if not self.method in ['stacking', 'multiplexing', None]:
            print('method needs to be "stacking", "multiplexing" or None')
            return

        if not self.method_type in ['regr', 'classif', None]:
            print('method_type needs to be "regr", "classif" or None')
            return

        if self.estimators is not None:
                if not type(estimators) is tuple and len(estimators) == 2:
                    print('custom estimators needs to be a tuple of 2 lists, ([regression estimators], '
                          '[classification estimators])')
                    return
                else:
                    for i in range(2):
                        if not type(estimators[i]) == list:
                            print('custom estimators needs to be a tuple of 2 lists, ([regression estimators], '
                                  '[classification estimators])')
                            return

        if not type(cutoff_categorical) == int:
            print('cutoff_categorical needs to be an integer')
            return

    def get_estim(self, y):
        '''
        Returns a list of estimators appropriate for the supervised learning problem.
        Distinctions are made between regression and classification problems, different sample
        sizes, and different types of output variables. When possible, seeds are set to achieve
        constant results
        '''

        self.estimators = []
        if self.method_type == 'regr':
            self.estimators.append(linear_model.ElasticNetCV(random_state=1, normalize=True))
            self.estimators.append(ensemble.GradientBoostingRegressor(random_state=1))
            self.estimators.append(ensemble.RandomForestRegressor(random_state=1))
            if y.shape[0] <= 5000:
                self.estimators.append(svm.SVR())

        else:
            if y.shape[0] < 50:
                if len(np.unique(y)) == 2:
                    self.estimators.append(naive_bayes.BernoulliNB())
                    self.estimators.append(linear_model.SGDClassifier(loss='log', random_state=1))
                else:
                    self.estimators.append(naive_bayes.MultinomialNB())
            self.estimators.append(naive_bayes.GaussianNB())
            self.estimators.append(ensemble.RandomForestClassifier(random_state=1))
            if y.shape[0] <= 5000:
                self.estimators.append(svm.SVC(probability=True))

    def fit(self, x, y):
        '''
        Fit method for the MetaEstimator. Output is a fitted estimator, that can then be used
        for prediction.
        '''
        
        # Determine if regression or classification problem, by comparing number of
        # unique values in output against threshold
        if self.method_type is None:
            is_above = len(np.unique(y, axis=0)) > self.cutoff_categorical
            self.method_type = ('classif','regr')[is_above]
        
        # Fetch the appropriate list of estimators
        if self.estimators is None:
            if self.method is not None:
                self.get_estim(y)
            else:
                if self.method_type == 'regr':
                    self.estimators = linear_model.LassoCV(normalize=True)
                elif self.method_type == 'classif':
                    self.estimators = ensemble.RandomForestClassifier(random_state=1)
        else:
            if self.method_type == 'regr':
                self.estimators = self.estimators[0]
            elif self.method_type == 'classif':
                self.estimators = self.estimators[1]

        # Collect information on classes in training set (needed later)
        if self.method_type == 'classif':
            self.classes = dummy.DummyClassifier().fit(x, y).classes_

        # Fit according to respective ensembling method
        if self.method == 'stacking':
            if self.method_type == 'regr':
                self.fitted = regressor.StackingRegressor(regressors=self.estimators,
                                            meta_regressor=linear_model.LinearRegression()).fit(x, y)

            elif self.method_type == 'classif':
                self.fitted = classifier.StackingClassifier(classifiers=self.estimators,
                            meta_classifier=linear_model.LogisticRegression(random_state = 1)).fit(x, y)

        elif self.method == 'multiplexing':
            for i in self.estimators:
                self.losses.append(np.mean(cross_val_score(i, x, y)))
            # For multiplexing, cross validation scores determine which estimator is chosen
            self.fitted  = self.estimators[np.argmin(self.losses)].fit(x, y)

        else:
            self.fitted = self.estimators.fit(x, y)

        return self

    def fit_baseline(self, x, y):
        '''
        Fit the baseline for the MetaEstimator. That is, depending on the loss function, determine
        the optimal constant predictor, based on the training data on the output
        '''
        
        # Determine if regression or classification problem
        if self.method_type is None:
            is_above = len(np.unique(y, axis=0)) > self.cutoff_categorical
            self.method_type = ('classif','regr')[is_above]
        
        # Fit a Dummy (constant) estimator
        if self.method_type == 'regr':
            self.fitted = dummy.DummyRegressor().fit(x, y)
        else:
            self.fitted = dummy.DummyClassifier().fit(x, y)
            self.classes = dummy.DummyClassifier().fit(x, y).classes_

    def predict(self, x):
        '''Make predictions on a new set of data x using the fitted MetaEstimator'''
        if self.fitted == None:
            error('model needs to be fitted before predictions can be made')
            
        if self.method_type == 'regr':
            self.predictions = self.fitted.predict(x)
        else:
            self.predictions = self.fitted.predict_proba(x)

        return self.predictions

    def get_resid(self, x_train, x_test, y_train, y_test, baseline = False):
        '''Returns the residuals for the prediction. To avoid excess code, this is
        called directly on the unfitted estimator'''

        self.baseline = baseline
        if baseline == False:
            self.fit(x_train,y_train)
            if self.method_type == 'regr':
                self.resid = np.power(self.predict(x_test) - y_test, 2)
                # Squared loss residuals
            else:
                self.resid = log_loss_resid(self.fitted, self.predict(x_test), y_test, self.classes, baseline)
                # Log loss residuals

        else:
            self.fit_baseline(x_train, y_train)
            if self.method_type == 'regr':
                self.resid = np.power(self.predict(x_test) - y_test, 2)
                # Squared loss residuals
            else:
                self.resid = log_loss_resid(self.fitted, self.predict(x_test), y_test, self.classes, baseline)
                #Log loss residuals

        return self.resid
